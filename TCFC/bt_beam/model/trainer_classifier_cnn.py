#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import os
import random
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import math
import torch.tensor
from .dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss, SoftCrossEntropy
from .gumbel import gumbel_softmax, gumbel_temperature
from .filtering import top_k_top_p_filtering
import tqdm
from metrics.metrics_recorder import MetricsRecorder
from metrics.eval_distinct import eval_distinct
from metrics.eval_bleu import eval_bleu
from metrics.eval_f1 import eval_f1
from metrics.eval_ppl import eval_ppl


class ClsTrainer:
    def __init__(self, model, vocab,
                 train_dataset, valid_dataset,
                 config, log_dir, logger, device=torch.device('cuda'),
                 ignore_idxs=[], distributed=False, valid_writer=None):
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.vocab = vocab
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train_classifier'))
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid_classifier'))
        else:
            self.valid_writer = valid_writer
        self.ignore_idxs = ignore_idxs
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        base_optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(config.embeddings_size, 0.1, 1.0, base_optimizer)

        self.distributed = distributed
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed and train_dataset is not None else None
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if distributed and valid_dataset is not None else None

        # torch.multiprocessing.set_start_method('spawn', force=True)
        self.train_dataloader = DataLoader(train_dataset, sampler=self.train_sampler,
                                           batch_size=config.batch_size,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(vocab.pad_id))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size,
                                           sampler=self.valid_sampler,
                                           num_workers=config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(vocab.pad_id))

    def reload_valid_data(self, valid_dataset):
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset) if self.distributed and valid_dataset is not None else None
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.config.batch_size,
                                           sampler=self.valid_sampler,
                                           num_workers=self.config.n_jobs, pin_memory=True,
                                           collate_fn=PadBatchSeq(self.vocab.pad_id))

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        distributed = self.rank != -1
        sm = state_dict['model']
        if (distributed):
            for k in list(sm.keys()):
                if k.startswith('transformer_module') and not k.startswith('transformer_module.module'):
                    sm['transformer_module.module' + k[len('transformer_module'):]] = sm[k]
                    sm.pop(k)
        else:
            for k in list(sm.keys()):
                if k.startswith('transformer_module.module'):
                    sm['transformer_module' + k[len('transformer_module.module'):]] = sm[k]
                    sm.pop(k)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch, after_step_funcs=[]):
        self.model.train()

        loss, acc, step_count = 0, 0, 0
        # self.logger.info('epoch %d, rank %d, before loop' % (epoch, self.rank))
        total = len(self.train_dataloader)
        for i, data in tqdm.tqdm(enumerate(self.train_dataloader), total=total):
            d_data = data

            text, style = d_data['text'].to(self.device), d_data['style'].to(self.device)
            text_len = d_data['text_len'].to(self.device)

            outputs = self.model(text[:, 1:], text_len - 1)
            batch_loss = self.criterion(outputs, style)
            batch_acc = (torch.argmax(outputs, dim=1) == style).float().mean()

            full_loss = batch_loss / self.config.batch_split
            full_loss.backward()

            loss += batch_loss.item()
            acc += batch_acc.item()
            step_count += 1

            # self.logger.info('epoch %d, batch %d' % (epoch, i))
            if (i + 1) % self.config.batch_split == 0:
                if self.config.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.config.clip_grad)
                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.optimizer.curr_step() % self.config.save_interval == 0:
                    for func in after_step_funcs:
                        func(self.optimizer.curr_step(), self.device)

                # shit log if you are node 0 in every step
                if self.rank == -1 or self.rank == 0:
                    loss /= step_count
                    acc /= step_count

                    self.train_writer.add_scalar('loss/loss', loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('acc/acc', acc, self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())
                    loss, acc, step_count = 0, 0, 0

                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config.eval_steps == 0:
                    self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate())

    def _eval_test(self, epoch, step, rate):
        recorder = MetricsRecorder(self.device, 'loss', 'acc')

        with torch.no_grad():
            self.model.eval()
            # self.logger.info("evaluating on rank {}, with datasize {}".format(self.rank, len(self.valid_dataloader)))

            for i, data in enumerate(self.valid_dataloader):
                d_data = data

                text, style = d_data['text'].to(self.device), d_data['style'].to(self.device)
                text_len = d_data['text_len'].to(self.device)

                outputs = self.model(text[:, 1:], text_len - 1)
                batch_loss = self.criterion(outputs, style)
                batch_acc = (torch.argmax(outputs, dim=1) == style).float().mean()

                recorder.metric_update(i, batch_loss, batch_acc)

        if self.rank != -1:
            recorder.all_reduce()

        # but only shit log if you are node 0
        if self.rank == -1 or self.rank == 0:
            recorder.add_to_writer(self.valid_writer, step)
            recorder.write_to_logger(self.logger, epoch, step)

        self.model.train()

    def test(self, epoch, step, rate):
        self._eval_test(epoch, step, rate)

    def predict(self):
        with torch.no_grad():
            self.model.eval()
            results = []
            # self.logger.info("evaluating on rank {}, with datasize {}".format(self.rank, len(self.valid_dataloader)))

            total = len(self.valid_dataloader)
            for i, data in tqdm.tqdm(enumerate(self.valid_dataloader), total=total):
                d_data = data

                text, style = d_data['text'].to(self.device), d_data['style'].to(self.device)
                text_len = d_data['text_len'].to(self.device)

                outputs = self.model(text[:, 1:], text_len - 1)
                outputs = torch.softmax(outputs, dim=-1)
                results += outputs[:, 1].reshape(-1).tolist()

        return results

    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            # self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate(), self.cls_optimizer.rate())
            # print(self._pred_sample_topk_topp(5, topp=0.9))
            # print(self._pred_sample_beam(5))
            # print(self._pred_sample_gumbel(5))
            self._eval_train(epoch, after_step_funcs=after_step_funcs)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)
