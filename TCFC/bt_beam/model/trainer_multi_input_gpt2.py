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
import fasttext
import json


class Trainer:
    def __init__(self, model, model_r2p,
                 train_dialogue_dataset, valid_dialogue_dataset, train_text_dataset,
                 config, log_dir, logger, device=torch.device('cuda'), distributed=False, valid_writer=None,
                 config_path=None):
        self.config = config
        self.config_path = config_path
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.valid_dialogue_dataset = valid_dialogue_dataset
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=60)
        if valid_writer is None:
            self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        else:
            self.valid_writer = valid_writer
        self.model = model.to(device)
        self.model_r2p = model_r2p.to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.tokenizer.vocab_size,
                                            smoothing=config.label_smoothing).to(device)
        self.ce_criterion = nn.CrossEntropyLoss(reduction='none').to(device)  # for calculation of ppl
        base_optimizer = Adam([{'params': self.model.parameters()},
                               {'params': self.model_r2p.parameters()}],
                              lr=config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.embeddings_size, 0.1, config.lr_warmup, base_optimizer)

        self.train_dialogue_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dialogue_dataset) if distributed and train_dialogue_dataset is not None else None
        self.valid_dialogue_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dialogue_dataset) if distributed and valid_dialogue_dataset is not None else None
        self.train_text_sampler = torch.utils.data.RandomSampler(train_text_dataset,
                                                                 replacement=True,
                                                                 num_samples=len(self.train_dialogue_sampler)) \
            if train_text_dataset is not None and self.train_dialogue_sampler is not None else None

        # torch.multiprocessing.set_start_method('spawn', force=True)
        self.train_dialogue_dataloader = DataLoader(train_dialogue_dataset, sampler=self.train_dialogue_sampler,
                                                    batch_size=config.batch_size, num_workers=config.n_jobs,
                                                    pin_memory=True, collate_fn=PadBatchSeq(0))
        self.valid_dialogue_dataloader = DataLoader(valid_dialogue_dataset, batch_size=config.batch_size,
                                                    sampler=self.valid_dialogue_sampler, num_workers=config.n_jobs,
                                                    pin_memory=True, collate_fn=PadBatchSeq(0))
        self.train_text_dataloader = DataLoader(train_text_dataset, batch_size=config.batch_size,
                                               sampler=self.train_text_sampler, num_workers=config.n_jobs,
                                               pin_memory=True, collate_fn=PadBatchSeq(0))

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'model_r2p': self.model_r2p.state_dict(),
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

        sm = state_dict['model_r2p']
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

        self.model_r2p.load_state_dict(state_dict['model_r2p'])

        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch, after_step_funcs=None):
        self.model.train()
        self.model_r2p.train()

        p2r_loss, r2p_loss, s1_bt_loss, step_count = 0, 0, 0, 0
        s1_bt_len = 0
        total = len(self.train_dialogue_dataloader)
        if not (self.rank == -1 or self.rank == 0):
            ITER = enumerate(zip(self.train_dialogue_dataloader, self.train_text_dataloader))
        else:
            ITER = tqdm.tqdm(enumerate(zip(self.train_dialogue_dataloader, self.train_text_dataloader)), total=total,
                             dynamic_ncols=True)
        for i, data in ITER:
            # for i, data in enumerate(self.train_dialogue_dataloader):
            d_data, t_data = data

            post, resp = d_data['post'].to(self.device), d_data['resp'].to(self.device)
            post_len, resp_len = d_data['post_len'].to(self.device), d_data['resp_len'].to(self.device)
            bs = resp.shape[0]
            style_0 = torch.zeros((bs,)).long().to(self.device)
            style_1 = torch.ones((bs,)).long().to(self.device)

            # p2r loss
            enc_contexts = list()
            enc_contexts.append(self.model.encode(post, post_len))

            output_logits = self.model.decode(resp[:, :-1], resp_len - 1, enc_contexts, style_ids=style_0)
            outputs = F.log_softmax(output_logits, dim=-1)

            batch_p2r_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), resp.view(-1))
            batch_p2r_loss = batch_p2r_loss.reshape([bs, -1])
            pad_mask = self.model.get_mask(resp_len)
            batch_p2r_loss = torch.sum(batch_p2r_loss * pad_mask) / pad_mask.sum()

            # r2p loss
            enc_contexts = list()
            enc_contexts.append(self.model_r2p.encode(resp, resp_len))

            output_logits = self.model_r2p.decode(post[:, :-1], post_len - 1, enc_contexts, style_ids=style_0)
            outputs = F.log_softmax(output_logits, dim=-1)

            batch_r2p_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), post.view(-1))
            batch_r2p_loss = batch_r2p_loss.reshape([bs, -1])
            pad_mask = self.model_r2p.get_mask(post_len)
            batch_r2p_loss = torch.sum(batch_r2p_loss * pad_mask) / pad_mask.sum()

            # bt loss, only s1
            bt_weight = 0
            if self.optimizer.curr_step() >= self.config.bt_freeze_step:
                bt_weight = self.config.bt_weight * min(1, (self.optimizer.curr_step() - self.config.bt_freeze_step) / (
                            self.config.lr_warmup - self.config.bt_freeze_step))

                text, text_len = t_data['text'].to(self.device), t_data['text_len'].to(self.device)
                bs = text.shape[0]
                style_0 = torch.zeros((bs,)).long().to(self.device)
                style_1 = torch.ones((bs,)).long().to(self.device)

                with torch.no_grad():
                    self.model_r2p.eval()
                    enc_text = list()
                    enc_text.append(self.model_r2p.encode(text, text_len))
                    fp_s1, fp_s1_len = self.model_r2p.beam_search(enc_text, style_ids=style_0, return_lens=True,
                                                                  beam_size=self.config.bt_beam_size)
                    self.model_r2p.train()

                max_len = max(fp_s1_len)
                fp_s1_len = torch.LongTensor(fp_s1_len).to(self.device)
                fp_s1 = torch.LongTensor(
                    [x + [0] * (max_len - len(x)) for x in fp_s1]).to(self.device)

                # --------------------TEST PRINT START-----------------------------
                if self.optimizer.curr_step() % 300 == 5 and (self.rank == -1 or self.rank == 0):
                    print(self.optimizer.curr_step())
                    for j in range(bs):
                        tx = self.ids2string(text[j], cut_head=False)
                        fp_s1_tx = self.ids2string(fp_s1[j], cut_head=False)
                        print("Text:", tx, "FP:", fp_s1_tx)
                # --------------------TEST PRINT END-----------------------------

                enc_fp_s1 = list()
                enc_fp_s1.append(self.model.encode(fp_s1, fp_s1_len))

                s1_bt_logits = self.model.decode(text[:, :-1], text_len - 1, enc_fp_s1, style_ids=style_1)    # dec_id = style_id for generating stylized dialogue
                s1_bt_log_softmax = F.log_softmax(s1_bt_logits, dim=-1)
                batch_s1_bt_loss = self.criterion(s1_bt_log_softmax.view(-1, s1_bt_log_softmax.shape[-1]), text.view(-1))
                batch_s1_bt_loss = batch_s1_bt_loss.reshape([bs, -1])   # [bs, text_seq_len]
                batch_s1_bt_mask = self.model.get_mask(text_len)
                batch_s1_bt_loss = (batch_s1_bt_loss * batch_s1_bt_mask).sum() / batch_s1_bt_mask.sum()

            else:
                batch_s1_bt_loss = 0
                fp_s1_len = 0

            p2r_loss += batch_p2r_loss.item()
            r2p_loss += batch_r2p_loss.item()
            if type(batch_s1_bt_loss) is int:
                s1_bt_loss += 0
                s1_bt_len += 0
            else:
                s1_bt_loss += batch_s1_bt_loss.item()
                s1_bt_len += torch.max(fp_s1_len.float())
            step_count += 1

            full_loss = (batch_p2r_loss + batch_r2p_loss + bt_weight * batch_s1_bt_loss) / self.config.batch_split
            full_loss.backward()

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
                    p2r_loss /= step_count
                    r2p_loss /= step_count
                    s1_bt_loss /= step_count
                    s1_bt_len /= step_count
                    ITER.set_postfix(p2r_loss=p2r_loss, r2p_loss=r2p_loss, s1_bt_loss=s1_bt_loss)

                    self.train_writer.add_scalar('loss/p2r_loss', p2r_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/r2p_loss', r2p_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s1_bt_loss', s1_bt_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s1_bt_len', s1_bt_len, self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())
                    p2r_loss, r2p_loss, s1_bt_loss, step_count = 0, 0, 0, 0
                    s1_bt_len = 0

                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config.eval_steps == 0:
                    self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate())

    def _eval_test(self, epoch, step, rate):
        recorder = MetricsRecorder(self.device, 's2s_loss', 's2s_loss2', 'ppl', 'ppl2')
        recorder.add_metric_groups('metric', 'metric', 'metric', 'metric')
        all_preds = dict()

        with torch.no_grad():
            self.model.eval()
            if not (self.rank == -1 or self.rank == 0):
                ITER = enumerate(self.valid_dialogue_dataloader)
            else:
                total = len(self.valid_dialogue_dataloader)
                ITER = tqdm.tqdm(enumerate(self.valid_dialogue_dataloader), total=total, dynamic_ncols=True)

            for i, d_data in ITER:

                post, resp = d_data['post'].to(self.device), d_data['resp'].to(self.device)
                style = d_data['style'].to(self.device)
                post_len, resp_len = d_data['post_len'].to(self.device), d_data['resp_len'].to(self.device)
                bs = post.shape[0]

                # s2s loss
                enc_contexts = list()
                enc_contexts.append(self.model.encode(post, post_len))

                prevs, nexts = resp[:, :-1].contiguous(), resp.contiguous()
                output_logits = self.model.decode(prevs, resp_len - 1, enc_contexts, style_ids=style)
                outputs = F.log_softmax(output_logits, dim=-1)
                batch_s2s_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                batch_s2s_loss = batch_s2s_loss.reshape([bs, -1])
                pad_mask = self.model.get_mask(resp_len)

                s2s_mask = torch.eq(style, 0).reshape([-1, 1])
                s2s_mask2 = torch.eq(style, 1).reshape([-1, 1])

                s2s_loss1 = torch.sum(batch_s2s_loss * s2s_mask * pad_mask) / ((pad_mask * s2s_mask).sum() + 1e-5)
                s2s_loss2 = torch.sum(batch_s2s_loss * s2s_mask2 * pad_mask) / ((pad_mask * s2s_mask2).sum() + 1e-5)

                ppl_s2s_loss = self.ce_criterion(output_logits.view(-1, output_logits.shape[-1]), nexts.view(-1))
                ppl_s2s_loss = ppl_s2s_loss.reshape([bs, -1])
                batch_ppl = eval_ppl(ppl_s2s_loss, pad_mask)

                batch_ppl1 = torch.sum(batch_ppl * s2s_mask.view(-1)) / (s2s_mask.sum() + 1e-5)
                batch_ppl2 = torch.sum(batch_ppl * s2s_mask2.view(-1)) / (s2s_mask2.sum() + 1e-5)
                # print(batch_ppl1.item(), batch_ppl2.item())

                top_k = self.config.annealing_topk
                prediction, lens = self.model.top_k_top_p_search(enc_contexts, top_k=top_k, style_ids=style,
                                                                 sample_size=1)
                prediction = prediction[:, 1:]
                lens = lens - 1
                # prediction = self.model.beam_search(enc_contexts, style_ids=style)

                for j in range(bs):
                    st = style[j].item()

                    post_str = self.ids2string(post[j], cut_head=False)
                    resp_str = self.ids2string(resp[j], cut_head=False)
                    pred_str = self.ids2string(prediction[j], cut_head=False)
                    if post_str not in all_preds:
                        all_preds[post_str] = {'post': post_str,
                                               'resp_style0': [], 'resp_style1': [],
                                               'pred_style0': [], 'pred_style1': []}
                    all_preds[post_str]['resp_style%d' % st].append(resp_str)
                    all_preds[post_str]['pred_style%d' % st].append(pred_str)

                recorder.metric_update(i, s2s_loss1, s2s_loss2, batch_ppl1, batch_ppl2)

        if self.rank != -1:
            recorder.all_reduce()
            print('recorder', recorder.metrics)

        # but only shit log if you are node 0
        if self.rank == -1 or self.rank == 0:
            recorder.add_to_writer(self.valid_writer, step)
            recorder.write_to_logger(self.logger, epoch, step)

            # and only predicts sample on node 0
            writer_text = ''

            writer_text += 'topk  \n'
            sample_dialog = self._pred_sample_topk_topp(3, topk=self.config.annealing_topk)
            for j, d in enumerate(sample_dialog):
                self.logger.info('--epoch {} step{} topk sample {}--'.format(
                    epoch, self.optimizer.curr_step(), j))
                self.logger.info('post: {}'.format(d['post']))
                self.logger.info('resp: {}'.format(d['resp']))
                writer_text += 'Post: {}  \nResp: {}  \n'.format(d['post'], d['resp'])
                for style in d['pred']:
                    self.logger.info('target style: {}'.format(style))
                    writer_text += 'Style: {}  \n'.format(style)
                    for utter in d['pred'][style]:
                        self.logger.info(utter)
                        writer_text += '{}  \n'.format(utter)
                writer_text += '  \n'

            writer_text += 'topp  \n'
            sample_dialog = self._pred_sample_topk_topp(3, topp=self.config.annealing_topp)
            for j, d in enumerate(sample_dialog):
                self.logger.info('--epoch {} step{} topp sample {}--'.format(
                    epoch, self.optimizer.curr_step(), j))
                self.logger.info('post: {}'.format(d['post']))
                self.logger.info('resp: {}'.format(d['resp']))
                writer_text += 'Post: {}  \nResp: {}  \n'.format(d['post'], d['resp'])
                for style in d['pred']:
                    self.logger.info('target style: {}'.format(style))
                    writer_text += 'Style: {}  \n'.format(style)
                    for utter in d['pred'][style]:
                        self.logger.info(utter)
                        writer_text += '{}  \n'.format(utter)
                writer_text += '  \n'

            writer_text += 'beam  \n'
            sample_dialog = self._pred_sample_beam(3)
            for j, d in enumerate(sample_dialog):
                self.logger.info('--epoch {} step{} topp sample {}--'.format(
                    epoch, self.optimizer.curr_step(), j))
                self.logger.info('post: {}'.format(d['post']))
                self.logger.info('resp: {}'.format(d['resp']))
                writer_text += 'Post: {}  \nResp: {}  \n'.format(d['post'], d['resp'])
                for style in d['pred']:
                    self.logger.info('target style: {}'.format(style))
                    writer_text += 'Style: {}  \n'.format(style)
                    for utter in d['pred'][style]:
                        self.logger.info(utter)
                        writer_text += '{}  \n'.format(utter)
                writer_text += '  \n'

            self.valid_writer.add_text('dialog', writer_text, self.optimizer.curr_step())
        self.model.train()
        self.model_r2p.train()

        if self.config_path is not None:
            eval_dir = os.path.join(self.config_path, self.config['eval_dir'])
            with open(os.path.join(eval_dir, 'step%d' % step + self.config.pred_path), 'w', encoding='utf8') as f:
                for ps in all_preds:
                    json.dump(all_preds[ps], f)
                    f.write('\n')


    def ids_cut(self, ids, length=None, cut_head=True):
        """
        Cut a list of ids to its real length. (ids[0] & eos would be cut)
        :param ids: A list of ids. Note: ids[0] would be ignored.
        :param length: Length of ids including eos (but eos would be cut). If is None,
            length would be inferred from ids.
        :return: Result id list.
        """
        if type(ids) is not list:
            ids = ids.tolist()
        if length is None:
            try:
                length = ids[1:].index(self.model.tokenizer.eos_token_id) + 1
            except ValueError:
                length = len(ids)
        if cut_head:
            return ids[1: length]
        else:
            return ids[0: length]

    def ids2string(self, ids, length=None, cut_head=True):
        """
        :param ids: A list of ids. Note: ids[0] would be ignored.
        :param length: Length of ids including eos (but eos would not be translated). If is None,
            length would be inferred from ids.
        :return: Result string
        """
        return self.model.tokenizer.decode(self.ids_cut(ids, length, cut_head=cut_head))

    def _pred_sample_topk_topp(self, n_sample, topk=0, topp=0.0):
        with torch.no_grad():
            self.model.eval()
            samples_idxs_single = random.sample(range(len(self.valid_dialogue_dataset)), n_sample)
            samples_idxs = []
            for idx in samples_idxs_single:
                samples_idxs += [idx] * self.config.n_styles
            samples = PadBatchSeq(0)([self.valid_dialogue_dataset[idx] for idx in samples_idxs])
            styles = torch.tensor([i for i in range(self.config.n_styles)] * n_sample).long().to(self.device)
            enc_contexts = [self.model.encode(c, c_len) for c, c_len in [[samples['post'].to(self.device),
                                                                          samples['post_len'].to(self.device)]]]
            prediction, lens = self.model.top_k_top_p_search(enc_contexts, top_k=topk, top_p=topp, style_ids=styles)
            res = []
            for j in range(n_sample):
                post_str = self.ids2string(samples['post'][j * self.config.n_styles], cut_head=False)
                resp_str = self.ids2string(samples['resp'][j * self.config.n_styles], cut_head=False)
                pred = {}
                for style in range(self.config.n_styles):
                    utters = []
                    for i in range(self.config.top_p_top_k_sample_size):
                        idx = j * self.config.n_styles * self.config.top_p_top_k_sample_size + \
                              style * self.config.top_p_top_k_sample_size + i
                        utter = prediction[idx].tolist()
                        utter_len = lens[idx].item()
                        utters.append(self.ids2string(utter, utter_len - 1, cut_head=True))
                    pred[style] = utters
                res.append({"post": post_str, "resp": resp_str, "pred": pred})

        return res

    def _pred_sample_beam(self, n_sample, specified=None):
        with torch.no_grad():
            self.model.eval()
            if specified:
                n_sample = len(specified)
                samples_idxs_single = specified
            else:
                samples_idxs_single = random.sample(range(len(self.valid_dialogue_dataset)), n_sample)
            samples_idxs = []
            for idx in samples_idxs_single:
                samples_idxs += [idx] * self.config.n_styles
            samples = PadBatchSeq(0)([self.valid_dialogue_dataset[idx] for idx in samples_idxs])
            styles = torch.tensor([i for i in range(self.config.n_styles)] * n_sample).long().to(self.device)
            enc_contexts = [self.model.encode(c, c_len) for c, c_len in [[samples['post'].to(self.device),
                                                                          samples['post_len'].to(self.device)]]]
            prediction = self.model.beam_search(enc_contexts, style_ids=styles)
            res = []
            for j in range(n_sample):
                post_str = self.ids2string(samples['post'][j * self.config.n_styles], cut_head=False)
                resp_str = self.ids2string(samples['resp'][j * self.config.n_styles], cut_head=False)
                pred = {}
                for style in range(self.config.n_styles):
                    utters = []
                    idx = j * self.config.n_styles + style
                    utter = prediction[idx]
                    utters.append(self.ids2string(utter, cut_head=False))
                    pred[style] = utters
                res.append({"post": post_str, "resp": resp_str, "pred": pred})

        return res

    def test(self, epoch, step, rate):
        self._eval_test(epoch, step, rate)

    def pred(self, sample_size=1):
        """
        Do prediction on train data.
        """
        predictions = []
        with torch.no_grad():
            self.model.eval()
            total = len(self.train_dialogue_dataloader)
            for i, data in tqdm.tqdm(enumerate(self.train_dialogue_dataloader), total=total):
                d_data = data

                post = d_data['post'].to(self.device)
                bs = len(post)

                enc_contexts = list()
                enc_contexts.append(self.model.encode(post))

                styles = torch.ones(bs).long().to(self.device)
                top_p = self.config.annealing_topp
                prediction, lens = self.model.top_k_top_p_search(enc_contexts, top_p=top_p, styles=styles,
                                                                 sample_size=sample_size)

                for j in range(bs):
                    post_str = self.ids2string(post[j])
                    for k in range(sample_size):
                        pred_str = self.ids2string(prediction[j * sample_size + k], lens[j * sample_size + k] - 1)
                        predictions.append((post_str, pred_str))

        return predictions

    def pred_loop(self):
        while True:
            idx = int(input('>>> '))
            sample_dialog = self._pred_sample_beam(0, specified=[idx])
            for j, d in enumerate(sample_dialog):
                self.logger.info('post: {}'.format(d['post']))
                self.logger.info('resp: {}'.format(d['resp']))
                for style in d['pred']:
                    self.logger.info('target style: {}'.format(style))
                    for utter in d['pred'][style]:
                        self.logger.info(utter)

    def train(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_dialogue_sampler:
                self.train_dialogue_sampler.set_epoch(epoch)
            # self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate(), self.cls_optimizer.rate())
            # print(self._pred_sample_topk_topp(5, topp=0.9))
            # print(self._pred_sample_beam(5))
            # print(self._pred_sample_gumbel(5))
            # self._eval_test(epoch, 0, 0)
            # self._eval_test(epoch, 100000, 0.0)
            # self.pred_loop()
            self._eval_train(epoch, after_step_funcs=after_step_funcs)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)

    def train_reverse(self, start_epoch, epochs, after_epoch_funcs=[], after_step_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_dialogue_sampler:
                self.train_dialogue_sampler.set_epoch(epoch)
            # self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate(), self.cls_optimizer.rate())
            # print(self._pred_sample_topk_topp(5, topp=0.9))
            # print(self._pred_sample_beam(5))
            # print(self._pred_sample_gumbel(5))
            # self._eval_test(epoch, 0, 0)
            # self._eval_test(epoch, 100000, 0.0)
            self._eval_train_reverse(epoch, after_step_funcs=after_step_funcs)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)

    def test_reverse(self, epoch, step, rate):
        return self._eval_test_reverse(epoch, step, rate)
