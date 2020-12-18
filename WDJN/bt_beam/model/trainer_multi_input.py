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
import torch.tensor
from .dataset import PadBatchSeq
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss, SoftCrossEntropy
from .filtering import top_k_top_p_filtering
import tqdm
from metrics.metrics_recorder import MetricsRecorder
from metrics.eval_distinct import eval_distinct
from metrics.eval_bleu import eval_bleu
from metrics.eval_f1 import eval_f1
from metrics.eval_ppl import eval_ppl


class Trainer:
    def __init__(self, model, model_fp, train_dialogue_dataset, train_text_dataset, valid_dialogue_dataset,
                 config, log_dir, logger, device=torch.device('cuda'), distributed=False):
        self.config = config
        self.device = device
        self.logger = logger
        self.log_dir = log_dir
        self.valid_dialogue_dataset = valid_dialogue_dataset
        self.rank = torch.distributed.get_rank() if distributed else -1
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), flush_secs=60)
        self.valid_writer = SummaryWriter(os.path.join(log_dir, 'valid'))
        self.fp_writer = SummaryWriter(os.path.join(log_dir, 'fake_post'))
        self.model = model.to(device)
        self.model_fp = model_fp.to(device)
        self.criterion = LabelSmoothingLoss(n_labels=len(self.model.vocab), smoothing=config.label_smoothing,
                                            ignore_index=self.model.vocab.pad_id).to(device)
        self.ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.model.vocab.pad_id).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.config.embeddings_size, 0.1, config.lr_warmup, base_optimizer)
        base_optimizer_fp = Adam(self.model_fp.parameters(), lr=config.lr, weight_decay=0.01)
        self.optimizer_fp = NoamOpt(self.model_fp.config.embeddings_size, 0.1, config.lr_warmup, base_optimizer_fp)

        self.train_dialogue_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dialogue_dataset) if distributed else torch.utils.data.RandomSampler(train_dialogue_dataset)

        self.train_text_sampler = torch.utils.data.RandomSampler(
            train_text_dataset, replacement=True,
            num_samples=int(len(self.train_dialogue_sampler) * config.text_batch_size / config.dialog_batch_size))

        self.valid_dialogue_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dialogue_dataset) if distributed else None

        self.train_dialogue_dataloader = DataLoader(
            train_dialogue_dataset, sampler=self.train_dialogue_sampler, batch_size=config.dialog_batch_size,
            num_workers=config.n_jobs, pin_memory=True, collate_fn=PadBatchSeq(self.model.vocab.pad_id))
        self.train_text_dataloader = DataLoader(
            train_text_dataset, sampler=self.train_text_sampler, batch_size=config.text_batch_size,
            num_workers=config.n_jobs, pin_memory=True, collate_fn=PadBatchSeq(self.model.vocab.pad_id))
        self.valid_dialogue_dataloader = DataLoader(
            valid_dialogue_dataset, batch_size=config.dialog_batch_size, num_workers=config.n_jobs,
            sampler=self.valid_dialogue_sampler, pin_memory=True, collate_fn=PadBatchSeq(self.model.vocab.pad_id))

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        distributed = self.rank != -1
        sm = state_dict['model']
        module_names = ['transformer_module']
        if distributed:
            for k in list(sm.keys()):
                for module_name in module_names:
                    if k.startswith(module_name) and not k.startswith(module_name + '.module'):
                        sm[module_name + '.module' + k[len(module_name):]] = sm[k]
                        sm.pop(k)
        else:
            for k in list(sm.keys()):
                for module_name in module_names:
                    if k.startswith(module_name + '.module'):
                        sm[module_name + k[len(module_name + '.module'):]] = sm[k]
                        sm.pop(k)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch, after_step_funcs=[]):
        self.model.train()

        p2r_loss, r2p_loss, s0_bt_loss, s1_bt_loss = 0, 0, 0, 0
        s0_bt_len, s1_bt_len = 0, 0
        step_count = 0
        total = len(self.train_dialogue_dataloader)
        if self.rank == -1 or self.rank == 0:
            ITER = tqdm.tqdm(enumerate(zip(self.train_dialogue_dataloader,
                                           self.train_text_dataloader)), dynamic_ncols=True, total=total)
        else:
            ITER = enumerate(zip(self.train_dialogue_dataloader, self.train_text_dataloader))

        for i, data in ITER:
            d_data, t_data = data

            post, resp = d_data['post'].to(self.device), d_data['resp'].to(self.device)
            post_len, resp_len = d_data['post_len'].to(self.device), d_data['resp_len'].to(self.device)
            style_id = d_data['style'].to(self.device)
            bs = post.shape[0]

            # p2r loss
            enc_post = list()
            enc_post.append(self.model.encode(post, post_len, enc_id=torch.zeros_like(style_id)))   # enc_id = 0 for posts

            p2r_logits = self.model.decode(resp[:, :-1], resp_len-1, enc_post, dec_id=style_id)   # dec_id = style_id for generating stylized response
            p2r_log_softmax = F.log_softmax(p2r_logits, dim=-1)
            batch_p2r_loss = self.criterion(p2r_log_softmax.view(-1, p2r_log_softmax.shape[-1]), resp.view(-1))
            batch_p2r_loss = batch_p2r_loss.reshape([bs, -1])   # [bs, resp_seq_len]
            batch_p2r_mask = self.model.get_mask(resp_len)
            batch_p2r_loss = (batch_p2r_loss * batch_p2r_mask).sum() / batch_p2r_mask.sum()
            p2r_loss += batch_p2r_loss.item()

            # r2p loss
            enc_resp = list()
            enc_resp.append(self.model_fp.encode(resp, resp_len, enc_id=torch.zeros_like(style_id)))   # enc_id = 1 for responses

            r2p_logits = self.model_fp.decode(post[:, :-1], post_len-1, enc_resp, dec_id=torch.zeros_like(style_id))   # dec_id = 2 for generating posts
            r2p_log_softmax = F.log_softmax(r2p_logits, dim=-1)
            batch_r2p_loss = self.criterion(r2p_log_softmax.view(-1, r2p_log_softmax.shape[-1]), post.view(-1))
            batch_r2p_loss = batch_r2p_loss.reshape([bs, -1])   # [bs, post_seq_len]
            batch_r2p_mask = self.model_fp.get_mask(post_len)
            batch_r2p_loss = (batch_r2p_loss * batch_r2p_mask).sum() / batch_r2p_mask.sum()
            r2p_loss += batch_r2p_loss.item()

            ((batch_r2p_loss + batch_p2r_loss)/ self.config.batch_split).backward()


            bt_weight = 0
            fp_s0, fp_s0_len, fp_s1, fp_s1_len = None, None, None, None
            text, text_len, text_style_id = None, None, None
            if self.optimizer.curr_step() >= self.config.bt_freeze_step:
                bt_weight = (self.config.bt_weight / 2) * (1 + min(1, (self.optimizer.curr_step() - self.config.bt_freeze_step) / (self.config.lr_warmup - self.config.bt_freeze_step)))
                # s0_back_trans
                # fp_s0, fp_s0_len = self.model_fp.beam_search_train_bt(enc_resp, dec_id=torch.zeros_like(style_id))
                #
                # fp_s0 = fp_s0[:, 1:]
                # fp_s0_len = fp_s0_len - 1
                #
                # enc_fp_s0 = list()
                # enc_fp_s0.append(self.model.encode(fp_s0, fp_s0_len, enc_id=torch.zeros_like(style_id)))   # enc_id = 0 for posts
                #
                # s0_bt_logits = self.model.decode(resp[:, :-1], resp_len-1, enc_fp_s0, dec_id=style_id)   # dec_id = style_id for generating stylized dialogue
                # s0_bt_log_softmax = F.log_softmax(s0_bt_logits, dim=-1)
                # batch_s0_bt_loss = self.criterion(s0_bt_log_softmax.view(-1, s0_bt_log_softmax.shape[-1]), resp.view(-1))
                # batch_s0_bt_loss = batch_s0_bt_loss.reshape([bs, -1])   # [bs, post_seq_len]
                # batch_s0_bt_mask = self.model.get_mask(resp_len)
                # batch_s0_bt_loss = (batch_s0_bt_mask * batch_s0_bt_loss).sum() / batch_s0_bt_mask.sum()
                # s0_bt_loss += batch_s0_bt_loss.item()
                # s0_bt_len += torch.mean(fp_s0_len.float())
                #
                # (bt_weight * batch_s0_bt_loss / self.config.batch_split).backward()

                # s1_back_trans
                text, text_len = t_data['text'].to(self.device), t_data['text_len'].to(self.device)
                text_style_id = t_data['style'].to(self.device)
                bs = text.shape[0]

                self.model_fp.eval()
                enc_text = list()
                enc_text.append(self.model_fp.encode(text, text_len, enc_id=torch.zeros_like(text_style_id)))   # enc_id = 1 for responses

                fp_s1, fp_s1_len = self.model_fp.beam_search_train_bt(enc_text, dec_id=torch.zeros_like(text_style_id))
                self.model_fp.train()
                fp_s1 = fp_s1[:, :, 1:]
                fp_s1_len = fp_s1_len - 1
                beam_size = fp_s1.shape[1]
                fp_s1 = fp_s1.reshape([bs * beam_size, -1])
                fp_s1_len = fp_s1_len.reshape([-1])

                text = torch.unsqueeze(text, 1).repeat([1, beam_size, 1]).reshape([bs * beam_size, -1])
                text_len = torch.unsqueeze(text_len, 1).repeat([1, beam_size]).reshape([-1])
                text_style_id = torch.unsqueeze(text_style_id, 1).repeat([1, beam_size]).reshape([-1])

                enc_fp_s1 = list()
                enc_fp_s1.append(self.model.encode(fp_s1, fp_s1_len, enc_id=torch.zeros_like(text_style_id)))   # enc_id = 0 for posts

                s1_bt_logits = self.model.decode(text[:, :-1], text_len-1, enc_fp_s1, dec_id=text_style_id)    # dec_id = text_style_id for generating stylized dialogue
                s1_bt_log_softmax = F.log_softmax(s1_bt_logits, dim=-1)
                batch_s1_bt_loss = self.criterion(s1_bt_log_softmax.view(-1, s1_bt_log_softmax.shape[-1]), text.view(-1))
                batch_s1_bt_loss = batch_s1_bt_loss.reshape([bs * beam_size, -1])   # [bs, text_seq_len]
                batch_s1_bt_mask = self.model.get_mask(text_len)
                batch_s1_bt_loss = (batch_s1_bt_loss * batch_s1_bt_mask).sum() / batch_s1_bt_mask.sum()
                s1_bt_loss += batch_s1_bt_loss.item()
                s1_bt_len += torch.mean(fp_s1_len.float())

                (bt_weight * batch_s1_bt_loss / self.config.batch_split).backward()
            else:
                batch_s0_bt_loss = 0
                s0_bt_loss += 0
                batch_s1_bt_loss = 0
                s1_bt_loss += 0
                s0_bt_len += 0
                s1_bt_len += 0


            # optimization
            # full_loss = (batch_p2r_loss + batch_r2p_loss + bt_weight * (batch_s0_bt_loss + batch_s1_bt_loss)) / self.config.batch_split
            # full_loss.backward()
            step_count += 1

            # self.logger.info('epoch %d, batch %d' % (epoch, i))
            if (i + 1) % self.config.batch_split == 0:
                if self.config.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.config.clip_grad)
                # update weights
                self.optimizer.step()
                self.optimizer_fp.step()
                self.optimizer.zero_grad()
                self.optimizer_fp.zero_grad()

                # shit log if you are node 0 in every step
                if self.rank == -1 or self.rank == 0:
                    p2r_loss /= step_count
                    r2p_loss /= step_count
                    s0_bt_loss /= step_count
                    s1_bt_loss /= step_count
                    s0_bt_len /= step_count
                    s1_bt_len /= step_count

                    self.train_writer.add_scalar('loss/p2r_loss', p2r_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/r2p_loss', r2p_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s0_bt_loss', s0_bt_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s1_bt_loss', s1_bt_loss, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s0_bt_len', s0_bt_len, self.optimizer.curr_step())
                    self.train_writer.add_scalar('loss/s1_bt_len', s1_bt_len, self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/lr', self.optimizer.rate(), self.optimizer.curr_step())
                    self.train_writer.add_scalar('lr/bt_weight', bt_weight, self.optimizer.curr_step())
                    p2r_loss, r2p_loss, s0_bt_loss, s1_bt_loss = 0, 0, 0, 0
                    s0_bt_len, s1_bt_len = 0, 0
                    step_count = 0

                if self.optimizer.curr_step() % self.config.save_interval == 0:
                    for fun in after_step_funcs:
                        fun(self.optimizer.curr_step())
                # only valid on dev and sample on dev data at every eval_steps
                if self.optimizer.curr_step() % self.config.eval_steps == 0:
                    writer_text = ''
                    for style, post, post_len, resp, resp_len in [[style_id, fp_s0, fp_s0_len, resp, resp_len], [text_style_id, fp_s1, fp_s1_len, text, text_len]]:
                        if post is None:
                            continue
                        for j in range(min(10, post.shape[0])):
                            writer_text += 'post: {}  \nstyle{} resp: {}  \n  \n'.format(self.model_fp.vocab.ids2string(post[j][:post_len[j]-1].tolist()),
                                                                                     style[j].item(),
                                                                                     self.model_fp.vocab.ids2string(resp[j][:resp_len[j]-1].tolist()))
                    if len(writer_text) != 0:
                        self.fp_writer.add_text('online_fp', writer_text, self.optimizer.curr_step())

                    self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate())


    def _eval_test(self, epoch, step, rate):
        recorder = MetricsRecorder(self.device, 's2s_loss', 's2s_loss2', 'ppl', 'ppl2')
        recorder.add_metric_groups('eval', 'eval', 'eval', 'eval')
        all_refs_ids = []
        all_pred_ids = []

        with torch.no_grad():
            self.model.eval()
            self.model_fp.eval()
            for i, d_data in enumerate(self.valid_dialogue_dataloader):
                post, resp = d_data['post'].to(self.device), d_data['resp'].to(self.device)
                post_len, resp_len = d_data['post_len'].to(self.device), d_data['resp_len'].to(self.device)
                style = d_data['style'].to(self.device)
                bs = post.shape[0]

                style0_mask = (style == 0).float()
                style1_mask = (style == 1).float()

                # s2s loss
                enc_contexts = list()
                enc_contexts.append(self.model.encode(post, post_len, enc_id=torch.zeros_like(style)))   # enc_id = 0 for posts

                prevs, nexts = resp[:, :-1].contiguous(), resp
                output_logits = self.model.decode(prevs, resp_len-1, enc_contexts, dec_id=style)   # dec_id = style for generating stylized response
                outputs = F.log_softmax(output_logits, dim=-1)
                batch_s2s_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
                batch_s2s_loss = batch_s2s_loss.reshape([bs, -1])   # [bs, seq_len]
                batch_s2s_loss_mask = self.model.get_mask(resp_len)
                batch_s2s_loss = (batch_s2s_loss * batch_s2s_loss_mask).sum(dim=-1) / batch_s2s_loss_mask.sum(dim=-1)

                batch_ce_loss = self.ce_criterion(output_logits.view(-1, output_logits.shape[-1]), nexts.view(-1))
                batch_ce_loss = batch_ce_loss.reshape(bs, -1)
                batch_ppl = eval_ppl(batch_ce_loss, batch_s2s_loss_mask)

                # blue
                preds, lens = self.model.top_k_top_p_search(enc_contexts, top_p=self.config.annealing_topp, sample_size=1,
                                                            temperature=self.config.temperature, dec_id=style)   # dec_id = style for generating stylized response
                for j in range(bs):
                    refs_ids = [self.ids_cut(resp[j], cut_head=False)]
                    pred_ids = self.ids_cut(preds[j], lens[j], cut_head=True)
                    all_refs_ids.append(refs_ids)
                    all_pred_ids.append(pred_ids)

                recorder.metric_update(i, (torch.sum(batch_s2s_loss * style0_mask), torch.sum(style0_mask)),
                                       (torch.sum(batch_s2s_loss * style1_mask), torch.sum(style1_mask)),
                                       (torch.sum(batch_ppl * style0_mask), torch.sum(style0_mask)),
                                       (torch.sum(batch_ppl * style1_mask), torch.sum(style1_mask)))

            recorder.add_permanent_metric('bleu', eval_bleu(all_refs_ids, all_pred_ids), 1, 'metrics')
            recorder.add_permanent_metric('distinct', eval_distinct(all_pred_ids), 1, 'metrics')
            recorder.add_permanent_metric('f1', eval_f1(all_refs_ids, all_pred_ids), 1, 'metrics')

        if self.rank != -1:
            recorder.all_reduce()

        # but only shit log if you are node 0
        if self.rank == -1 or self.rank == 0:
            recorder.add_to_writer(self.valid_writer, step)
            recorder.write_to_logger(self.logger, epoch, step)

            # and only predicts sample on node 0
            writer_text = ''

            writer_text += 'topk  \n'
            sample_dialog = self._pred_sample_topk_topp(5, temperature=self.config.temperature, topk=self.config.annealing_topk)
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
            sample_dialog = self._pred_sample_topk_topp(5, temperature=self.config.temperature, topp=self.config.annealing_topp)
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
        self.model_fp.train()
        return all_pred_ids

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
            length = ids.index(self.model.vocab.eos_id)
        if cut_head:
            return ids[1: length]
        else:
            return ids[0: length]

    def _pred_sample_gumbel(self, n_sample):
        with torch.no_grad():
            self.model.eval()
            sample_size = 2
            samples_idxs_single = random.sample(range(len(self.valid_dialogue_dataset)), n_sample)
            samples_idxs = []
            for idx in samples_idxs_single:
                samples_idxs += [idx] * self.config.n_styles * sample_size
            samples = PadBatchSeq(self.model.vocab.pad_id)([self.valid_dialogue_dataset[idx] for idx in samples_idxs])
            styles = torch.tensor([i // sample_size for i in range(self.config.n_styles * sample_size)] * n_sample).long().to(self.device)
            enc_contexts = [self.model.encode(c) for c in [samples['post'].to(self.device)]]
            _, lens, prediction = self.model.sample_gumbel(styles, self.optimizer.curr_step(), enc_contexts)
            res = []
            for j in range(n_sample):
                post_str = samples['post'][j * self.config.n_styles].tolist()[1:]
                post_str = self.model.vocab.ids2string(post_str[:post_str.index(self.model.vocab.eos_id)])
                resp_str = samples['resp'][j * self.config.n_styles].tolist()[1:]
                resp_str = self.model.vocab.ids2string(resp_str[:resp_str.index(self.model.vocab.eos_id)])
                pred = {}
                for style in range(self.config.n_styles):
                    utters = []
                    for i in range(sample_size):
                        utter = prediction[j * self.config.n_styles * sample_size +
                                           style * sample_size + i].tolist()
                        utter_len = lens[j * self.config.n_styles * sample_size +
                                         style * sample_size + i].item()
                        utters.append(self.model.vocab.ids2string(utter[1:utter_len - 1]))
                    pred[style] = utters
                res.append({"post": post_str, "resp": resp_str, "pred": pred})

        return res

    def _pred_sample_topk_topp(self, n_sample, temperature, topk=0, topp=0.0):
        with torch.no_grad():
            self.model.eval()
            samples_idxs_single = random.sample(range(len(self.valid_dialogue_dataset)), n_sample)
            samples_idxs = []
            for idx in samples_idxs_single:
                samples_idxs += [idx] * self.config.n_styles
            samples = PadBatchSeq(self.model.vocab.pad_id)([self.valid_dialogue_dataset[idx] for idx in samples_idxs])
            styles = torch.tensor([i for i in range(self.config.n_styles)] * n_sample).long().to(self.device)
            enc_contexts = [self.model.encode(samples['post'].to(self.device), samples['post_len'].to(self.device), enc_id=torch.zeros_like(styles))]

            prediction, lens = self.model.top_k_top_p_search(enc_contexts, sample_size=self.config.top_p_top_k_sample_size,
                                                             top_k=topk, top_p=topp, temperature=temperature, dec_id=styles)
            res = []
            for j in range(n_sample):
                post_str = samples['post'][j * self.config.n_styles].tolist()[0:]
                post_str = self.model.vocab.ids2string(post_str[:post_str.index(self.model.vocab.eos_id)])
                resp_str = samples['resp'][j * self.config.n_styles].tolist()[0:]
                resp_str = self.model.vocab.ids2string(resp_str[:resp_str.index(self.model.vocab.eos_id)])
                pred = {}
                for style in range(self.config.n_styles):
                    utters = []
                    for i in range(self.config.top_p_top_k_sample_size):
                        utter = prediction[j * self.config.n_styles * self.config.top_p_top_k_sample_size +
                                           style * self.config.top_p_top_k_sample_size + i].tolist()
                        utter_len = lens[j * self.config.n_styles * self.config.top_p_top_k_sample_size +
                                         style * self.config.top_p_top_k_sample_size + i].item()
                        utters.append(self.model.vocab.ids2string(utter[1:utter_len - 1]))
                    pred[style] = utters
                res.append({"post": post_str, "resp": resp_str, "pred": pred})

        return res

    def _pred_sample_beam(self, n_sample):
        with torch.no_grad():
            self.model.eval()
            samples_idxs_single = random.sample(range(len(self.valid_dialogue_dataset)), n_sample)
            samples_idxs = []
            for idx in samples_idxs_single:
                for i in range(self.config.n_styles):
                    samples_idxs.append(idx)
            samples = PadBatchSeq(self.model.vocab.pad_id)([self.valid_dialogue_dataset[idx] for idx in samples_idxs])
            styles = torch.tensor([i for i in range(self.config.n_styles)] * n_sample).long().to(self.device)
            prediction = self.model.predict([samples['post'].to(self.device)], styles)
            # enc_contexts = [self.model.encode(c) for c in [samples['post'].to(self.device)]]
            # prediction = self.model.predict_next(enc_contexts)
            res = []
            for j in range(len(samples_idxs)):
                post_str = samples['post'][j].tolist()[1:]
                post_str = self.model.vocab.ids2string(post_str[:post_str.index(self.model.vocab.eos_id)])
                resp_str = samples['resp'][j].tolist()[1:]
                resp_str = self.model.vocab.ids2string(resp_str[:resp_str.index(self.model.vocab.eos_id)])
                pred_str = self.model.vocab.ids2string(prediction[j])
                res.append({"post": post_str, "resp": resp_str, "target_style": styles[j], "pred": pred_str})

        return res


    def test(self, epoch, step, rate):
        return self._eval_test(epoch, step, rate)

    def train(self, start_epoch, epochs, after_step_funcs=[], after_epoch_funcs=[]):
        for epoch in range(start_epoch + 1, epochs):
            self.logger.info('Training on process {}, epoch {}, step {}'.format(
                self.rank, epoch, self.optimizer.curr_step()))
            if self.train_dialogue_sampler and hasattr(self.train_dialogue_sampler, 'set_epoch'):
                self.train_dialogue_sampler.set_epoch(epoch)
            # self._eval_test(epoch, self.optimizer.curr_step(), self.optimizer.rate(), self.cls_optimizer.rate())
            # print(self._pred_sample_topk_topp(5, topp=0.9))
            # print(self._pred_sample_beam(5))
            # print(self._pred_sample_gumbel(5))
            self._eval_train(epoch, after_step_funcs=after_step_funcs)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch, self.device)
