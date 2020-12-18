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
#  top-k-top-p referenced from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from .modeling_gpt2 import GPT2LMHeadModel
from .configuration_gpt2 import GPT2Config
from .gumbel import gumbel_softmax, gumbel_temperature
from .filtering import top_k_top_p_filtering


class MultiInputModel(nn.Module):
    def __init__(self, config, tokenizer):
        super(MultiInputModel, self).__init__()
        self.config = config
        self.tokenizer = tokenizer

        with open(config.gpt2_config_path, encoding='utf-8') as f:
            gpt2_config_dict = json.load(f)
        gpt2_config = GPT2Config.from_dict(gpt2_config_dict)
        self.transformer_module = GPT2LMHeadModel(gpt2_config, config.n_styles)

    def forward(self, x, x_len, contexts=[], x_embed=None):
        enc_contexts = [self.encode(c, c_len) for c, c_len in contexts]
        return self.decode(x, x_len, enc_contexts, x_embed=x_embed)

    def encode(self, x, x_len, x_embed=None, style_ids=None):
        mask = self.get_mask(x_len)
        result = self.transformer_module(x,
                                         attention_mask=mask,
                                         inputs_embeds=x_embed,
                                         style_ids=style_ids,
                                         get_representation=True)[0]
        if style_ids is not None:
            result = result[:, 1:]
        return result, mask

    def decode(self, x, x_len, enc_contexts=None, x_embed=None, style_ids=None):
        mask = self.get_mask(x_len)
        x = self.transformer_module(x,
                                    attention_mask=mask,
                                    enc_contexts=enc_contexts,
                                    inputs_embeds=x_embed,
                                    style_ids=style_ids)[0]
        return x

    @staticmethod
    def get_mask(x_len):
        mask = torch.arange(max(x_len), device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
        return mask

    def classify(self, seq_lengths, x=None, x_embed=None):
        # return self.classifier(seq_lengths, x, x_embed)
        if x_embed is None:
            if hasattr(self.transformer_module, 'wte'):
                embeddings = self.transformer_module.wte
            else:
                embeddings = self.transformer_module.module.wte
            x_embed = embeddings(x)
        return self.classifier(x_embed, seq_lengths)

    def sample_gumbel(self, styles, step, enc_contexts=[], max_len=None):
        device = next(self.parameters()).device
        x = self.vocab.style_tensor_conversion(styles).unsqueeze(1)
        x_prob = torch.zeros(x.shape + (len(self.vocab),)).float().to(device)
        x_prob[:, 0, self.vocab.eos_id] = 1.0
        batch_size = styles.shape[0]
        is_end = torch.zeros(batch_size, dtype=torch.uint8, device=device).bool()
        if hasattr(self.transformer_module, 'wte'):
            embeddings = self.transformer_module.wte
        else:
            embeddings = self.transformer_module.module.wte
        x_embed = embeddings(x)

        styles_0 = torch.zeros_like(styles)
        x_0 = self.vocab.style_tensor_conversion(styles_0).unsqueeze(1)
        x_embed = (x_embed + embeddings(x_0)) / 2
        z = list()
        z.append((enc_contexts[0][0] / 2, enc_contexts[0][1]))

        lens = torch.ones(batch_size, dtype=torch.long, device=device)
        temperature = gumbel_temperature(step, self.config.start_temperature, self.config.anneal_constant)

        if max_len is None:
            max_len = self.config.max_seq_len
        for i in range(1, max_len):
            decode_logits = self.decode(x, z, x_embed)[:, -1, :]
            decode_gumbel = gumbel_softmax(decode_logits, temperature, device)
            decode_x = torch.argmax(decode_gumbel, dim=1)
            # decode_x[is_end] = self.vocab.pad_id
            lens[~is_end] += 1
            is_end[decode_x == self.vocab.eos_id] = 1

            decode_embed = torch.matmul(decode_gumbel, embeddings.weight).unsqueeze(1)
            decode_x = decode_x.unsqueeze(1)
            x_embed = torch.cat((x_embed, decode_embed), dim=1)
            x = torch.cat((x, decode_x), dim=1)
            x_prob = torch.cat((x_prob, decode_gumbel.unsqueeze(1)), dim=1)

            # if all(is_end):
            #    break

        return x_prob, lens, x

    def predict(self, contexts=[], styles=None):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts, styles=styles)
        return prediction

    def predict_beam(self, contexts=[], styles=None):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts, return_beams=True, styles=styles)

        return prediction

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.config.length_penalty / (5 + 1) ** self.config.length_penalty

    def predict_next(self, enc_contexts=[], return_beams=False, prefix=[]):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            ind = len(prefix)
            if ind:
                assert batch_size == 1
                prefix_sentence = [self.vocab.bos_id] + prefix
                prevs = torch.LongTensor(prefix_sentence).to(device)
                prevs = prevs.expand(self.config.beam_size, ind + 1)
            else:
                prevs = torch.full((batch_size * self.config.beam_size, 1), fill_value=self.vocab.bos_id, dtype=torch.long,
                                   device=device)
            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.config.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.config.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))
            outputs, _ = self.transformer_module(prevs, beam_enc_contexts)
            logits = self.generate(outputs[:, -1, :])
            probs = F.softmax(logits, dim=-1)
        return probs[0].tolist()

    def beam_search(self, enc_contexts=[], return_beams=False, style_ids=None, return_lens=False, beam_size=None):
        if beam_size is None:
            beam_size = self.config.beam_size
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            assert style_ids is not None

            if hasattr(self.transformer_module, 'transformer'):
                wte = self.transformer_module.transformer.wte
            else:
                wte = self.transformer_module.module.transformer.wte

            if hasattr(self.transformer_module, 'style_embed'):
                se = self.transformer_module.style_embed
            else:
                se = self.transformer_module.module.style_embed

            batch_size = style_ids.shape[0]
            device = next(self.parameters()).device
            beam_style_ids = prevs = style_ids.unsqueeze(1).repeat(1, beam_size).view(-1, 1)
            beam_style_ids = beam_style_ids.reshape(-1)

            beam_scores = torch.zeros(batch_size, beam_size, device=device)
            beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            current_sample_prob = 1
            group_size = beam_size // self.config.diversity_groups
            diversity_penalty = torch.zeros((batch_size, len(self.tokenizer)), device=device)

            # zrs:
            repeat = [{} for i in range(batch_size * beam_size)]
            embeds = se(prevs)
            # **********
            for i in range(self.config.max_seq_len):
                mask = self.get_mask(beam_lens.reshape(-1))
                logits = self.transformer_module(None,
                                                 attention_mask=mask,
                                                 enc_contexts=beam_enc_contexts,
                                                 inputs_embeds=embeds,
                                                 style_ids=beam_style_ids)[0]
                logits = logits[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                # zrs: remove n repeat. prevs: (batch_size*beam_size, 1)
                for idx in range(batch_size * beam_size):
                    for key in repeat[idx]:
                        for value in repeat[idx][key]:
                            log_probs[idx][value] = -1000
                # **********
                log_probs = log_probs.view(batch_size, beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                # zrs, log_probs: batch * beam * dim
                ba, be, dim = beam_scores.shape
                for ba_idx in range(ba):
                    for be_idx in range(be):
                        if int(torch.max(beam_scores[ba_idx][be_idx]) == torch.min(beam_scores[ba_idx][be_idx])):
                            temp = float(beam_scores[ba_idx][be_idx][0])
                            beam_scores[ba_idx][be_idx] = -float('inf')
                            beam_scores[ba_idx][be_idx][0] = temp
                # **********
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, len(self.tokenizer))
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
                else:

                    penalty = penalty.view(batch_size, self.config.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.config.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.config.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.config.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            # print('*********')
                            beam_probas = F.softmax(g_beam_scores/self.config.temperature, dim=-1)
                            if self.config.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.config.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            # print('|||||||||')
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * len(self.tokenizer)

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, len(self.tokenizer)),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / len(self.tokenizer)).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs).bool()
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = 0
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.tokenizer.eos_token_id] = 1

                sym_idxs = sym_idxs.view(batch_size * beam_size, 1)
                prevs = prevs.view(batch_size, beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)
                embeds = torch.cat([embeds, wte(sym_idxs.view(-1, 1))], dim=1)

                # zrs:
                prevs_list = prevs.tolist()
                for b in range(batch_size * beam_size):
                    b_list = prevs_list[b]
                    if len(b_list) > 2 and b_list[-1] != 0 and b_list[-1] != self.tokenizer.eos_token_id:
                        key = (int(b_list[-3]), int(b_list[-2]))
                        if key in repeat[b]:
                            repeat[b][key].append(int(b_list[-1]))
                        else:
                            repeat[b][key] = [int(b_list[-1])]
                # ********

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.config.annealing

            predicts = []
            predict_lens = []
            result = prevs.view(batch_size, beam_size, -1)

            if return_beams:
                bests = torch.argsort(beam_scores, dim=-1, descending=True)
                for i in range(batch_size):
                    temp = []
                    for j in range(beam_size):
                        best_len = beam_lens[i, bests[i][j]]
                        best_seq = result[i, bests[i][j], 1:best_len - 1]
                        temp.append(best_seq.tolist())
                    predicts.append(temp)
                return predicts

            if self.config.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())
                if return_lens:
                    predicts[-1].append(self.tokenizer.eos_token_id)
                    predict_lens.append(best_len.item() - 1)

        if return_lens:
            return predicts, predict_lens
        else:
            return predicts

    def top_k_top_p_search(self, enc_contexts=[], top_k=0, top_p=0.0, filter_value=-float('Inf'),
                           temperature=0.8, style_ids=None, sample_size=None):
        if sample_size is None:
            sample_size = self.config.top_p_top_k_sample_size
        with torch.no_grad():
            assert style_ids is not None

            if hasattr(self.transformer_module, 'transformer'):
                wte = self.transformer_module.transformer.wte
            else:
                wte = self.transformer_module.module.transformer.wte

            if hasattr(self.transformer_module, 'style_embed'):
                se = self.transformer_module.style_embed
            else:
                se = self.transformer_module.module.style_embed

            batch_size = style_ids.shape[0]
            device = next(self.parameters()).device
            beam_style_ids = prevs = style_ids.unsqueeze(1).repeat(1, sample_size).view(-1, 1)
            beam_style_ids = beam_style_ids.reshape([-1])

            lens = torch.ones(batch_size * sample_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size * sample_size, device=device).bool()

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, sample_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, sample_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            embeds = se(prevs)
            for i in range(self.config.max_seq_len):
                mask = self.get_mask(lens)
                logits = self.transformer_module(None,
                                                  attention_mask=mask,
                                                  enc_contexts=beam_enc_contexts,
                                                  inputs_embeds=embeds,
                                                  style_ids=beam_style_ids)[0]
                logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).view(-1)

                next_token[is_end] = self.tokenizer.eos_token_id
                lens[~is_end] += 1
                is_end[next_token == self.tokenizer.eos_token_id] = 1
                prevs = torch.cat((prevs, next_token.view(-1, 1)), dim=-1)
                embeds = torch.cat([embeds, wte(next_token.view(-1, 1))], dim=1)

                if all(is_end.view(-1)):
                    break

        return prevs, lens
