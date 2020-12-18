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

from torch.utils.data import Dataset
import torch
import json
import pickle
import os
import copy


class TCFCDialogDataset(Dataset):
    def __init__(self, paths, tokenizer, logger, cache_file, max_lengths=2048):
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_lengths = max_lengths
        if not os.path.isfile(cache_file):
            self.data = TCFCDialogDataset.make_dataset(paths, tokenizer, logger, max_lengths)
        else:
            logger.info('reading cached data from {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info('{} cached data record loaded'.format(len(self.data)))

    @staticmethod
    def make_dataset(paths, tokenizer, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                res = [i.strip().split('\t') for i in f.readlines()]
            for i in res:
                d = [int(i[0]), tokenizer.encode(i[1]), tokenizer.encode(i[2])]
                if len(d[1]) < max_lengths and len(d[2]) < max_lengths:
                    dataset.append(d)
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        style, post, resp = self.data[idx]
        # encode style here
        post = post + [self.tokenizer.eos_token_id]
        resp = resp + [self.tokenizer.eos_token_id]
        return {"type": "tcfc_dialogue", "post": post, "post_len": len(post),
                "resp": resp, "resp_len": len(resp), "style": style}


class TCFCTextDataset(Dataset):
    def __init__(self, paths, tokenizer, logger, cache_file, max_lengths=2048):
        self.logger = logger
        self.tokenizer = tokenizer
        self.max_lengths = max_lengths
        if not os.path.isfile(cache_file):
            self.data = TCFCTextDataset.make_dataset(paths, tokenizer, logger, max_lengths)
        else:
            logger.info('reading cached data from {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info('{} cached data record loaded'.format(len(self.data)))

    @staticmethod
    def make_dataset(paths, tokenizer, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                res = [i.strip().split('\t') for i in f.readlines()]
            for i in res:
                d = tokenizer.encode(i[2])
                if len(d) < max_lengths:
                    dataset.append(d)
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # encode style here
        text = text + [self.tokenizer.eos_token_id]
        return {"type": "tcfc_text", "text": text, "text_len": len(text)}


class TCFCValidDataset(Dataset):
    def __init__(self, paths, vocab, logger, max_lengths=2048, style=None):
        self.logger = logger
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.style = style
        self.data = TCFCValidDataset.make_dataset(paths, vocab, logger, max_lengths, style)

    @staticmethod
    def make_dataset(paths, vocab, logger, max_lengths, style):
        def adjust_id(vocab, id):
            if id == 50256:  # eos
                return vocab.eos_id
            else:
                return id + vocab.n_special_tokens

        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf8') as f:
                j = json.load(f)
            for d in j:
                dataset.append([[adjust_id(vocab, x) for x in d['post'][:max_lengths]],
                                [adjust_id(vocab, x) for x in d['resp'][:max_lengths]],
                                [adjust_id(vocab, x) for x in d['formal1'][:max_lengths]],
                                [adjust_id(vocab, x) for x in d['formal2'][:max_lengths]]])
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        post, resp, formal1, formal2 = self.data[idx]
        post = [self.vocab.eos_id] + post + [self.vocab.eos_id]
        resp = [self.vocab.style_id(0)] + resp + [self.vocab.eos_id]
        formal1 = [self.vocab.style_id(1)] + formal1 + [self.vocab.eos_id]
        formal2 = [self.vocab.style_id(1)] + formal2 + [self.vocab.eos_id]
        return {"type": "test", "post": post, "resp": resp, "formal1": formal1, "formal2": formal2,
                "post_len": len(post), "resp_len": len(resp),
                "formal1_len": len(formal1), "formal2_len": len(formal2)}


class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()

        def _pad_single(name):
            res['%s_len' % name] = torch.LongTensor([i['%s_len' % name] for i in batch])
            max_len = max([len(i[name]) for i in batch])
            res[name] = torch.LongTensor(
                [i[name] + [self.pad_id] * (max_len - len(i[name])) for i in batch])

        def _pad(*names):
            for name in names:
                _pad_single(name)

        if len(batch) > 0:
            t = batch[0]["type"]
            if t == "dialogue":
                res['style'] = torch.LongTensor([i['style'] for i in batch])
                _pad('post', 'resp')
            elif t == "text":
                res['style'] = torch.LongTensor([i['style'] for i in batch])
                _pad('text')
            elif t == 'tcfc_dialogue':
                _pad('post', 'resp')
                res['style'] = torch.LongTensor([i['style'] for i in batch])
            elif t == 'tcfc_text':
                _pad('text')
            elif t == 'test':
                _pad('post', 'resp', 'formal1', 'formal2')
            elif t == 'tcfc_dialogue_ncand':
                _pad('post', 'resp')
                res['resp_style_len'] = torch.LongTensor([i['resp_style_len'] for i in batch])
                resp_style_max_len = max([max(i['resp_style_len']) for i in batch])
                res['resp_style'] = torch.LongTensor(
                    [[i + [self.pad_id] * (resp_style_max_len - len(i)) for i in j['resp_style']] for j in batch])
                res['resp_style_prob'] = torch.FloatTensor([i['resp_style_prob'] for i in batch])
        return res
