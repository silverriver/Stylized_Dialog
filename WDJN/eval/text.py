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

class Vocab:
    spl = '<p>'
    pad = '<pad>'
    eos = '</s>'
    unk = '<unk>'
    p1 = '<p1>'
    p2 = '<p2>'

    def __init__(self, vocab_file):
        # TODO: add check for special tokens
        self.spec_tokens = [Vocab.spl, Vocab.pad, Vocab.eos, Vocab.unk]
        with open(vocab_file, 'r', encoding='utf8') as fr:
            vocab = [line.strip('\n').split()[0] for line in fr.readlines()]
        vocab = self.spec_tokens + vocab
        # self.spec_tokens = [Vocab.spl, Vocab.pad, Vocab.eos, Vocab.unk, Vocab.p1, Vocab.p2]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for i, t in enumerate(vocab)}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[Vocab.pad]

    @property
    def spl_id(self):
        return self.token2id[Vocab.spl]

    @property
    def p1_id(self):
        return self.token2id[Vocab.p1]

    @property
    def p2_id(self):
        return self.token2id[Vocab.p2]

    @property
    def bos_id(self):
        return self.token2id[Vocab.eos]

    @property
    def eos_id(self):
        return self.token2id[Vocab.eos]

    def string2ids(self, string):
        tokens = string.split()
        ids = [self.token2id[t] for t in tokens if t in self.token2id]
        return ids

    def ids2string(self, ids):
        tokens = [self.id2token[id] for id in ids]
        return ''.join(tokens)

    def ids2string_wo_eos(self, ids):
        res = ''
        for id in ids[1:]:
            if id == self.eos_id:
                return res
            else:
                res += self.id2token[id]


