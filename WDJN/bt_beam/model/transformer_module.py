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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import checkpoint_sequential


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1)

        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0).bool()
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        if padding_mask is not None:
            w.masked_fill_(padding_mask.all(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3), 0)

        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = True  # self-attention
        elif kv_same:
            q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
            query = F.linear(query, q_w, q_b)
            kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
            key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, apply_future_mask, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, *contexts, id_embed=None):
        '''contexts = [(context1, padding_mask1), ...]'''

        inputs = (x, padding_mask) + contexts

        full_attn = 0
        n_attn = len(inputs) // 2
        for i in range(0, len(inputs), 2):
            c, m = inputs[i], inputs[i + 1].bool()
            a = self.attn(x, c, c, m)
            full_attn += (a / n_attn)

        if id_embed is not None:
            full_attn = full_attn * n_attn / (n_attn + 1) + id_embed / (n_attn + 1)

        full_attn = self.dropout(full_attn)
        # full_attn.masked_fill_(padding_mask.unsqueeze(2), 0)
        x = self.attn_norm(x + full_attn)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)

        return (x, padding_mask) + contexts


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, padding_idx, n_heads, dropout,
                 embed_dropout, attn_dropout, ff_dropout, n_styles):
        super(TransformerModule, self).__init__()

        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0)
        self.enc_embeddings = nn.Embedding(n_styles, embeddings_size)
        self.dec_embeddings = nn.Embedding(n_styles, embeddings_size)

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout) for _ in range(n_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    @staticmethod
    def get_mask(x_len):
        mask = torch.arange(torch.max(x_len), device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
        return mask.bool()

    def forward(self, x, x_len, enc_contexts=[], x_embed=None, enc_id=None, dec_id=None):
        '''contexts = [(context1, padding_mask1), ...]'''
        assert (dec_id is None) ^ (enc_id is None)
        assert (x is None) ^ (x_embed is None)
        assert (x_len is not None)

        if enc_id is not None:
            id_embed = self.enc_embeddings(enc_id).reshape(enc_id.shape[0], 1, self.embeddings.embedding_dim)
        else:
            id_embed = self.dec_embeddings(dec_id).reshape(dec_id.shape[0], 1, self.embeddings.embedding_dim)

        if x_embed is None:
            x_embed = self.embeddings(x)
            x_embed = torch.cat([id_embed, x_embed], dim=1)
            x_len = x_len + 1

        x_embed = x_embed * math.sqrt(self.embeddings.embedding_dim)
        x_mask = self.get_mask(x_len)
        padding_mask = ~x_mask
        positions = torch.cumsum(x_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)

        x_embed = x_embed + self.pos_embeddings(positions)
        x_embed = x_embed + id_embed

        x_embed = self.embed_dropout(x_embed)
        x_embed.masked_fill_(padding_mask.unsqueeze(2), 0)
        enc_contexts = sum(enc_contexts, ())

        for layer in self.layers:
            out = layer(x_embed, padding_mask, *enc_contexts, id_embed=id_embed)
            x_embed = out[0]

        return x_embed, padding_mask

