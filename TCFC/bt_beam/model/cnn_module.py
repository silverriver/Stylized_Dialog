import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_size, out_size, activation=F.relu):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class CNNModule(nn.Module):
    def __init__(self, num_cls,
                 n_embeddings, embedding_size, padding_idx, embed_dropout,
                 feature_size=[128, 128, 128], kernel_size=[2, 3, 4],
                 fc_size=[265, 265], dropout=0.8):
        super(CNNModule, self).__init__()
        self.embeddings = nn.Embedding(n_embeddings, embedding_size, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)

        self.dropout = dropout
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, fs, ks) for fs, ks in zip(feature_size, kernel_size)])
        self.dropout = nn.Dropout(dropout)
        fc_size = list(fc_size)
        fc_size = list(zip([sum(feature_size)] + fc_size[1:], fc_size))
        self.fc = nn.ModuleList([Dense(i, j) for i, j in fc_size])
        self.output_layer = nn.Linear(fc_size[-1][-1], num_cls)

    def forward(self, x, x_len):
        '''x: [bs, len, embed_size], x_len: [bs]'''
        x_embed = self.embeddings(x)
        x_embed = self.embed_dropout(x_embed)
        mask = torch.arange(x_embed.shape[1], device=x_len.device)[None, :] < x_len[:, None]  # [bs, max_len]
        x_embed = x_embed * mask.unsqueeze(2)
        x_embed = x_embed.permute([0, 2, 1])   # [bs, embed_size, len]
        x_embed = [conv(x_embed) for conv in self.convs]  # [(bs, fs, len), ...]
        x_embed = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_embed]  # [(bs, fs), ...]
        x_embed = torch.cat(x_embed, 1)
        x_embed = self.dropout(x_embed)
        for fc in self.fc:
            x_embed = fc(x_embed)
        x_embed = self.dropout(x_embed)
        logits = self.output_layer(x_embed)
        return logits   # [bs, logits]


if __name__ == "__main__":
    embed_size = 100
    seq_len = 20
    n_embeddings = 50257
    model = CNNModule(5, n_embeddings, embed_size, 1, 0.1, [128, 128, 128], [2, 3, 4], [265, 265], 0.8)
    input = torch.randint(high=n_embeddings, size=(10, seq_len))
    input_len = torch.randint(1, 21, (30,))
    print(input.shape)
    output = model(input, input_len)
    print(output.shape)
