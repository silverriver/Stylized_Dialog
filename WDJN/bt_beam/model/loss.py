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
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels, smoothing=0.0, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing <= 1

        self.ignore_index = ignore_index
        self.confidence = 1 - smoothing

        if smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
            n_ignore_idxs = 1 + (ignore_index >= 0)   # 1 for golden truth, later one for ignore_index
            one_hot = torch.full((1, n_labels), fill_value=(smoothing / (n_labels - n_ignore_idxs)))
            if ignore_index >= 0:
                one_hot[0, ignore_index] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            self.criterion = nn.NLLLoss(reduction='none', ignore_index=ignore_index)
        
    def forward(self, log_inputs, targets):
        if self.confidence < 1:
            tdata = targets.data
  
            tmp = self.one_hot.repeat(targets.shape[0], 1)
            tmp.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if self.ignore_index >= 0:
                mask = torch.nonzero(tdata.eq(self.ignore_index)).squeeze(-1)
                if mask.numel() > 0:
                    tmp.index_fill_(0, mask, 0)

            targets = tmp
            res = self.criterion(log_inputs, targets).sum(dim=-1)
        else:
            res = self.criterion(log_inputs, targets)
        return res


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='none')

    def forward(self, input, target_prob, target_lens):
        """
        :param input: predictions (batch_size, sentence_max_len, vocab_len), result of log_softmax
        :param target_prob: target labels (batch_size, sentence_max_len, vocab_len), probabilities
        :param target_lens: target lengths (batch_size)
        :return: loss
        """

        assert input.shape == target_prob.shape
        assert input.shape[0] == target_lens.shape[0]
        bs = input.shape[0]
        mask = torch.arange(target_prob.shape[1], device=target_prob.device).view(1, -1) < target_lens.view(-1, 1)
        input = input.reshape([-1, input.shape[2]])
        target_prob = target_prob.reshape([-1, target_prob.shape[2]])
        loss = self.criterion(input, target_prob)
        loss = loss.sum(dim=-1).reshape([bs, -1])   # [bs, max_seq]
        return loss, mask


if __name__ == '__main__':
    inputs = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]).float()
    inputs.requires_grad = True
    target = torch.tensor([[[1, 0], [0, 1]], [[0.5, 0.5], [0.5, 0.5]]]).float()
    target.requires_grad = True
    target_lens = torch.tensor([2, 1])
    criterion = SoftCrossEntropy()
    loss = criterion(inputs, target, target_lens)
    loss.backward()
    print(loss)
