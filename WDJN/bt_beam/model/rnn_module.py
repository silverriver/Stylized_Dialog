# Modified from https://github.com/kamigaito/rnnlm-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from .filtering import top_k_top_p_filtering


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]


class ResRNNBase(nn.Module):
    """
    RNN with residual connections
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, nonlinearity='tanh', dropout=0.2, direction="left2right"):
        super(ResRNNBase, self).__init__()
        self.bidirectional = direction == "both"
        self.direction = direction
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.nhid = nhid
        self.in_linear = nn.Linear(ninp, nhid) if ninp != nhid else None
        self.out_linear = nn.Linear(nhid, ninp) if ninp != nhid else None
        if self.bidirectional:
            if rnn_type in ['LSTM', 'GRU']:
                self.forward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(nhid, nhid, 1, dropout=0.0, bidirectional=False) for _ in range(nlayers)])
                self.backward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(nhid, nhid, 1, dropout=0.0, bidirectional=False) for _ in range(nlayers)])
            elif rnn_type == 'RNN':
                self.forward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(nhid, nhid, 1, nonlinearity=nonlinearity, dropout=0.0, bidirectional=False)
                     for _ in range(nlayers)])
                self.backward_rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(nhid, nhid, 1, nonlinearity=nonlinearity, dropout=0.0, bidirectional=False)
                     for _ in range(nlayers)])
        else:
            if rnn_type in ['LSTM', 'GRU']:
                self.rnns = nn.ModuleList(
                    [getattr(nn, rnn_type)(nhid, nhid, 1, dropout=0.0, bidirectional=self.bidirectional) for _ in
                     range(nlayers)])
            elif rnn_type == 'RNN':
                self.rnns = nn.ModuleList([getattr(nn, rnn_type)(nhid, nhid, 1, nonlinearity=nonlinearity, dropout=0.0,
                                                                 bidirectional=self.bidirectional) for _ in
                                           range(nlayers)])

    def init_hidden(self, bsz):
        """
        Initialize the weights of the model

        Inputs
        ----------
            bsz : batch size

        Return
        ----------
            Tuple
        """
        # make a tensor whose type is similar to current model weights
        weight = next(self.parameters())
        # Separately initialize the memory and hidden states for each layer
        if self.rnn_type == "LSTM":
            if self.direction == "both":
                return (
                    [(weight.new_zeros(1, bsz, self.nhid), weight.new_zeros(1, bsz, self.nhid)) for
                     i in
                     range(self.nlayers)],
                    [(weight.new_zeros(1, bsz, self.nhid), weight.new_zeros(1, bsz, self.nhid)) for
                     i in
                     range(self.nlayers)])
            elif self.direction == "left2right" or self.direction == "right2left":
                return [(weight.new_zeros(1, bsz, self.nhid), weight.new_zeros(1, bsz, self.nhid))
                        for i in range(self.nlayers)]
            else:
                assert (False)
        # Separately initialize the hidden states for each layer
        elif self.rnn_type in ["GRU", "RNN", "RNN_TANH", "RNN_RELU"]:
            if self.direction == "both":
                return ([weight.new_zeros(1, bsz, self.nhid) for i in range(self.nlayers)],
                        [weight.new_zeros(1, bsz, self.nhid) for i in range(self.nlayers)])
            elif self.direction == "left2right" or self.direction == "right2left":
                return [weight.new_zeros(1, bsz, self.nhid) for i in range(self.nlayers)]
            else:
                assert (False)
        # Jointly initialize the hidden states for each layer
        else:
            if self.direction == "both":
                return weight.new_zeros(self.nlayers * 2, bsz, self.nhid)
            elif self.direction == "left2right" or self.direction == "right2left":
                return weight.new_zeros(self.nlayers, bsz, self.nhid)
            else:
                assert (False)

    def encode(self, emb):
        emb = emb.permute(1, 0, 2)
        bsz = emb.shape[1]
        hidden = self.init_hidden(bsz)
        hidden = repackage_hidden(hidden)
        return self.forward(emb, hidden)[0].permute(1, 0, 2)

    def forward(self, emb, hidden):
        if self.bidirectional:
            return self.forward_both(emb, hidden)
        else:
            return self.forward_one(emb, hidden)

    def forward_one(self, emb, init_hidden_list):
        """
        Inputs
        ----------
        emb: [seq_len, nbatch, emb]
        init_hidden_list: tuple

        Returns
        ----------
        rnn_out: [seq_len, nbatch, emb]
        hidden_list: tuple
        """

        """ The number of layers should be same. """
        assert (len(self.rnns) == len(init_hidden_list))

        emb = self.drop(emb)
        rnn_out = emb if self.in_linear is None else self.in_linear(emb)
        hidden_list = []
        for layer_id in range(len(self.rnns)):
            rnn = self.rnns[layer_id]
            init_hidden = init_hidden_list[layer_id]
            res_out = rnn_out
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            rnn_out, hidden = rnn(rnn_out, init_hidden)
            # residual connection
            rnn_out = rnn_out + res_out
            # dropout
            if layer_id < len(self.rnns) - 1:
                rnn_out = self.drop(rnn_out)
            # store hidden states
            hidden_list.append(hidden)
        rnn_out = rnn_out if self.out_linear is None else self.out_linear(rnn_out)
        return rnn_out, hidden_list

    def forward_both(self, emb, init_hidden_list):
        # TODO: need to apply in_linear and out_linear if needed
        """
        Inputs
        ----------
        emb: [seq_len, nbatch, emb]
        init_hidden_list: tuple

        Returns
        ----------
        rnn_out: [seq_len, nbatch, emb]
        hidden_list: tuple
        """

        """ The number of layers should be same. """
        assert (len(self.forward_rnns) == len(init_hidden_list))
        assert (len(self.backward_rnns) == len(init_hidden_list))

        emb = self.drop(emb)
        forward_rnn_out = emb
        """ Reverse the order of token embeddings for the backward rnn. """
        backward_rnn_out = torch.flip(emb, [0])
        forward_hidden_list = []
        backward_hidden_list = []

        """ forward """
        for layer_id in range(len(init_hidden_list)):
            forward_res_out = forward_rnn_out
            forward_init_hidden = init_hidden_list[layer_id][0]
            forward_rnn = self.forward_rnns[layer_id]
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            forward_rnn_out, forward_rnn_hidden = forward_rnn(forward_rnn_out, forward_init_hidden)
            forward_hidden_list.append(forward_rnn_hidden)
            forward_rnn_out = self.drop(forward_rnn_out) + forward_res_out
        """ Shift the forward hidden states. """
        forward_rnn_out = torch.cat(
            [torch.zeros(1, forward_rnn_out.shape[1], forward_rnn_out.shape[2], device=forward_rnn_out.device),
             forward_rnn_out[:-1]], 0)

        """ backward """
        for layer_id in range(len(init_hidden_list)):
            backward_res_out = backward_rnn_out
            backward_init_hidden = init_hidden_list[layer_id][1]
            backward_rnn = self.backward_rnns[layer_id]
            # output: [seq_len, nbatch, nhid], hidden: [1, nbatch, nhid] or ([1, nbatch, nhid], [1, nbatch, nhid])
            backward_rnn_out, backward_rnn_hidden = backward_rnn(backward_rnn_out, backward_init_hidden)
            backward_rnn_out = self.drop(backward_rnn_out) + backward_res_out
        """ Reverse the order of hidden states """
        backward_rnn_out = torch.flip(backward_rnn_out, [0])
        """ Shift the backward hidden states """
        backward_rnn_out = torch.cat([backward_rnn_out[1:],
                                      torch.zeros(1, backward_rnn_out.shape[1], backward_rnn_out.shape[2],
                                                  device=backward_rnn_out.device)], 0)

        """ concatenate output states """
        rnn_out = torch.cat([forward_rnn_out, backward_rnn_out], 2)
        hidden_list = zip(forward_hidden_list, backward_hidden_list)

        return rnn_out, hidden_list


class ResLSTM(ResRNNBase):
    """
    LSTM with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, dropout, direction):
        super(ResLSTM, self).__init__('LSTM', ninp, nhid, nlayers, dropout=dropout, direction=direction)


class ResGRU(ResRNNBase):
    """
    GRU with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, dropout, direction):
        super(ResGRU, self).__init__('GRU', ninp, nhid, nlayers, dropout=dropout, direction=direction)


class ResRNN(ResRNNBase):
    """
    RNN with residual connections
    """

    def __init__(self, ninp, nhid, nlayers, nonlinearity='tanh', dropout=0.0, direction=False):
        super(ResRNN, self).__init__('RNN', ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout,
                                     direction=direction)


if __name__ == '__main__':
    rnn = ResGRU(1000, 256, 3, 0.8, 'left2right')
    hidden = repackage_hidden(rnn.init_hidden(15))
    i = torch.randn((30, 15, 1000))
    o = rnn(i, hidden)
