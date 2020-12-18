import torch
from torch.distributions.gumbel import Gumbel
import torch.nn.functional as F
import numpy as np

gumbel = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))


def gumbel_softmax(logits, temperature, device):
    s = gumbel.sample(logits.shape).to(device).squeeze(2)
    y = logits + s
    return F.softmax(y / temperature, dim=-1)


def gumbel_temperature(step, start_temperature, anneal_constant):
    return start_temperature * np.exp(-step / anneal_constant)
