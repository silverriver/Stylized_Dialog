import torch


def eval_ppl(ce_loss, mask):
    """
    :param ce_loss: CrossEntropyLoss with shape (batch size, length)
    :param resp: Response with same shape as ce_loss
    :param pad_id: Pad ID
    :return: Sentence-level average ppl value
    """
    mask = mask.float()
    lens = mask.sum(dim=-1)
    loss = (ce_loss * mask).sum(dim=-1)
    loss /= (lens + 1e-5)
    return torch.exp(loss)
