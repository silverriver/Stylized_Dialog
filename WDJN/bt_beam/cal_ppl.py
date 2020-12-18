import os
import torch
import random
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input import MultiInputModel
from model.trainer_multi_input import Trainer
from torch.utils.data import DataLoader
from model.text import Vocab
from tqdm import tqdm
from torch import nn
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='infer_config.json')
parser.add_argument('--ckpt', help='out_file', default='model-10.ckpt')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='7')

args = parser.parse_args()
config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))

train_dir = os.path.join(config_path, config['train_dir'])
data_dir = os.path.join(config_path, config['data_dir'])
eval_dir = os.path.join(config_path, config['eval_dir'])
log_dir = os.path.join(config_path, config['log_dir'])
best_model = os.path.join(config_path, config['best_dir'])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


try:
    logger.info('pytorch version: {}'.format(torch.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    device = torch.device("cuda", 0)

    vocab = Vocab(config.vocab_path)
    test_dataset = dataset.DialogDataset(
        [os.path.join(data_dir, i) for i in config.infer_dialogue_data], vocab, logger, config.max_seq_len)

    test_dataloader = DataLoader(test_dataset, pin_memory=True,
                                 batch_size=config.dialog_batch_size, collate_fn=dataset.PadBatchSeq(vocab.pad_id))

    logger.info('Building models')
    model =  MultiInputModel(config, vocab).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    step = 9999
    min_ppl0, min_ppl1 = 10000, 10000
    latest_ckpt = args.ckpt
    logger.info('step:{}, Weights loading from {}'.format(step, os.path.join(train_dir, latest_ckpt)))
    weights = torch.load(os.path.join(train_dir, latest_ckpt))['model']
    weight_keys = list(weights.keys())
    for key in weight_keys:
        if key.startswith('transformer_module.module'):
            weights['transformer_module' + key[len('transformer_module.module'):]] = weights[key]
            weights.pop(key)

    model.load_state_dict(weights, strict=True)

    with torch.no_grad():
        model.eval()
        res = []
        s0_ppl, s1_ppl = 0, 0
        s0_total, s1_total = 0, 0
        ce_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction='none').to(device)
        for data in tqdm(test_dataloader, dynamic_ncols=True, total=len(test_dataloader)):
            post, resp = data['post'].to(device), data['resp'].to(device)
            post_len, resp_len = data['post_len'].to(device), data['resp_len'].to(device)
            style = data['style'].to(device)
            bs = data['post'].shape[0]

            enc_contexts = list()
            enc_contexts.append(model.encode(post, post_len, enc_id=torch.zeros_like(style)))
            prevs, nexts = resp[:, :-1].contiguous(), resp
            output_logits = model.decode(prevs, resp_len-1, enc_contexts, dec_id=style)

            ce_loss = ce_criterion(output_logits.view(-1, output_logits.shape[-1]), nexts.view(-1))
            ce_loss = ce_loss.reshape(bs, -1)

            padding_mask = model.get_mask(resp_len).float()
            lens = padding_mask.sum(dim=-1)
            loss = (ce_loss * padding_mask).sum(dim=-1)
            loss = loss / lens
            ppl = torch.exp(loss)

            s0_mask = (style==0).float()
            s1_mask = (style==1).float()

            s0_total += s0_mask.sum().item()
            s1_total += s1_mask.sum().item()

            s0_ppl += (ppl * s0_mask).sum().item()
            s1_ppl += (ppl * s1_mask).sum().item()

        mark0, mark1 = '', ''
        if s0_ppl / s0_total < min_ppl0:
            min_ppl0 = s0_ppl / s0_total
            mark0 = '*'
        if s1_ppl / s1_total < min_ppl1:
            min_ppl1 = s1_ppl / s1_total
            mark1 = '*'

        logger.info('s0_total: {}, s0_ppl: {} {}'.format(s0_total, s0_ppl / s0_total, mark0))
        logger.info('s1_total: {}, s1_ppl: {} {}'.format(s1_total, s1_ppl / s1_total, mark1))

except:
    logger.error(traceback.format_exc())

