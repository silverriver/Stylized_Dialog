import os
import torch
import random
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input import MultiInputModel
from torch.utils.data import DataLoader
from model.text import Vocab
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='infer_config.json')
parser.add_argument('--out_file', help='out_file', default='infer_out_best')
parser.add_argument('--ckpt', help='out_file', default='model-20.ckpt')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='3')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

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

    # code for distributed training
    distributed = (args.local_rank != -1)
    if distributed:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.manual_seed(config.seed)
    else:
        device = torch.device("cuda", 0)

    vocab = Vocab(config.vocab_path)
    test_dataset = dataset.DialogDataset([os.path.join(data_dir, i) for i in  config.infer_dialogue_data], vocab,
                                         logger, config.max_seq_len)

    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None

    test_dataloader = DataLoader(test_dataset, sampler=sampler, pin_memory=True,
                                 batch_size=config.dialog_batch_size, collate_fn=dataset.PadBatchSeq(vocab.pad_id))

    logger.info('Building models')
    model =  MultiInputModel(config, vocab).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    latest_ckpt = args.ckpt
    logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
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
        out_file_name = os.path.join(eval_dir, args.out_file + '_' + latest_ckpt + str(args.local_rank) + '.txt')
        logger.info('outputing to {}'.format(out_file_name))

        with open(out_file_name, 'w') as f:
            if args.local_rank == -1 or args.local_rank == 0:
                ITER = tqdm(test_dataloader, dynamic_ncols=True, total=len(test_dataloader))
            else:
                ITER = test_dataloader

            for data in ITER:
                bs = data['post'].shape[0]
                tmp=[]
                for i in range(bs):
                    tmp.append({})
                    post_str = data['post'][i].tolist()[1:]
                    tmp[-1]['post'] = vocab.ids2string(post_str[:post_str.index(vocab.eos_id)])
                    resp_str = data['resp'][i].tolist()[1:]
                    tmp[-1]['resp'] = vocab.ids2string(resp_str[:resp_str.index(vocab.eos_id)])
                    tmp[-1]['style'] = data['style'][i].item()

                prediction = model.predict([[data['post'].to(device), data['post_len'].to(device)]],
                                           torch.zeros_like(data['style']).to(device),
                                           torch.ones_like(data['style']).to(device))
                for i in range(bs):
                    tmp[i]['pred_style1'] = [vocab.ids2string(prediction[i])]

                prediction = model.predict([[data['post'].to(device), data['post_len'].to(device)]],
                                           torch.zeros_like(data['style']).to(device),
                                           torch.zeros_like(data['style']).to(device))
                for i in range(bs):
                    tmp[i]['pred_style0'] = [vocab.ids2string((prediction[i]))]

                for i in tmp:
                    print(json.dumps(i, ensure_ascii=False), file=f)

except:
    logger.error(traceback.format_exc())

