import os
import torch
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input import MultiInputModel
from model.trainer_multi_input import Trainer
from model.text import Vocab
from torch.nn.parallel import DistributedDataParallel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='4')
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


# helpers -----------------------------------------------------
def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model', epoch)
    torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
    if os.path.exists(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch-80))):
        os.remove(os.path.join(train_dir, utils.get_ckpt_filename('model', epoch-80)))


def save_func_step(step):
    filename = utils.get_ckpt_filename('step_model', step)
    torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
    if os.path.exists(os.path.join(train_dir, utils.get_ckpt_filename('step_model', step-80))):
        os.remove(os.path.join(train_dir, utils.get_ckpt_filename('step_model', step-80)))
# helpers -----------------------------------------------------


try:
    if args.local_rank == -1 or args.local_rank == 0:
        logger.info('pytorch version: {}'.format(torch.__version__))
        for i in config:
            logger.info('{}: {}'.format(i, config[i]))
        for i in vars(args):
            logger.info('{}: {}'.format(i, getattr(args, i)))

        dirs = [train_dir, eval_dir, log_dir, best_model]

        for d in dirs:
            if not os.path.isdir(d):
                logger.info('cannot find {}, mkdiring'.format(d))
                os.makedirs(d)

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
    train_dialogue_dataset = dataset.DialogDataset(
        [os.path.join(data_dir, config.train_dialogue_data)], vocab, logger, config.max_seq_len)

    train_text_dataset = dataset.TextDataset(
        [os.path.join(data_dir, config.train_text_data)], vocab, logger, config.max_seq_len)

    valid_dialogue_dataset = dataset.DialogDataset(
        [os.path.join(data_dir, i) for i in config.valid_dialogue_data], vocab, logger, config.max_seq_len)

    logger.info('Building models')
    model = MultiInputModel(config, vocab)
    model_fp = MultiInputModel(config, vocab)
    if args.local_rank == -1 or args.local_rank == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    latest_ckpt = utils.get_latest_ckpt(train_dir)
    if latest_ckpt is None:  # start from scratch
        logger.info('start from CGPT weights')
        cgpt_model = torch.load(config.cgpt_parameters_dir, map_location=device)
        cgpt_model.pop('decoder.pre_softmax.weight')

        b = list(cgpt_model.keys())
        for i in b:
            cgpt_model[i.split('.', 1)[1]] = cgpt_model.pop(i)
        cgpt_model['dec_embeddings.weight'] = (torch.randn((config.n_styles, config.embeddings_size),
                                                           dtype=torch.float32) * 0.01).to(device)
        cgpt_model['enc_embeddings.weight'] = (torch.randn((config.n_styles, config.embeddings_size),
                                                           dtype=torch.float32) * 0.01).to(device)
        model.transformer_module.load_state_dict(cgpt_model, strict=True)
        model.tie_pre_softmax()
        model_fp.transformer_module.load_state_dict(cgpt_model, strict=True)
        model_fp.tie_pre_softmax()
        logger.info('CGPT weights loaded from {}'.format(config.cgpt_parameters_dir))

    trainer = Trainer(
        model, model_fp, train_dialogue_dataset, train_text_dataset, valid_dialogue_dataset, config, log_dir, logger, device,
        distributed=distributed)

    if distributed:
        trainer.model.transformer_module = DistributedDataParallel(
            trainer.model.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)
        trainer.model_fp.transformer_module = DistributedDataParallel(
            trainer.model_fp.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0
    if latest_ckpt is not None:
        logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
        start_epoch = utils.get_epoch_from_ckpt(latest_ckpt)
        trainer.load_state_dict(torch.load(os.path.join(train_dir, latest_ckpt), map_location=device))

    try:
        if args.local_rank in [-1, 0]:
            trainer.train(start_epoch, config.n_epochs, after_step_funcs=[save_func_step], after_epoch_funcs=[save_func])
        else:
            trainer.train(start_epoch, config.n_epochs)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(trainer.state_dict(), os.path.join(train_dir, 'interrupt.pt'))
        raise e
except:
    logger.error(traceback.format_exc())

