import os
import torch
import traceback
import model.utils as utils
import model.dataset as dataset
from model.model_multi_input_gpt2 import MultiInputModel
from model.trainer_multi_input_gpt2 import Trainer
from model.tokenization_gpt2 import GPT2Tokenizer
from torch.nn.parallel import DistributedDataParallel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file', default='config.json')
parser.add_argument('--gpu', help='which gpu to use', type=str, default='0')
parser.add_argument("--local_rank", help='used for distributed training', type=int, default=-1)

args = parser.parse_args()
config = utils.load_config(args.config)
config_path = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(config_path, 'main.log'))
if not hasattr(config, 'save_interval'):
    config.save_interval = config.eval_steps

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


def save_step_func(step, device):
    filename = utils.get_ckpt_step_filename('model', step)
    torch.save(trainer.state_dict(), os.path.join(train_dir, filename))
    if os.path.exists(os.path.join(train_dir, utils.get_ckpt_step_filename('model', step - 20 * config.save_interval))):
        os.remove(os.path.join(train_dir, utils.get_ckpt_step_filename('model', step - 20 * config.save_interval)))


def get_cache_file_name(file_name):
    return os.path.join(os.path.dirname(file_name), os.path.basename(file_name) + '.cache')


def test_model(model, device, tokenizer):
    new_user_input_ids = tokenizer.encode('Can money buy happiness?'+tokenizer.eos_token, return_tensors='pt').to(device)

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(new_user_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)))


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

    tokenizer = GPT2Tokenizer(vocab_file=config.gpt2_tokenizer_vocab, merges_file=config.gpt2_tokenizer_merges)
    tokenizer.init_kwargs = {'vocab_file': config.gpt2_tokenizer_vocab, 'merges_file': config.gpt2_tokenizer_merges}
    tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))

    train_dialogue_dataset = dataset.TCFCDialogDataset(
        [os.path.join(data_dir, config.train_dialogue_data)], tokenizer, logger,
        get_cache_file_name(os.path.join(data_dir, config.train_dialogue_data)), max_lengths=config.max_seq_len)
    valid_dialogue_dataset = dataset.TCFCDialogDataset(
        [os.path.join(data_dir, config.valid_data)], tokenizer, logger,
        get_cache_file_name(os.path.join(data_dir, config.valid_data)), max_lengths=config.max_seq_len)
    train_text_dataset = dataset.TCFCTextDataset(
        [os.path.join(data_dir, config.train_text_data)], tokenizer, logger,
        get_cache_file_name(os.path.join(data_dir, config.train_text_data)), max_lengths=config.max_seq_len)

    logger.info('Building models')
    model = MultiInputModel(config, tokenizer).to(device)
    total_elements = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            total_elements += param.nelement()
    model_r2p = MultiInputModel(config, tokenizer).to(device)
    for name, param in model_r2p.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            total_elements += param.nelement()

    print("Total elements", total_elements)

    latest_ckpt = utils.get_latest_ckpt(train_dir)
    if latest_ckpt is None:  # start from scratch
        logger.info('start from GPT2 weights')
        utils.load_gpt2(model, config.gpt2_weights_path, device)
        utils.load_gpt2(model_r2p, config.gpt2_weights_path, device)
        logger.info('GPT weights loaded from {}'.format(config.gpt2_weights_path))

    trainer = Trainer(model, model_r2p,
                      train_dialogue_dataset, valid_dialogue_dataset, train_text_dataset,
                      config, log_dir, logger, device,
                      distributed=distributed, config_path=config_path)

    if distributed:
        trainer.model.transformer_module = DistributedDataParallel(
            trainer.model.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)
        trainer.model_r2p.transformer_module = DistributedDataParallel(
            trainer.model_r2p.transformer_module, device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0
    if latest_ckpt is not None:
        logger.info('Weights loading from {}'.format(os.path.join(train_dir, latest_ckpt)))
        start_epoch = utils.get_epoch_from_ckpt(latest_ckpt)
        trainer.load_state_dict(torch.load(os.path.join(train_dir, latest_ckpt), map_location=device))

    try:
        if args.local_rank in [-1, 0]:
            # trainer.train(start_epoch, config.n_epochs,
            #               after_epoch_funcs=[], after_step_funcs=[save_step_func])
            trainer.train(start_epoch, config.n_epochs,
                          after_epoch_funcs=[save_func], after_step_funcs=[save_step_func])
        else:
            trainer.train(start_epoch, config.n_epochs)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(trainer.state_dict(), os.path.join(train_dir, 'interrupt.pt'))
        raise e
except:
    logger.error(traceback.format_exc())
