from model.tokenization_gpt2 import GPT2Tokenizer
import torch
import torch.nn as nn
import numpy as np
import tqdm


tokenizer = GPT2Tokenizer(vocab_file='../data/DialoGPT/small/vocab.json',
                          merges_file='../data/DialoGPT/small/merges.txt')
tokenizer.init_kwargs = {'vocab_file': '../data/DialoGPT/small/vocab.json',
                         'merges_file': '../data/DialoGPT/small/merges.txt'}
tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))

weight_file = "../data/DialoGPT/small/pytorch_model.bin"
weight_dict = torch.load(weight_file, map_location='cpu')
embeddings_weight = weight_dict['transformer.wte.weight'].float()
embeddings_shape = embeddings_weight.shape
embeddings = nn.Embedding(embeddings_shape[0], embeddings_shape[1])
embeddings.weight = nn.Parameter(embeddings_weight, requires_grad=False)


def do_tokenize(list_of_str):
    ids = [tokenizer.encode(s) for s in list_of_str]
    res = [embeddings(torch.LongTensor(i)).mean(dim=0).detach().numpy() for i in ids]
    return np.asarray(res)
