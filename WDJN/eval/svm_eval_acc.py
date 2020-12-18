import torch
import torch.nn as nn
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
from text import Vocab
import random
import os
import json
from tqdm import tqdm, trange
from random import sample
import pickle

models = [
["bt_beam", "eval/infer_out_beam_model-10.ckpt-1.txt_lp2.0"],
]


vocab_path = 'chinese_gpt_original/dict.txt'
cgpt_model = 'chinese_gpt_original/Cgpt_model.bin'
cls_model = '../data/svm_cls/train/linear_model.bin'

vocab = Vocab(vocab_path)

cgpt_model = torch.load(cgpt_model, map_location='cpu')
# embedding = cgpt_model['decoder.embeddings.weight']
embeddings = nn.Embedding(13088, 768, padding_idx=1)
embeddings.weight = nn.Parameter(cgpt_model['decoder.embeddings.weight'])

def sent_vec(texts):
    ids = [vocab.string2ids(' '.join(t.replace(' ', ''))) for t in texts]
    res = [embeddings(torch.LongTensor(i)).mean(dim=0).detach().numpy() for i in ids]
    return np.asarray(res)

clf = joblib.load(cls_model)

def test_model(input):
    with open(input) as f:
        test = [json.loads(i) for i in f.readlines()]

    test_text_s0 = []
    test_text_s1 = []
    for i in test:
        text = [i for i in i['pred_style0'] if len(i.strip()) != 0]
        test_text_s0 = test_text_s0 + text
        text = [i for i in i['pred_style1'] if len(i.strip()) != 0]
        test_text_s1 = test_text_s1 + text

    test_label_s0 = np.asarray([0] * len(test_text_s0))
    test_label_s1 = np.asarray([1] * len(test_text_s1))
    test_vec_s0 = sent_vec(test_text_s0)
    test_vec_s1 = sent_vec(test_text_s1)

    pred_s0 = clf.predict(test_vec_s0)
    pred_s1 = clf.predict(test_vec_s1)

    return pred_s0, pred_s1, test_label_s0, test_label_s1


def main(file_path):
    pred_s0, pred_s1, test_label_s0, test_label_s1 = test_model(file_path)
    r = []
    acc0 = metrics.accuracy_score(test_label_s0, pred_s0)
    acc1 = metrics.accuracy_score(test_label_s1, pred_s1)
    print('SVM:', 's0', acc0 * 100, 's1', acc1 * 100, 'mean', (acc0 + acc1) / 2 * 100)
