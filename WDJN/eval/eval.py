import os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import json
from tqdm import tqdm, trange
from random import sample
import numpy as np
import pickle
import argparse
import bert_eval_acc
import svm_eval_acc


smooth = SmoothingFunction()


def eval_bleu(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    return corpus_bleu(ref, pred, smoothing_function=smooth.method1)

def eval_bleu_detail(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    return corpus_bleu(ref, pred, weights=[1, 0, 0, 0], smoothing_function=smooth.method1),\
           corpus_bleu(ref, pred, weights=[0, 1, 0, 0], smoothing_function=smooth.method1), \
           corpus_bleu(ref, pred, weights=[0, 0, 1, 0], smoothing_function=smooth.method1), \
           corpus_bleu(ref, pred, weights=[0, 0, 0, 1], smoothing_function=smooth.method1)

def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


def eval_distinct_detail(hyps_resp):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [[str(x) for x in l] for l in hyps_resp]
    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram(hyps_resp, 1) / float(num_tokens)
    dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)

    return dist1, dist2


def eval_f1(ref, pred):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: f1 score
    """
    assert len(ref) == len(pred) > 0
    precisions = []
    recalls = []
    for i, s in enumerate(pred):
        ref_set = set()
        for rs in ref[i]:
            for w in rs:
                ref_set.add(w)
        pred_set = set()
        for w in s:
            pred_set.add(w)

        p = 0
        for w in s:
            if w in ref_set:
                p += 1
        if len(s) > 0:
            p /= len(s)
        r = 0
        for rs in ref[i]:
            for w in rs:
                if w in pred_set:
                    r += 1
        tot_l = sum([len(rs) for rs in ref[i]])
        if tot_l > 0:
            r /= tot_l

        precisions.append(p)
        recalls.append(r)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return 0.0 if precision == recall == 0 else 2 * precision * recall / (precision + recall)


def calc_metrics_value(task, fn, n_sample=None):
    with open(fn) as f:
        res = [json.loads(i) for i in f.readlines()]
    s0_pred, s0_ref = [], []
    s1_pred, s1_ref = [], []

    for d in res:
        if d['style'] == 0:
            s0_ref.append([list(d['resp'])])
            s0_pred.append(list(d['pred_style0'][0]))
        else:
            s1_ref.append([list(d['resp'])])
            s1_pred.append(list(d['pred_style1'][0]))
    
    if n_sample:
        assert len(s0_ref) >= n_sample
        assert len(s1_ref) >= n_sample
        sampled_idxs = sample(range(len(s0_ref)), n_sample)
        s0_ref = [x for i, x in enumerate(s0_ref) if i in sampled_idxs]
        s0_pred = [x for i, x in enumerate(s0_pred) if i in sampled_idxs]
        sampled_idxs = sample(range(len(s1_ref)), n_sample)
        s1_ref = [x for i, x in enumerate(s1_ref) if i in sampled_idxs]
        s1_pred = [x for i, x in enumerate(s1_pred) if i in sampled_idxs]

    bleu_s0 = eval_bleu_detail(s0_ref, s0_pred)
    bleu_s1 = eval_bleu_detail(s1_ref, s1_pred)
    dist_s0 = eval_distinct_detail(s0_pred)
    dist_s1 = eval_distinct_detail(s1_pred)
    f1_s0 = eval_f1(s0_ref, s0_pred)
    f1_s1 = eval_f1(s1_ref, s1_pred)

    for k in range(1, 4):
        print('%d-gram BLEU:' % k,
              's0', bleu_s0[k - 1] * 100,
              's1', bleu_s1[k - 1] * 100,
              'mean', (bleu_s0[k - 1] + bleu_s1[k - 1]) / 2 * 100)
    print('F1:',
          's0', f1_s0 * 100, 's1', f1_s1 * 100,
          'mean', (f1_s0 + f1_s1) / 2 * 100)
    print('Dist:',
          's0', dist_s0[1] * 100, 's1', dist_s1[1] * 100,
          'mean', (dist_s0[1] + dist_s1[1]) / 2 * 100)


parser = argparse.ArgumentParser()
parser.add_argument('--eval_file_path', help='path of the eval file', required=True)
args = parser.parse_args()
file_path = args.eval_file_path

calc_metrics_value(None, file_path)
print("Evaluating acc results:")
bert_eval_acc.main(file_path)
svm_eval_acc.main(file_path)
