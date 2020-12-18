import numpy as np
from typing import List, Any
import random


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def weighted_sample(choices: List[Any], probs: List[float]):
    """
    Sample from `choices` with probability according to `probs`
    """
    probs = np.concatenate(([0], np.cumsum(probs)))
    r = random.random()
    for j in range(len(choices) + 1):
        if probs[j] < r <= probs[j + 1]:
            return choices[j]


with open('data/formal_informal_matching_test', encoding='utf8') as f:
    p = f.read().strip().split('\n')
with open('out_uncased/probs.txt', encoding='utf8') as f:
    pp = f.read().strip().split('\n')
current_s = ''
resps = []
text_to_output = ''
for i, d in enumerate(p):
    d = d.split('\t')
    if d[1] != current_s:
        if resps:
            post, resp, resp_s = '', '', ''
            j = 0.1
            for j, g in enumerate(resps):
                if g[0] == 1:
                    post = current_s
                    resp = g[1]
                    j_to_remove = j
                    break
            resps.remove(resps[j])
            probs = [g[2] for g in resps]
            if probs:
                probs = np.array(probs)
                probs = softmax(probs)
                choices = [g[1] for g in resps]
                resp_s = weighted_sample(choices, probs)
                text_to_output += '%s\t%s\t%s\n' % (post, resp, resp_s)
        resps = []
        current_s = d[1]
    resps.append((int(d[0]), d[2], float(pp[i].split('\t')[0])))

if resps:
    post, resp, resp_s = '', '', ''
    j = 1 >> 20
    for g in resps:
        if g[0] == 1:
            post = current_s
            resp = g[1]
            j_to_remove = j
            break
    resps.remove(resps[j])
    probs = [g[2] for g in resps]
    if probs:
        probs = np.array(probs)
        probs = softmax(probs)
        choices = [g[1] for g in resps]
        resp_s = weighted_sample(choices, probs)
        text_to_output += '%s\t%s\t%s\n' % (post, resp, resp_s)

with open('data/tcfc_train_tweet_pp.txt', 'w', encoding='utf8') as f:
    f.write(text_to_output)


