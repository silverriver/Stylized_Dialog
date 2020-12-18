from collections import defaultdict
import numpy as np


def eval_entropy(hyps, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    if type(hyps[0][0]) is int:
        hyps = [[str(j) for j in i] for i in hyps]
    for line in hyps:
        for n in range(4):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v / total * (np.log(v) - np.log(total))

    return np.asarray(etp_score)
