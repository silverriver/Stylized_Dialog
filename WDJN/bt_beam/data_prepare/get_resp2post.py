import os
from collections import Counter
import copy
dialog_file = 'data/crowded_300k_jinyong_dialog_filter_sort.txt'
prob_file = 'data/crowded_300k_jinyong_dialog_probs_filter_sort.txt'
out_file = 'data/crowded_300k_jinyong_dialog_resp2post_5cand.txt'
n_cand = 5

dialog_data = []

with open(dialog_file) as f:
    res = [i.strip().split('\t') for i in f.readlines()]
    print('{} lines read'.format(len(res)))

i = 0
count = 0
dialog_len = len(res)
while i < dialog_len:
    if res[i][0] == '0':
        count = 0
        text_tmp = [res[i][1], res[i][2]]
    else:
        count += 1
        text_tmp.append(res[i][2])
        if count == n_cand:
            dialog_data.append(copy.copy(text_tmp))
    i += 1

print('{} post obtained'.format(len(dialog_data)))

with open(out_file, 'w') as f:
    for i in dialog_data:
        print('{}\t{}\t{}'.format(0, i[1], i[0]), file=f)
        for j in i[2:]:
            print('{}\t{}\t{}'.format(1, j, i[0]), file=f)

print('fin.')
