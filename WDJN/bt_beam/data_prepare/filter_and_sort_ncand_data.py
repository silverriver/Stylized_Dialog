import os
from collections import Counter
import copy
dialog_file = 'data/crowded_300k_jinyong_dialog.txt'
prob_file = 'data/crowded_300k_jinyong_dialog_probs.txt'

dialog_file_out = 'data/crowded_300k_jinyong_dialog_filter_sort.txt'
prob_file_out = 'data/crowded_300k_jinyong_dialog_probs_filter_sort.txt'

dialogs = []
with open(prob_file) as f:
    probs = [float(i.strip().split('\t')[0]) for i in f.readlines()]

with open(dialog_file) as f:
    res = [i.strip() for i in f.readlines()]
    print('{} lines read'.format(len(res)))

tmp = []
total_len = len(res)
for i in range(total_len):
    if res[i][0] == '0' and len(tmp) != 0:
        dialogs.append(copy.deepcopy(tmp))
        tmp = []
    tmp.append((res[i], probs[i]))

len_counter = Counter([len(i) for i in dialogs])
print('{} post obtained'.format(len(dialogs)))
common_len = len_counter.most_common()[0][0]
print('{} cand is most common'.format(common_len - 1))

dialogs = [i for i in dialogs if len(i) == common_len]
print('{} post remains'.format(len(dialogs)))


for i in range(len(dialogs)):
    dialogs[i][1:] = sorted(dialogs[i][1:], key=lambda x: -x[1])

with open(dialog_file_out, 'w') as d_f, open(prob_file_out, 'w') as p_f:
    for d in dialogs:
        print(d[0][0], file=d_f)
        print(d[0][1], file=p_f)
        for i in d[1:]:
            print(i[0], file=d_f)
            print(i[1], file=p_f)

print('fin.')
