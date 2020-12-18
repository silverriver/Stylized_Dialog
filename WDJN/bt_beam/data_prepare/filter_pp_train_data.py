pp_dialog_file = 'data/pseudo_parallel.txt'
pp_probs_file = 'data/pseudo_parallel_probs.txt'

pp_dialog_file_out = 'data/pseudo_parallel_20candidate.txt'
pp_probs_file_out = 'data/pseudo_parallel_probs_20candidate.txt'
cand_count = 20

with open(pp_dialog_file) as f:
    dialog = [i.strip().split('\t') for i in f.readlines()]

with open(pp_probs_file) as f:
    probs = [i.strip().split('\t') for i in f.readlines()]

print('dialog', len(dialog))
print('probs', len(probs))

index1 = []
dialog_len = len(dialog)
i = 0
count = 0
while i < dialog_len:
    if probs[i][1] == '1':
        count = 0
    else:
        count += 1
        if count == cand_count:
            index1.append(i - cand_count)
    i += 1

print('{} dialogs have {} cand'.format(len(index1), cand_count))

with open(pp_dialog_file_out, 'w') as f:
    for i in index1:
        for j in range(cand_count + 1):
            print('\t'.join(dialog[j + i]), file=f)

with open(pp_probs_file_out, 'w') as f:
    for i in index1:
        for j in range(cand_count + 1):
            print('\t'.join(probs[j + i]), file=f)

print('fin.')

