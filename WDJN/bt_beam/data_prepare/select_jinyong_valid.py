import random


with open('data/evaluation/jinyong.txt', encoding='utf8') as f:
    l = f.read().strip().split('\n')
r = []
for i in range(len(l) - 1):
    if l[i] and l[i + 1]:
        a = l[i].split('\t')
        b = l[i + 1].split('\t')
        r.append((a[1], b[0], a[2], b[2]))
random.shuffle(r)
r = r[: 2000]


with open('data/evaluation/jinyong_2k.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(['\t'.join(p) for p in r]) + '\n')
