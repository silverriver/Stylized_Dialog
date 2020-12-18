text_infile = 'continuous/jinyong_all_zh2en.txt'
jinyong_dialog_eval_infile = 'data/jinyong_2k_eval.txt'
outfile = 'data/jinyong_continuous_text.txt'

with open(jinyong_dialog_eval_infile) as f:
    tmp = [i.strip().split('\t') for i in f.readlines()]

dialog = [i[-2:] for i in tmp]
sent2index = dict([(d[0], i) for d, i in zip(dialog, range(len(dialog)))])

print('len(dialog):', len(dialog))

with open(text_infile) as f:
    text = [i.strip().split('\t')[0] for i in f.readlines()]

print('len(text):', len(text))
filtered_text = []
i = 0
text_len = len(text) - 1
while i < text_len:
    if text[i] in sent2index:
        index = sent2index[text[i]]
        if text[i+1] == dialog[index][1]:
            del sent2index[text[i]]
            i += 2
            continue
    filtered_text.append(text[i])
    i += 1

print('len(filtered_text):', len(filtered_text))

with open(outfile, 'w') as f:
    for i in filtered_text:
        f.write(i)

