from metrics.eval_distinct import eval_distinct, eval_distinct1, eval_distinct2
from metrics.eval_bleu import eval_bleu
from metrics.eval_f1 import eval_f1
from metrics.eval_entropy import eval_entropy
import json
from model.tokenization_gpt2 import GPT2Tokenizer
import numpy as np
import argparse
import bert_eval_acc
import svm_eval_acc


parser = argparse.ArgumentParser()
parser.add_argument('--eval_file_path', help='path of the eval file', required=True)
args = parser.parse_args()
file_path = args.eval_file_path


tokenizer = GPT2Tokenizer(vocab_file='../data/DialoGPT/small/vocab.json',
                          merges_file='../data/DialoGPT/small/merges.txt')
tokenizer.init_kwargs = {'vocab_file': '../data/DialoGPT/small/vocab.json',
                         'merges_file': '../data/DialoGPT/small/merges.txt'}
tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))


with open(file_path, encoding='utf8') as f:
    d = f.read().strip().split('\n')
    j = json.loads(d[0])
    if 'pred_style0' in j:
        sz = len(j['pred_style0'])
    else:
        sz = 0

distinct0, f10 = 0, 0
ngram_bleus0 = [0, 0, 0]

model_file_name = 'model_file.bin'
pred_file_name = 'pred_temp.txt'

for i in range(sz):
    all_refs = []
    all_pred = []
    all_pred_str = []
    with open(file_path, encoding='utf8') as f:
        d = f.read().strip().split('\n')
        for s in d:
            j = json.loads(s)
            refs = j['resp_style0']
            refs = [tokenizer.encode(r) for r in refs]
            pred = j['pred_style0'][i]
            all_pred_str.append(pred)
            pred = tokenizer.encode(pred)
            all_refs.append(refs)
            all_pred.append(pred)

    distinct0 = (distinct0 * i + eval_distinct2(all_pred)) / (i + 1)
    f10 = (f10 * i + eval_f1(all_refs, all_pred)) / (i + 1)
    for k in range(1, 4):
        ngram_bleus0[k - 1] = (ngram_bleus0[k - 1] * i + eval_bleu(all_refs, all_pred, n_gram_only=k)) / (i + 1)


with open(file_path, encoding='utf8') as f:
    d = f.read().strip().split('\n')
    j = json.loads(d[0])
    if 'pred_style1' in j:
        sz = len(j['pred_style1'])
    else:
        sz = 0

distinct1, f11 = 0, 0
ngram_bleus1 = [0, 0, 0]


for i in range(sz):
    all_refs = []
    all_pred = []
    all_pred_str = []
    with open(file_path, encoding='utf8') as f:
        d = f.read().strip().split('\n')
        for s in d:
            j = json.loads(s)
            refs = j['resp_style1']
            refs = [tokenizer.encode(r) for r in refs]
            pred = j['pred_style1'][i]
            all_pred_str.append(pred)
            pred = tokenizer.encode(pred)
            all_refs.append(refs)
            all_pred.append(pred)

    distinct1 = (distinct1 * i + eval_distinct2(all_pred)) / (i + 1)
    f11 = (f11 * i + eval_f1(all_refs, all_pred)) / (i + 1)
    for k in range(1, 4):
        ngram_bleus1[k - 1] = (ngram_bleus1[k - 1] * i + eval_bleu(all_refs, all_pred, n_gram_only=k)) / (i + 1)


for k in range(1, 4):
    print('%d-gram BLEU:' % k,
          's0', ngram_bleus0[k - 1] * 100,
          's1', ngram_bleus1[k - 1] * 100,
          'mean', (ngram_bleus0[k - 1] + ngram_bleus1[k - 1]) / 2 * 100)
print('F1:',
      's0', f10 * 100, 's1', f11 * 100,
      'mean', (f10 + f11) / 2 * 100)
print('Dist:',
      's0', distinct0 * 100, 's1', distinct1 * 100,
      'mean', (distinct0 + distinct1) / 2 * 100)
print("Evaluating acc results:")
bert_eval_acc.main(file_path)
svm_eval_acc.main(file_path)