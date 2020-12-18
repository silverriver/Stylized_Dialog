import json
from svm_calculate_acc import calculate_acc


def main(file_path):
    with open(file_path, encoding='utf8') as f:
        d = f.read().strip().split('\n')
        j = json.loads(d[0])
        if 'pred_style0' in j:
            sz = len(j['pred_style0'])
        else:
            sz = 0

    acc0 = 0
    for i in range(sz):
        all_pred_str = []
        with open(file_path, encoding='utf8') as f:
            d = f.read().strip().split('\n')
            for s in d:
                j = json.loads(s)
                pred = j['pred_style0'][i]
                all_pred_str.append(pred)
        acc0 = (acc0 * i + calculate_acc(all_pred_str, 0)) / (i + 1)

    with open(file_path, encoding='utf8') as f:
        d = f.read().strip().split('\n')
        j = json.loads(d[0])
        if 'pred_style1' in j:
            sz = len(j['pred_style1'])
        else:
            sz = 0

    acc1 = 0
    for i in range(sz):
        all_pred_str = []
        with open(file_path, encoding='utf8') as f:
            d = f.read().strip().split('\n')
            for s in d:
                j = json.loads(s)
                pred = j['pred_style1'][i]
                all_pred_str.append(pred)
        acc1 = (acc1 * i + calculate_acc(all_pred_str, 1)) / (i + 1)

    print('SVM:', 's0', acc0 * 100, 's1', acc1 * 100, 'mean', (acc0 + acc1) / 2 * 100)
