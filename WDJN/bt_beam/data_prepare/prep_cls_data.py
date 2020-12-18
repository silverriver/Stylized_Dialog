d_file = 'data/crowded_300k_split.txt'
jinyong_file = 'data/jinyong_text_train.txt'

with open(d_file) as f:
    d = [i.strip().split('\t') for i in f.readlines()]

