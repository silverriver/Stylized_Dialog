#!/usr/bin/python
# -*- coding: utf-8 -*-
import tqdm

if __name__ == '__main__':
    evaluation_files = ['../../data/evaluation/%s_sampled.txt' % s for s in ['jinyong', 'honglou', 'xiyou']]
    evaluation_sentences = []
    for file_name in evaluation_files:
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                evaluation_sentences += line.strip().split('\t')
    text_files = ['../../data/plain_text/%s.txt' % s for s in ['jinyong', 'honglou', 'xiyou']]
    filtered_files = ['../../data/plain_text/%s_f.txt'  % s for s in ['jinyong', 'honglou', 'xiyou']]
    for file_name, filtered_file_name in zip(text_files, filtered_files):
        n_deleted_sentences = 0
        sentences = []
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                # text = text.replace(' ', '').replace(r'\"', r'"').replace('”', r'"').replace('“', r'"')
                sentences.append(text)
        with open(filtered_file_name, 'w', encoding='utf-8') as f:
            for text in tqdm.tqdm(sentences):
                if text in evaluation_sentences:
                    n_deleted_sentences += 1
                    # print('deleted: %s' % text)
                else:
                    f.write('%s\n' % text)
        print(n_deleted_sentences)
