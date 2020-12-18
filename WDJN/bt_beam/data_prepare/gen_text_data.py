import random


def gen_text_data(dialogue, jinyong, honglou, xiyou):
    data_groups = []
    for i, file_name in enumerate([dialogue, jinyong, honglou, xiyou]):
        with open(file_name, encoding='utf-8') as f:
            lines = [s.strip() for s in f.readlines() if len(s.strip()) != 0]
            data_groups.append(lines)
    max_sentences = len(max(data_groups, key=lambda x: len(x)))
    for i, d in enumerate(data_groups):   # to make number of sentences in each group equal
        d += [random.choice(d) for _ in range(max_sentences - len(d))]
    data = []
    for i, d in enumerate(data_groups):
        for s in d:
            data.append((i, s))
    random.shuffle(data)
    with open('../data/text.txt', 'w', encoding='utf-8') as f:
        for d in data:
            f.write('%d\t%s\n' % (d[0], d[1]))


if __name__ == '__main__':
    gen_text_data('../../data/plain_text/dialogue_f.txt',
                  '../../data/plain_text/jinyong_f.txt',
                  '../../data/plain_text/honglou_f.txt',
                  '../../data/plain_text/xiyou_f.txt')
