import random


if __name__ == '__main__':
    sentences = []
    with open('../data/crowded_300k.txt', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            sentences.append(line[1])
            sentences.append(line[2])
    random.shuffle(sentences)
    sentences = sentences[: 267132]
    with open('../../data/plain_text/dialogue_f.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences) + '\n')
