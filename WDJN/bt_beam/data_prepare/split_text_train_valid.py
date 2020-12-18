import random


if __name__ == '__main__':
    lines = []
    with open('data/pseudo_parallel_all.txt', encoding='utf-8') as f:
        for line in f:
            lines.append(line)
    random.shuffle(lines)
    with open('data/pseudo_parallel_train.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(lines[: -2000]))
    with open('data/pseudo_parallel_valid.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(lines[-2000:]))
