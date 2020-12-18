import re
from zhon.hanzi import punctuation
import json


def extract_jinyong():
    src_file_name1 = "data/continuous/jinyong_all_zh2en.txt"
    dst_file_name = "data/jinyong_dialogue_text.txt"

    src1 = []
    with open(src_file_name1, encoding='utf-8') as f:
        for line in f:
            src1.append(line.strip().split('\t')[0])
    src1.append("")

    original_text_set = set()
    with open("data/evaluation/jinyong_2k.txt", encoding='utf-8') as f:
        l = f.read().strip().split('\n')
    for p in l:
        p = p.split('\t')
        original_text_set.add(p[2])
        original_text_set.add(p[3])

    tot_len = 0
    tot_sentences = 0
    tot_dropped_sentences = 0
    with open(dst_file_name, 'w', encoding='utf-8') as f:
        p = re.compile(r".*“(.+)”.*")
        for i in range(len(src1)):
            m1 = p.match(src1[i])
            if m1 and m1.group(1)[-1] in punctuation:
                t = m1.group(1)
                if src1[i] not in original_text_set:
                    f.write("%s\n" % t)
                    tot_len += len(t)
                    tot_sentences += 1
                else:
                    tot_dropped_sentences += 1

    print(tot_sentences, tot_len / tot_sentences, tot_dropped_sentences)


def extract_xiyou():
    src_file_name = "../data/continuous/xiyou_back.txt"
    dst_file_name = "../data/evaluation/xiyou.txt"

    src1, src2 = [], []
    p = re.compile(r'{"source": "(.+)", "inter": ".+", "target": "(.+)"}')
    with open(src_file_name, encoding='utf-8') as f:
        for line in f:
            m = p.match(line.strip())
            src1.append(m.group(1))
            src2.append(m.group(2))

    tot_len = 0
    tot_dialogues = 0
    tot_sentences = 0
    tot_sentence_len = 0
    with open(dst_file_name, 'w', encoding='utf-8') as f:
        dst = []
        p1 = re.compile(r".*“(.+)”.*")
        p2 = re.compile(r".*“(.+)”.*")
        for i in range(len(src1)):
            m1 = p1.match(src1[i])
            m2 = p2.match(src2[i])
            if m1 and m2 and m1.group(1)[-1] in punctuation and m2.group(1)[-1] in punctuation:
                dst.append((m1.group(1), m2.group(1), src1[i], src2[i]))
            else:
                if len(dst) >= 2:
                    for e in dst:
                        f.write("\t".join(e) + "\n")
                        tot_sentences += 1
                        tot_sentence_len += len(e[0])
                    f.write("\n")
                    tot_len += len(dst)
                    tot_dialogues += 1
                dst = []

    print("Dialogues extracted:", tot_dialogues)
    print("Average dialogue sentences:", tot_len / tot_dialogues)
    print("Average sentence length:", tot_sentence_len / tot_sentences)


def extract_qiongyao():
    src_file_name = "../data/continuous/qiongyao_back.txt"
    dst_file_name = "../data/evaluation/qiongyao.txt"

    src1, src2 = [], []
    p = re.compile(r'{"source": "(.+)", "inter": ".+", "target": "(.+)"}')
    with open(src_file_name, encoding='utf-8') as f:
        for line in f:
            m = p.match(line.strip())
            src1.append(m.group(1))
            src2.append(m.group(2))

    tot_len = 0
    tot_dialogues = 0
    tot_sentences = 0
    tot_sentence_len = 0
    with open(dst_file_name, 'w', encoding='utf-8') as f:
        dst = []
        p1 = re.compile(r".*“(.+)”.*")
        p2 = re.compile(r".*“(.+)”.*")
        for i in range(len(src1)):
            m1 = p1.match(src1[i])
            m2 = p2.match(src2[i])
            if m1 and m2 and m1.group(1)[-1] in punctuation and m2.group(1)[-1] in punctuation:
                dst.append((m1.group(1), m2.group(1), src1[i], src2[i]))
            else:
                if len(dst) >= 2:
                    for e in dst:
                        f.write("\t".join(e) + "\n")
                        tot_sentences += 1
                        tot_sentence_len += len(e[0])
                    f.write("\n")
                    tot_len += len(dst)
                    tot_dialogues += 1
                dst = []

    print("Dialogues extracted:", tot_dialogues)
    print("Average dialogue sentences:", tot_len / tot_dialogues)
    print("Average sentence length:", tot_sentence_len / tot_sentences)


extract_jinyong()
