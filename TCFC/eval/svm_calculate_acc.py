from do_tokenize import do_tokenize
import pickle


with open('../data/cls', 'rb') as f:
    cls = pickle.load(f)


def calculate_acc(list_of_str, label):
    sentences = do_tokenize(list_of_str)
    result = cls.predict(sentences)
    return sum(result == label) / (len(result) + 1e-8)
