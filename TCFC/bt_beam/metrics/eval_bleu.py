from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


smooth = SmoothingFunction()


def eval_bleu(ref, pred, n_gram_only=None):
    """
    :param ref: list(list(list(any))), a list of reference sentences, each element of the list is a list of references
    :param pred: list(list(any)), a list of predictions
    :return: corpus bleu score
    """
    if n_gram_only:
        if n_gram_only == 1:
            weights = [1, 0, 0, 0]
        elif n_gram_only == 2:
            weights = [0, 1, 0, 0]
        elif n_gram_only == 3:
            weights = [0, 0, 1, 0]
        elif n_gram_only == 4:
            weights = [0, 0, 0, 1]
        else:
            assert False
        bleu = corpus_bleu(ref, pred, weights=weights, smoothing_function=smooth.method1)
    else:
        bleu = corpus_bleu(ref, pred, smoothing_function=smooth.method1)
    if not 0 <= bleu <= 1:
        bleu = 0.0
    return bleu
