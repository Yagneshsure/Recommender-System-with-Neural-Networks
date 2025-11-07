import numpy as np

def recall_at_k(actual, predicted, k):
    predicted = predicted[:k]
    return int(any(a in predicted for a in actual))

def ndcg_at_k(actual, predicted, k):
    predicted = predicted[:k]
    if not actual:
        return 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            return 1 / np.log2(i + 2)
    return 0.0
