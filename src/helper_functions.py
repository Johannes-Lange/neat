import numpy as np


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    n = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/n
    return ce


def save_devide(x):
    if len(x) == 0:
        return 0.
    else:
        return sum(x) / len(x)


def softmax(x):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
