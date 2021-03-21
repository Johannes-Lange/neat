import numpy as np


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    n = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/n
    return ce


print(cross_entropy(np.array([.5, .5]), np.array([1, 0])))
