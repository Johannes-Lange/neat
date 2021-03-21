import numpy as np
import random

INPUT_CIRC = np.random.uniform(-1, 1, (100, 2))


INPUT_XOR = np.random.randint(0, 2, size=(100, 2))


def inside_circle(x, y):
    ret = np.zeros(2)
    if x**2 + y**2 <= 1:
        ret[0] = 1
    else:
        ret[1] = 1
    return ret


def inside(x, y, prediction):
    label = np.argmax(inside_circle(x, y))
    pred = np.argmax(prediction)
    if label == pred:
        return 1
    else:
        return 0


def xor(x, y, pred):
    # if pred[0] == pred[1]:
    #     return 0
    pred = np.argmax(pred)
    if x != y:
        label = 1
    else:
        label = 0
    return 1 if pred == label else 0
