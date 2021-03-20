import numpy as np

input_set = np.random.uniform(-1, 1, (1000, 2))


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
        return .1
    else:
        return -.1
