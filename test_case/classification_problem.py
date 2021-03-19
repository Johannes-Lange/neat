import numpy as np

input_set = np.random.uniform(-1, 1, (100000, 2))


def inside_circle(x, y):
    ret = np.zeros(2)
    ret[0] = 1 if x**2 + y**2 <= 1 else ret[1] = 1
    return ret


def inside(dataset):
    ins = 0
    for i in range(dataset.shape[0]):
        if inside_circle(*dataset[i]):
            ins += 1
    print(ins/dataset.shape[0])
