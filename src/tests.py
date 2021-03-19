from src.registry import Registry
from src.genom import Genom, crossover_genes
import numpy as np


def test_genom():
    reg = Registry(25, 5)

    g1 = Genom(reg)
    g2 = Genom(reg)

    g1.mutate_link()
    g2.mutate_add_node()
    g1.mutate_weight_shift()
    g2.mutate_enable_disable_connection()

    g1.fitness = 0
    g2.fitness = 1

    g3 = crossover_genes(g1, g2)


def test_forward():
    x = np.random.random(25)
    reg = Registry(25, 5)

    g1 = Genom(reg)
    for _ in range(1):
        g1.mutate_link()
        g1.mutate_add_node()
        g1.mutate_weight_shift()
        g1.mutate_enable_disable_connection()

    g1.set_next_nodes()
    out = g1.forward(x, verbose=True)
    print(out)
