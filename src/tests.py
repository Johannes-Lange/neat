from src.registry import Registry
from src.base_classes import Node, Connection
from src.genom import Genom, crossover_genes, distance
from src.graph import Graph
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

    delta = distance(g1, g2)

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

    out = None
    for _ in range(10):
        out = g1.forward(x,)
        out = g1.forward_rec(x, )
    print(out)


def test_graph():
    nodes = [Node(1, 'input'), Node(2, 'input'), Node(3, 'output'), Node(4, 'output'), Node(5), Node(6), Node(7)]

    connections = [Connection(nodes[0], nodes[4], id_=1), Connection(nodes[4], nodes[5], id_=2),
                   Connection(nodes[5], nodes[2], id_=3), Connection(nodes[1], nodes[6], id_=4),
                   Connection(nodes[6], nodes[3], id_=5), Connection(nodes[5], nodes[6], id_=6),
                   Connection(nodes[6], nodes[4], id_=7)]

    graph = Graph(nodes, connections)
    graph.get_computation_order()
    print(graph.computation_order)
    # print(graph.connections)


def test_forward_new():
    reg = Registry(2, 2)

    # 3 hidden nodes
    for _ in range(3):
        reg.create_node()

    nodes = reg.nodes

    connections = [reg.get_connection(n1, n2) for n1, n2 in [(nodes[0], nodes[5]), (nodes[5], nodes[6]),
                                                            (nodes[6], nodes[3]), (nodes[1], nodes[7]),
                                                            (nodes[7], nodes[5]), (nodes[6], nodes[7]),
                                                            (nodes[7], nodes[4])]]

    # connections = [Connection(nodes[0], nodes[5]), Connection(nodes[5], nodes[6]), Connection(nodes[6], nodes[3]),
    #                Connection(nodes[1], nodes[7]), Connection(nodes[7], nodes[5]), Connection(nodes[6], nodes[7]),
    #                Connection(nodes[7], nodes[4])]

    g1 = Genom(reg, connections)
    for c in g1.connections:
        c.weight = 1

    print('recurrent network, output should be different')
    out = g1.forward_rec(np.array([1, 0]))
    print(out)
    out = g1.forward_rec(np.array([1, 0]))
    print(out)


test_graph()

# [1 (input), 1: (1) --> (5), 2 (input), 4: (2) --> (7), 7 (hidden), 5: (7) --> (4), 7: (7) --> (5),
# 5 (hidden), 2: (5) --> (6), 6 (hidden), 3: (6) --> (3), 3 (output), 4 (output), 6: (6) --> (7)]
