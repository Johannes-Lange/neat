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

    g1.set_next_nodes()
    out = g1.forward(x, verbose=True)
    print(out)


def test_graph():
    nodes = []
    nodes.append(Node(1, 'input'))
    nodes.append(Node(2, 'input'))
    nodes.append(Node(3, 'output'))
    nodes.append(Node(4, 'output'))
    nodes.append(Node(5))
    nodes.append(Node(6))
    nodes.append(Node(7))

    connections = []
    connections.append(Connection(nodes[0], nodes[4], id_=1))
    connections.append(Connection(nodes[4], nodes[5], id_=2))
    connections.append(Connection(nodes[5], nodes[2], id_=3))
    connections.append(Connection(nodes[1], nodes[6], id_=4))
    connections.append(Connection(nodes[6], nodes[3], id_=5))
    connections.append(Connection(nodes[5], nodes[6], id_=6))
    connections.append(Connection(nodes[6], nodes[4], id_=7))

    graph = Graph(nodes, connections)
    graph.algo()
    print(graph.order)
    print(graph.recurrent)
    # print(graph.connections)

