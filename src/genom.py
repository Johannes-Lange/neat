from __future__ import annotations
from src.base_classes import Node, Connection
from src.registry import Registry
from copy import deepcopy
import random
import numpy as np


class Genom:
    def __init__(self, reg: Registry):
        # global registry to get nodes and connections from
        self.registry = reg

        # nodes and connections
        self.nodes = deepcopy(self.registry.nodes)
        self.connections = []

        # Fitness
        self.fitness = None

    def get_gene(self):
        sorted_connections = sorted(self.connections, key=lambda x: x.id)
        ret = dict()
        for c in sorted_connections:
            ret[c.id] = c
        return ret

    def add_connection(self):
        c = self.registry.get_connection(*random.choices(self.nodes, k=2))
        self.connections.append(c)

    def crossover(self, o: Genom):
        pass

    def mutate_link(self):
        # randomly add connection with weight [-2, 2]
        pass

    # def add_connection(self):
    #     # add connection to unconnected nodes
    #     pass

    def add_node(self):
        # add node in connection, old connection disabled
        pass

    def enable_disable_connection(self):
        # randomly enable/disable connections
        pass

    def weight_shift(self):
        # weight * [0, 2]
        pass

    def forward(self, x):
        # TODO
        pass


def crossover_genes(p1: Genom, p2: Genom):
    assert (p1.fitness is not None) and (p2.fitness is not None), 'Fitness has to be evaluated before crossover'
    p_high, p_low = (p1, p2) if p1.fitness > p2.fitness else (p2, p1)

    # get boths list of connections
    c_high = p_high.get_gene()
    c_low = p_low.get_gene()

    # align genes
    id_low = list(c_low.keys())
    id_high = list(c_high.keys())

    max_id = max(id_low+id_high)

    new = []
    for cid in range(max_id + 1):
        # connection in both genes
        if cid in id_low and cid in id_high:
            if random.random() < 0.5:
                new.append(c_low[cid])
                continue
            else:
                new.append(c_high[cid])
                continue

        # disjoint and excess
        elif cid in id_high:
            new.append(c_high[cid])
            continue

        return new


