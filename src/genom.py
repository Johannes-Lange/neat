from __future__ import annotations
from src.base_classes import Connection
from src.registry import Registry
from copy import deepcopy
import random
import numpy as np


class Genom:
    def __init__(self, reg: Registry, connections: [Connection] = None):
        # global registry to get nodes and connections from
        self.registry = reg

        # nodes and connections
        self.nodes = deepcopy(self.registry.nodes[:self.registry.inputs + self.registry.outputs + 1])
        self.connections = []

        # Fitness
        self.fitness = None

        # random connections and weights
        if connections is None:
            self.initial_connections()
        # initialize with existing connections
        else:
            self.connections = connections
            for c in self.connections:
                if c.n1 not in self.nodes:
                    self.nodes.append(self.registry.get_node(c.n1.id))
                if c.n2 not in self.nodes:
                    self.nodes.append(self.registry.get_node(c.n2.id))
        self.sort_nodes_connections()

        # TODO: enable/disable connections

    def set_next_nodes(self):
        for c in self.connections:
            n1 = next(x for x in self.nodes if x == c.n1)
            n2 = next(x for x in self.nodes if x == c.n2)

            if n2 not in list(map(lambda x: x[0], n1.next_nodes)):
                n1.next_nodes.append((n2, c.enabled, c.weight))

    def forward(self, x: np.array, verbose: bool = False):
        if verbose:
            print('be sure to call set_next_nodes before forward')
        assert x.size == self.registry.inputs, 'size of input {x.size} must match input nodes {self.registry.inputs}'

        # assign input values from x
        for i, n in enumerate(self.nodes):
            if n.type != 'input':
                break
            n.in_val = x[i]

        # evaluate (n times for recurrent connections to converge)
        last_out = np.zeros(self.registry.outputs)
        cycle = 0
        while 1:
            cycle += 1
            for n in self.nodes:
                n.evaluate()
            out = self.get_output()
            if np.allclose(out, last_out, rtol=1.e-3) or cycle > 1000:
                break
            last_out = out
        if verbose:
            print('cycles to converge:', cycle)
        return out

    def get_output(self):
        out = self.nodes[self.registry.inputs+1:self.registry.inputs+1+self.registry.outputs]
        ret = np.array([n.out_val for n in out])
        return softmax(ret)

    def initial_connections(self):
        inputs = self.nodes[:self.registry.inputs+1]
        outputs = self.nodes[self.registry.inputs + 1:]
        for _ in range(len(outputs)):
            n1 = random.choice(inputs)
            assert n1.type in ['input', 'bias']
            n2 = outputs.pop(0)
            assert n2.type == 'output'
            c = self.registry.get_connection(n1, n2)
            c.rand_weight()
            self.connections.append(c)

    def get_gene(self):
        ret = dict()
        for c in self.connections:
            ret[c.id] = c
        return ret

    def mutate_link(self):
        # add a new connection randomly, chance is 5%
        # recurrent connections possible!
        if random.random() < .95:
            return
        while True:
            n1 = random.choice(self.nodes)
            n2 = random.choice(self.nodes)
            if (n1.type != 'output') and (n2.type != 'bias'):
                break
        c = self.registry.get_connection(n1, n2)
        c.rand_weight()
        self.connections.append(c)

    def mutate_add_node(self):
        # add node in connection, old connection disabled, chance is 3%
        # register a new node
        if random.random() < .97:
            return
        n_add = self.registry.create_node()

        # select random connection, disable it
        c = random.choice(self.connections)
        c.disable()

        # create two new connections instead
        c1_add = self.registry.get_connection(c.n1, n_add)
        c1_add.weight = 1
        c2_add = self.registry.get_connection(n_add, c.n2)
        c2_add.weight = c.weight

        # append new connections and node
        self.nodes.append(n_add)
        self.connections.append(c1_add)
        self.connections.append(c2_add)
        self.sort_nodes_connections()

    def mutate_enable_disable_connection(self):
        # randomly enable/disable a connection
        c = random.choice(self.connections)
        c.en_dis_able()

    def mutate_weight_shift(self):
        # chance 80%, then eac weight
        # weight * [0, 2]
        if random.random() < .8:
            for c in self.connections:
                c.weight = random.uniform(0, 2) * c.weight

    def sort_nodes_connections(self):
        self.connections = sorted(self.connections, key=lambda x: x.id)
        self.nodes = sorted(self.nodes, key=lambda x: x.id)


def crossover_genes(p1: Genom, p2: Genom):
    assert (p1.fitness is not None) and (p2.fitness is not None), 'Fitness has to be evaluated before crossover'
    p_high, p_low = (p1, p2) if p1.fitness > p2.fitness else (p2, p1)

    # get boths list of connections
    c_high = p_high.get_gene()
    c_low = p_low.get_gene()

    # align genes
    id_low = sorted(list(c_low.keys()))
    id_high = sorted(list(c_high.keys()))

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
    child = Genom(p1.registry, new)
    return child


def softmax(x):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
