from __future__ import annotations
from src.base_classes import Connection
from src.registry import Registry
from src.graph import Graph
from copy import deepcopy
import random
import numpy as np

C1, C2, C3 = 1, 1, .4
P_WEIGHT = .8
P_ENDISABLE = .75
P_NEW_NODE = .03
P_NEW_LINK = .05


class Genom:
    def __init__(self, reg: Registry, connections: [Connection] = None):
        # global registry to get nodes and connections from
        self.registry = reg

        # for ordering
        self.net_graph, self.computation_order = None, None
        self.ready = False

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

    def forward(self, x: np.array, verbose: bool = False):
        if not self.ready:
            self.sort_nodes_connections()
            self.set_next_nodes()
            self.ready = True

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

    def forward_new(self, x: np.array):
        if not self.ready:
            self.ordered_graph()
        assert x.size == self.registry.inputs, 'size of input {x.size} must match input nodes {self.registry.inputs}'

        # assign input values from x
        for i, n in enumerate(self.nodes):
            if n.type != 'input':
                break
            n.in_val = x[i]

        for ob in self.computation_order:
            ob.eval_ordered()

        for n in self.nodes:
            n.in_val = 0

        return self.get_output()

    def get_output(self):
        out = self.nodes[self.registry.inputs+1:self.registry.inputs+1+self.registry.outputs]
        ret = np.array([n.out_val for n in out])
        return softmax(ret)

    def initial_connections(self):
        self.ready = False
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

    """ *** Mutations *** """
    def apply_mutations(self):
        # TODO: order?
        self.mutate_link()
        self.mutate_add_node()
        self.mutate_weight_shift()
        self.mutate_enable_disable_connection()

    # add a new connection randomly
    def mutate_link(self):
        self.ready = False
        # recurrent connections possible!
        if random.random() < 1-P_NEW_LINK:
            return
        while True:
            n1 = random.choice(self.nodes)
            n2 = random.choice(self.nodes)
            if (n1.type != 'output') and (n2.type != 'bias'):
                break
        c = self.registry.get_connection(n1, n2)
        c.rand_weight()
        self.connections.append(c)
        self.sort_nodes_connections()

    # add node in connection, old connection disabled
    def mutate_add_node(self):
        self.ready = False
        # register a new node
        if random.random() < 1-P_NEW_NODE:
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

    # randomly enable/disable a connection
    def mutate_enable_disable_connection(self):
        self.ready = False
        if random.random() < 1-P_ENDISABLE:
            return
        c = random.choice(self.connections)
        c.en_dis_able()
        self.sort_nodes_connections()

    # shift weight
    def mutate_weight_shift(self):
        self.ready = False
        # weight * [0, 2]
        if random.random() < 1-P_WEIGHT:
            return
        for c in self.connections:
            c.weight = random.uniform(0, 2) * c.weight
        self.sort_nodes_connections()

    """ *** prepare for forward *** """
    def sort_nodes_connections(self):
        self.connections = sorted(self.connections, key=lambda x: x.id)
        self.nodes = sorted(self.nodes, key=lambda x: x.id)

    def ordered_graph(self):
        # get ordered nodes and connections
        self.sort_nodes_connections()
        self.set_next_nodes()
        self.net_graph = Graph(self.nodes, self.connections)
        self.computation_order = self.net_graph.get_computation_order()

        self.ready = True

    def set_next_nodes(self):
        # find nodes from connection in self.nodes
        for c in self.connections:
            n1 = next(x for x in self.nodes if x == c.n1)
            n2 = next(x for x in self.nodes if x == c.n2)

            if n2 not in list(map(lambda x: x[0], n1.next_nodes)):
                n1.next_nodes.append((n2, c.enabled, c.weight))


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

    # TODO: does this work?
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


def distance(g1: Genom, g2: Genom):
    # dictionary with id as key and connection
    cons1, cons2 = g1.get_gene(), g2.get_gene()

    # innovation numers
    id1 = sorted(list(cons1.keys()))
    id2 = sorted(list(cons2.keys()))
    max_id = max(id1 + id2)

    # calculate disjoint and excess
    match = []
    disjoint = 0
    excess = abs(max(id2) - max(id1))
    for cid in range(max_id + 1 - excess):
        if (cid in id1) and (cid in id2):
            match.append(cid)
        if (cid in id1) or (cid in id2):
            disjoint += 1

    # calculate distance
    w_diff = [abs(cons1[cid].weight - cons2[cid].weight) for cid in match]
    w_diff = save_devide(w_diff)
    n = max(len(id1), len(id2))
    return C1 * excess / n + C2 * disjoint / n + C3 * w_diff


def save_devide(x):
    if len(x) == 0:
        return 0.
    else:
        return sum(x) / len(x)


def softmax(x):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
