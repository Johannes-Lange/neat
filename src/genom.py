from __future__ import annotations
from src.helper_functions import softmax, save_devide
from src.base_classes import Connection
from src.registry import Registry
from src.graph import Graph
from copy import deepcopy
import random
import numpy as np

C1, C2, C3 = 1, 1, .4
P_WEIGHT = .8
P_ENDISABLE = .7
P_NEW_NODE = .03  # 0.03
P_NEW_LINK = .01  # 0.01


class Genom:
    def __init__(self, reg: Registry, connections: list[Connection] = None):
        # global registry to get nodes and connections from
        self.registry = reg

        # for ordering
        self.net_graph, self.computation_order = None, None
        self.ready = False
        self.normal_ops = 0

        # nodes and connections
        self.nodes = []
        self.connections = []

        # Fitness
        self.fitness = None

        # random connections and weights
        if connections is None:
            self.initial_connections()
        # initialize with existing connections
        else:
            self.inherit_connections(connections)

    def forward(self, x: np.array):
        if not self.ready:
            self.ordered_graph()

        for n in self.nodes:
            n.reset_vals()

        assert x.size == self.registry.inputs, 'size of input {x.size} must match input nodes {self.registry.inputs}'

        # assign input values from x
        for i, n in enumerate([no for no in self.nodes if no.type == 'input']):
            n.in_val = x[i]

        for ob in self.computation_order:
            ob.eval_ordered()

        out = self.get_output()

        return out

    def forward_rec(self, x: np.array):
        if not self.ready:
            self.ordered_graph()

        assert x.size == self.registry.inputs, 'size of input {x.size} must match input nodes {self.registry.inputs}'

        # clear input values for all nodes
        for n in self.nodes:
            n.reset_vals()

        # assign input values from x
        for i, n in enumerate([no for no in self.nodes if no.type == 'input']):
            n.in_val = x[i]

        # calculate all recurrent connections first (output also recurrent, so do this later)
        for ob in self.computation_order[self.normal_ops:]:
            if isinstance(ob, Connection):
                if ob.n2.type == 'output':
                    print('dd')
                    continue
            else:
                raise ValueError('there shouldn\'t be a node in recurrent operations list... some error in graph.py')
            ob.eval_recurrent()

        # standard operations in ordered graph
        for ob in self.computation_order[:self.normal_ops]:
            ob.eval_ordered()

        out = self.get_output()
        # print(out)
        return out

    def get_output(self):
        out = [n for n in self.nodes if n.type == 'output']
        ret = np.array([n.out_val for n in out])
        # print(ret)
        return softmax(ret)

    def initial_connections(self):
        self.ready = False

        self.nodes = deepcopy(self.registry.nodes[:self.registry.inputs + self.registry.outputs + 1])
        for n in self.nodes:
            n.default()

        inputs = [n for n in self.nodes if (n.type == 'input') or (n.type == 'bias')]
        outputs = [n for n in self.nodes if n.type == 'output']
        for _ in range(len(outputs)):
            n1 = random.choice(inputs)
            assert n1.type in ['input', 'bias']
            n2 = outputs.pop(0)
            assert n2.type == 'output'
            c = self.registry.get_connection(n1, n2)
            c.rand_weight()
            self.connections.append(c)

    def inherit_connections(self, connections):
        self.connections = deepcopy(connections)
        for c in self.connections:
            if c.n1 not in self.nodes:
                n1 = self.registry.get_node(c.n1.id)
                self.nodes.append(n1)
            else:
                n1 = next(x for x in self.nodes if x == c.n1)

            if c.n2 not in self.nodes:
                n2 = self.registry.get_node(c.n2.id)
                self.nodes.append(n2)
            else:
                n2 = next(x for x in self.nodes if x == c.n2)
            c.set_nodes(n1, n2)
        self.sort_nodes_connections()

    def set_fitness(self, f: float):
        self.fitness = f

    def get_gene(self):
        ret = dict()
        for c in self.connections:
            ret[c.id] = c
        return ret

    """ *** Mutations *** """
    def apply_mutations(self):
        self.mutate_add_node()
        self.mutate_link()
        self.mutate_weight_shift()
        self.mutate_enable_disable_connection()

    # add a new connection randomly, recurrent connections are possible
    def mutate_link(self):
        self.ready = False
        if random.random() < 1-P_NEW_LINK:
            return

        n1 = random.choice(self.nodes)
        n2 = random.choice(self.nodes)

        # if this connection exists --> enable it
        if Connection(n1, n2) in self.connections:
            c = next(c for c in self.connections if c == Connection(n1, n2))
            c.enable()
        # else if this connection is possible --> create it
        elif (n1.type != 'output') and ((n2.type != 'bias') or (n2.type != 'input')):
            c = self.registry.get_connection(n1, n2)
            c.rand_weight()
            self.connections.append(c)
            self.sort_nodes_connections()

    # add node in connection, old connection disabled
    def mutate_add_node(self):
        self.ready = False
        if random.random() < 1-P_NEW_NODE:
            return

        # select random connection, disable it
        c = random.choice(self.connections)
        c.disable()

        # get the splitting node
        n_add = self.registry.split_connection(c)

        # check if max hidden size is reached
        if not n_add:
            return

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
        # for c in self.connections:
        #     if c.enabled is False:
        #         c.enable()
        #         break
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
            if random.random() < .9:
                c.weight += random.uniform(-.5, .5)
            else:
                c.weight = random.uniform(-2, 2)
        self.sort_nodes_connections()

    """ *** prepare for forward *** """
    def sort_nodes_connections(self):
        self.connections = sorted(self.connections, key=lambda x: x.id)
        self.nodes = sorted(self.nodes, key=lambda x: x.id)

    def ordered_graph(self):
        # get ordered nodes and connections
        self.sort_nodes_connections()
        # self.set_next_nodes()
        self.net_graph = Graph(self.nodes, self.connections)
        self.computation_order, self.normal_ops = self.net_graph.get_computation_order()

        self.ready = True

    """ OLD, don't use
    def set_next_nodes(self):
        # find nodes from connection in self.nodes
        for c in self.connections:
            n1 = next(x for x in self.nodes if x == c.n1)
            n2 = next(x for x in self.nodes if x == c.n2)

            if n2 not in list(map(lambda x: x[0], n1.next_nodes)):
                n1.next_nodes.append((n2, c.enabled, c.weight))
    
    def forward_old(self, x: np.array, verbose: bool = False):
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
    """


def crossover_genes(p1: Genom, p2: Genom):
    assert (p1.fitness is not None) and (p2.fitness is not None), 'Fitness has to be evaluated before crossover'
    p_high, p_low = (p1, p2) if p1.fitness > p2.fitness else (p2, p1)

    # get boths list of connections, dict with connection id as key -> connection
    c_low = deepcopy(p_low.get_gene())
    c_high = deepcopy(p_high.get_gene())

    # align genes
    id_low = sorted(list(c_low.keys()))
    id_high = sorted(list(c_high.keys()))
    max_id = max(id_low + id_high)

    # TODO: does this work?
    new = []
    for cid in range(max_id + 1):
        # connection in both genes
        if cid in id_low and cid in id_high:
            new.append(c_low[cid]) if random.random() < 0.5 else new.append(c_high[cid])
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
    match, disjoint, excess = [], [], []

    k1, k2 = [], []
    for i in range(max_id+1):
        k1.append(i if i in id1 else -1)
        k2.append(i if i in id2 else -1)

    for idx, (iv1, iv2) in enumerate(zip(k1, k2)):
        if idx <= max_id:
            if iv1 != -1 and iv2 != -1:
                match.append(idx)
                continue
            elif iv1 != -1 or iv2 != -1:
                disjoint.append(idx)
                continue
        else:
            if iv1 != -1 or iv2 != -1:
                excess.append(idx)

    w_diff = [abs(cons1[cid].weight - cons2[cid].weight) for cid in match]
    w_diff = save_devide(w_diff)

    n = max(len(id1), len(id2))
    return C1 * len(excess) / n + C2 * len(disjoint) / n + C3 * w_diff
