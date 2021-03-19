from copy import deepcopy
import numpy as np
import random


""" *** Node *** """


class Node:
    def __init__(self, id_: int, type_: str = 'hidden'):
        assert type_ in ['input', 'bias', 'hidden', 'output']
        # innovation number
        self.id = id_

        # type ('input' or 'hidden' or 'output' or 'bias'
        self.type = type_

        # for calculation
        self.in_val = 0  # sum of inputs
        self.out_val = 0  # after activation
        self.active = True

        self.next_nodes = []  # list of all outgoing connections (node, enabled, weight)

    def evaluate(self):
        self.activation()
        for (n, enabled, weight) in self.next_nodes:
            if enabled:
                n.in_val += weight * self.out_val

    # return this node
    def get(self):
        return deepcopy(self).default()

    def activation(self):
        if self.type in ['input', 'bias']:
            self.out_val = self.in_val
        else:
            # sigmoid activation
            self.out_val = 1 / (1 + np.exp(-self.in_val))

    def reset_vals(self):
        self.in_val = self.out_val = 0

    def default(self):
        self.in_val = self.out_val = 0
        self.next_nodes = []

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return str(self.id) + ', ' + str(self.type)

    def __repr__(self):
        return self.__str__()


""" *** Connection *** """


class Connection:
    def __init__(self, node1: Node, node2: Node, weight: float = 0., id_: int = None):
        self.id = id_
        self.n1 = node1
        self.n2 = node2
        self.weight = weight  # between -2 and 2
        self.enabled = True

    def clear(self):
        ret = deepcopy(self)
        ret.weight = 0
        ret.enabled = False
        return ret

    def rand_weight(self):
        self.weight = random.uniform(-2., 2.)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def en_dis_able(self):
        self.enabled = not self.enabled

    def __eq__(self, other):
        return (self.n1 == other.n1) and (self.n2 == other.n2)

    def __str__(self):
        return str(self.n1.id) + ' --> ' + str(self.n2.id)

    def __repr__(self):
        return self.__str__()