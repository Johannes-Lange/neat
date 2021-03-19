from copy import deepcopy
import numpy as np


""" *** Node *** """


class Node:
    def __init__(self, id_: int, in_: bool = False, out_: bool = False):
        # innovation number
        self.id = id_

        # bools
        self.input = in_
        self.output = out_

        # for calculation
        self.in_val = 0  # sum of inputs
        self.out_val = 0  # after activation
        self.active = True

        self.next_nodes = []  # list of all outgoing connections

    def __eq__(self, other):
        return self.id == other.id

    # return this node
    def get(self):
        return deepcopy(self).default()

    def activation(self):
        # sigmoid activation
        self.out_val = 1 / (1 + np.exp(-self.in_val))

    def reset_vals(self):
        self.in_val = self.out_val = 0

    def default(self):
        self.in_val = self.out_val = 0
        self.next_nodes = []


""" *** Connection *** """


class Connection:
    def __init__(self, node1: Node, node2: Node, weight: float = 0., id_: int = None):
        self.id = id_
        self.n1 = node1
        self.n2 = node2
        self.weight = weight  # between -2 and 2
        self.enabled = False

    def __eq__(self, other):
        return (self.n1 == other.n1) and (self.n2 == other.n2)

    def clear(self):
        ret = deepcopy(self)
        ret.weight = 0
        ret.enabled = False
        return ret
