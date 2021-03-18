from copy import deepcopy
import numpy as np


class Node:
    def __init__(self, id_: int, in_: bool = False, out_: bool = False):
        self.id = id_
        self.input = in_
        self.output = out_

        self.bias = None
        self.activation = None

    def __eq__(self, other):
        return self.id == other.id

    def create(self):
        return deepcopy(self)

    def compute_output(self):
        # TODO
        pass


class Connection:
    def __init__(self, node1: Node, node2: Node, id_: int = None):
        self.id = id_
        self.n1 = node1
        self.n2 = node2
        self.weight = 0  # between -2 and 2
        self.enabled = False

    def __eq__(self, other):
        return (self.n1 == other.n1) and (self.n2 == other.n2)

    def clear(self):
        ret = deepcopy(self)
        ret.weight = 0
        ret.enabled = False
        return ret


class Registry:
    def __init__(self, input_size: int, output_size: int):
        # Number of input and output nodes
        self.inputs = input_size
        self.outputs = output_size

        # Global nodes
        self.nb_nodes = 0
        self.nodes = []

        # Global connections
        self.nb_connections = 0
        self.connections = []

        # initialize input and output nodes
        for _ in range(input_size):
            self.nodes.append(Node(id_=self.nb_nodes, in_=True))
            self.nb_nodes += 1
        for _ in range(output_size):
            self.nodes.append(Node(id_=self.nb_nodes, out_=True))
            self.nb_nodes += 1

    # get a connection between two nodes, create new id if doesn't exist
    def get_connection(self, n1: Node, n2: Node):
        if n1 in self.nodes and n2 in self.nodes:
            if Connection(n1, n2) in self.connections:
                return self._check_connection(n1, n2)
            else:
                self.connections.append(Connection(n1, n2, id_=self.nb_connections))
                self.nb_connections += 1
                return self._check_connection(n1, n2)

    # create new node
    def create_node(self):
        self.nodes.append(Node(id_=self.nb_nodes))
        self.nb_nodes += 1

    def _check_connection(self, n1: Node, n2: Node):
        check = Connection(n1, n2)
        for c in self.connections:
            if c == check:
                return c

    def get_nb_nodes(self):
        return self.nb_nodes

    def get_nb_connections(self):
        return self.nb_connections


class Gen:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def compute(self):
        # TODO
        pass
