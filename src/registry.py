from src.base_classes import Node, Connection
from copy import deepcopy


class Registry:
    def __init__(self, input_size: int, output_size: int, max_hidden: int = None):
        # Number of input and output nodes
        self.inputs = input_size
        self.outputs = output_size
        self.hidden_max = max_hidden

        # Global nodes
        self.nb_nodes = 0
        self.nodes = []

        # Global connections
        self.nb_connections = 0
        self.connections = []

        # initialize input and output nodes
        for _ in range(input_size):
            self.nodes.append(Node(id_=self.nb_nodes, type_='input'))
            self.nb_nodes += 1

        # initialize bias node
        self.nodes.append(Node(id_=self.nb_nodes, type_='bias'))
        self.nodes[-1].out_val = 1
        self.nb_nodes += 1

        for _ in range(output_size):
            self.nodes.append(Node(id_=self.nb_nodes, type_='output'))
            self.nb_nodes += 1

    # get a connection between two nodes, create new id if doesn't exist
    def get_connection(self, n1: Node, n2: Node):
        assert (n1 in self.nodes) and (n2 in self.nodes)
        if Connection(n1, n2) in self.connections:
            c = self._check_connection(n1, n2)
            return c
        else:
            new_con = Connection(n1, n2, id_=self.nb_connections)
            self.connections.append(new_con)
            self.nb_connections += 1
            c = self._check_connection(n1, n2)
            return c

    # create new node
    def create_node(self):
        if self.hidden_max is not None:
            if len(self.nodes) - self.inputs - self.outputs >= self.hidden_max:
                return False
        new_node = Node(id_=self.nb_nodes)
        self.nodes.append(new_node)
        self.nb_nodes += 1
        return deepcopy(new_node)

    # split a connection and add a node
    def split_connection(self, c: Connection):
        # find this connection
        local_con = next(x for x in self.connections if x == c)

        # check if this connection has already been split
        if local_con.split is not None:
            # which node id resulted from splitting? --> return this
            n_id = local_con.split
            return self.get_node(n_id)
        else:
            # create a new node, add split id to connection
            new_node = self.create_node()
            if not new_node:
                return False
            local_con.split = new_node.id
            return new_node

    def _check_connection(self, n1: Node, n2: Node):
        # check if connection already exists
        check = Connection(n1, n2)
        for c in self.connections:
            if c == check:
                ret = deepcopy(c)
                # set nodes, if not adress would be wrong
                ret.set_nodes(n1, n2)
                return ret

        # else create connection
        new_con = Connection(n1, n2, id_=self.nb_connections)
        self.connections.append(new_con)
        self.nb_connections += 1
        ret = deepcopy(new_con)
        ret.set_nodes(n1, n2)
        return ret

    def get_node(self, id_: int):
        return deepcopy(self.nodes[id_])

    def get_nb_nodes(self):
        return self.nb_nodes

    def get_nb_connections(self):
        return self.nb_connections
