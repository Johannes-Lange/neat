from src.base_classes import Node, Connection


class Graph:
    def __init__(self, nodes: [Node], connections: [Connection]):
        self.nodes = nodes
        self.cons = connections

        self.types = {'input': [], 'bias': [], 'hidden': [], 'output': []}
        self.graph = dict()
        for n in nodes:
            self.types[n.type].append(n.id)
            if n.id not in self.graph:
                self.graph[n.id] = {'in': [], 'out': [], 'visited': False, 'active': False}

        self.connections = dict()
        for c in connections:
            self.connections[c.id] = {'left': c.n1.id, 'right': c.n2.id, 'calculated': False, 'rec': None, 'id': c.id}
            self.graph[c.n1.id]['out'].append(c.id)
            self.graph[c.n2.id]['in'].append(c.id)

        # for graph algos
        self.order = []
        self.recurrent = []

        self.computation_order = []

    def get_computation_order(self):
        # all outputs from the input layer may be calculated first
        for n_id in (self.types['input'] + self.types['bias']):
            self.graph[n_id]['active'] = True
            self.order.append(('node', n_id))
            for out in self.graph[n_id]['out']:
                self.connections[out]['calculated'] = True
                self.order.append((self.c_to_str(out), out))

        # now start from output layer
        for n_id in self.types['output']:
            self.recursive_solve(n_id)
            self.reset_visited()

        for (op, id_) in self.order + self.recurrent:
            if op == 'node':
                ob = next(x for x in self.nodes if x.id == id_)
            elif 'connection' in op:
                ob = next(x for x in self.cons if x.id == id_)
            else:
                raise ValueError('error in graph computation')
            self.computation_order.append(ob)
        return self.computation_order

    def recursive_solve(self, node_id, last_node=None):
        # have we visited this node?
        if self.graph[node_id]['visited'] and (last_node is not None):
            # find connection to last visited node and set it recurrent and calculated
            for c_id in self.graph[node_id]['out']:
                if self.connections[c_id]['right'] == last_node:
                    self.connections[c_id]['rec'] = True
                    self.connections[c_id]['calculated'] = True
                    self.recurrent.append((self.c_to_str(c_id), c_id))
                    break
            return
        else:
            # set this node visited
            self.graph[node_id]['visited'] = True

        # can we calculate this neuron? Yes, if all incoming connections are calculated (= this node is active)
        if self.graph[node_id]['active']:
            # calculate all out connections
            for c_id in self.graph[node_id]['out']:
                if not self.connections[c_id]['calculated']:
                    self.connections[c_id]['calculated'] = True
                    self.order.append((self.c_to_str(c_id), c_id))
            return

        # if incoming connection is not calculated, recursively go to the node this connection comes from
        ins = [self.connections[c] for c in self.graph[node_id]['in']]
        if not all([c['calculated'] for c in ins]):
            for c in ins:
                if not c['calculated']:
                    # if we can calculate the node before, we can calculate this connection
                    # print(node_id, c['left'])
                    self.recursive_solve(c['left'], node_id)
        # if all incoming connections are already calculated --> calculate this node and all outs
        else:
            self.graph[node_id]['active'] = True
            self.order.append(('node', node_id))
            for c_id in self.graph[node_id]['out']:
                self.connections[c_id]['calculated'] = True
                self.order.append((self.c_to_str(c_id), c_id))
            return

        if all([c['calculated'] for c in ins]):
            # set this node to active
            self.graph[node_id]['active'] = True
            self.order.append(('node', node_id))
            # calculate all outs
            for c_id in self.graph[node_id]['out']:
                if not self.connections[c_id]['calculated']:
                    self.connections[c_id]['calculated'] = True
                    self.order.append((self.c_to_str(c_id), c_id))
        return

    def reset_visited(self):
        for n in self.graph.keys():
            self.graph[n]['visited'] = False

    def c_to_str(self, cid):
        return 'connection {} -> {}'.format(self.connections[cid]['left'], self.connections[cid]['right'])
