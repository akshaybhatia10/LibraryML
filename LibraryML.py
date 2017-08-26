import numpy as np

class Node(object):
	"""
	Base Class NODE
	"""
	def __init__(self, inbound_nodes=[]):
		## Inbound nodes to 'this' node 
		self.inbound_nodes = inbound_nodes
		## Outbound nodes from 'this' node
		self.outbound_nodes = []
		## store the calculated values
		self.value = None
		## store the values calculated in forward prop
		## to be used in backward prop
		self.cache = {}
		self.gradients = {}

		for node in self.inbound_nodes:
			node.outbound_nodes.append(self)

	# These will be implemented in a subclass.

    # def forward(self):
    #     """
    #     Forward propagation.
    #     """
    #     raise NotImplemented

    # def backward(self):
    #     """
    #     Backward propagation.
        
    #     Compute the gradient of the current node with respect
    #     to the input nodes.
    #     """
    #     raise NotImplemented		


class Input(Node):
	"""
	 INPUT subclass of node just holds a value, such as a 
	 data feature or a model parameter (weight/bias).
	"""
	def __init__(self):
		## Input nodes or 'inputs' to the neural network will
		## not have any inbound nodes
		Node.__init__(self)

	def forward(self, value = None):
		if value is not None:
			self.value = value	


class Add(Node):
	"""
	ADD subclass of node performs a computation (Addition)
	"""
	def __init__(self, *inputs):
		Node.__init__(self, inputs)

	def forward(self, value = None):
		self.value = 0
		#x = self.inbound_nodes[0].value
		#y = self.inbound_nodes[1].value
		#self.value = x + y
		for x in range(len(self.inbound_nodes)):
			self.value += self.inbound_nodes[x].value


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value