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

		#values = [n.value for n in self.inbound_nodes]
		# self.value = sum(values)

		for x in range(len(self.inbound_nodes)):
			self.value += self.inbound_nodes[x].value

class Mul(Node):
	"""
	MUL subclass of node performs a computation (Multiplication)
	"""
	def __init__(self, *inputs):
		Node.__init__(self, inputs)

	def forward(self, value = None):
		self.value = 1

		#for n in self.inbound_nodes:
		#	self.value *= n.value

		for x in range(len(self.inbound_nodes)):
			self.value *= self.inbound_nodes[x].value				


class Linear(Node):
	"""
	LINEAR subclass of node performs a computation (Linear)
	"""
	def __init__(self, inputs, weights, bias):
		Node.__init__(self, [inputs, weights, bias])

	def forward(self, value = None):
		self.value = 0
		# for x in range(len(self.inbound_nodes)):
		# 	self.value += self.inbound_nodes[0].value[x] * self.inbound_nodes[1].value[x] 
		# self.value += self.inbound_nodes[2].value
		X = self.inbound_nodes[0].value
		W = self.inbound_nodes[1].value	
		b = self.inbound_nodes[2].value
		self.value += np.dot(X,W) + b


class Sigmoid(Node):
	"""
	SIGMOID subclass of node performs a computation (Sigmoid Activation Function)
	"""
	def __init__(self, x):
		Node.__init__(self, [x])

	def _sigmoid(self, x):
		z = 1.0 / (1.0 + np.exp(-x))
		return z 	

	def forward(self):
		self.value = self._sigmoid(self.inbound_nodes[0].value)	


class cost_mse(Node):
	"""
	COST MSE subclass of node calculates the loss of the model (Mean Squared Error)
	"""
	def __init__(self, y, y_hat):
		Node.__init__(self, [y, y_hat])

	def forward(self):
		y = self.inbound_nodes[0].value.reshape(-1, 1)
		y_hat = self.inbound_nodes[1].value.reshape(-1, 1)
		error = y - y_hat
		self.value = np.mean(error**2)
			

class cost_categorical_cross_entropy(Node):
	"""
	COST CROSS ENTROPY subclass node calulate softmax and loss (Categorical cross entropy)
	"""
	def __init__(self, y, y_hat):
		Node.__init__(self, [y, y_hat])

	def _softmax(self, x):
		z = np.exp(x)/ np.sum(np.exp(x), axis=0, keepdims=True)
		return z

	def forward(self):
		y = self.inbound_nodes[0].value
		y_hat = self.inbound_nodes[1].value
		m = y.shape[0]
		print (y.shape)
		logprobs = np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat))
		self.value = - np.sum(logprobs) / m
		self.value = np.squeeze(self.value)


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