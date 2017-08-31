import numpy as np
from helper import *

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

	def backward(self):
		self.gradients = {self: 0}
		for n in self.outbound_nodes:
			grad_cost = n.gradients[self]
			self.gradients[self] += n.gradients[self]			


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

	def backward(self):
		self.gradients = {n: 0 for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += 1 * grad  ## dX
			self.gradients[self.inbound_nodes[1]] += 1 * grad  ## dY


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

	def backward(self):
		self.cache[0] = self.inbound_nodes[0].value  ## X
		self.cache[1] = self.inbound_nodes[1].value  ## Y

		self.gradients = {n: 0 for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] = self.cache[1] * grad  ## dX					
			self.gradients[self.inbound_nodes[1]] = self.cache[0] * grad  ## dY

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

		self.cache[0] = self.inbound_nodes[0].value  ## X (inputs)
		self.cache[1] = self.inbound_nodes[1].value	 ## W (Weights)
		self.cache[2] = self.inbound_nodes[2].value  ## b (bias)
		self.value += np.dot(self.cache[0], self.cache[1]) + self.cache[2]

	def backward(self):
		self.gradients = {n: np.zeros_likes(n.value) for n in self.inbound_nodes}
		for n in self.outbound_nodes:
			grad = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += np.dot(grad, self.cache[1].T)	## dZ
			self.gradients[self.inbound_nodes[1]] += np.dot(self.cache[0].T, grad)  ## dw
			self.gradients[self.inbound_nodes[2]] += np.sum(grad, axis=0, keepdims=False)  ##db


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

	def backward(self):
		self.gradients = {n: np.zeros_likes(n.value) for n in self.inbound_nodes}	
		for n in self.outbound_nodes:
			grad = n.gradients[self]
			self.gradients[self.inbound_nodes[0]] += (1 - self.value) * self.value * grad


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
	
	def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff		

class cost_categorical_cross_entropy(Node):
	"""
	COST CROSS ENTROPY subclass node calulate softmax and loss (Categorical cross entropy)
	"""
	def __init__(self, y, y_hat):
		Node.__init__(self, [y, y_hat])

	def _predict(self):
        probs = self._softmax(self.inbound_nodes[0].value)
        return np.argmax(probs, axis=1)	

	def _accuracy(self):
        preds = self._predict()
        return np.mean(preds == self.inbound_nodes[1].value)	

	def _softmax(self, x):
		z = np.exp(x)/ np.sum(np.exp(x), axis=0, keepdims=True)
		return z	

	def forward(self):
		y = self.inbound_nodes[0].value
		y_hat = self.inbound_nodes[1].value
		self.cache[0] = self._softmax(self.inbound_nodes[0].value)
		self.cache[1] = y_hat
		m = y.shape[0]
		logprobs = np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat))
		self.value = - np.sum(logprobs) / m
		self.value = np.squeeze(self.value)

	def backward(self):
		assert len(self.outbound_nodes) == 0
		self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
		probs = np.copy(self.cache[0])
		y = self.cache[1]
		n = probs.shape[0]
		probs[range(n), y] -= 1
		probs /=n
		self.gradients[self.inbound_nodes[0]] = gprobs



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