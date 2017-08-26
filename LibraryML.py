import numpy as np

class Node(object):
	"""
	Base Class NODE
	"""
	def __init__(self, inbound_nodes):
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
    def forward(self):
        """
        Forward propagation.
        
        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.
        
        Compute the gradient of the current node with respect
        to the input nodes. The gradient of the loss with respect
        to the current node should already be computed in the `gradients`
        attribute of the output nodes.
        """
        raise NotImplemented		


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
		if value:
			self.value = value	


class Add(Node):
	"""
	ADD subclass of node performs a computation (Addition)
	"""
	def __init__(self, x, y):
		Node.__init__(self, [x, y])

	def forward(self, value = None):
		self.value = self.inbound_nodes[0].value + self.inbound_nodes[1].value
