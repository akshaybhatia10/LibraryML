import numpy as np

class Node(object):
	"""
	Base Node Class
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


class Input(Node):
	"""
	 Input subclass just holds a value, such as a data 
	 feature or a model parameter (weight/bias).
	"""
	def __init__(self):
		## Input nodes or 'inputs' to the neural network will
		## not have any inbound nodes
		Node.__init__(self)

	def forward(self, value = None):
		if value:
			self.value = value	