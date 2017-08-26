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

	