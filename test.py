from LibraryML import *

"""
For Testing Purposes only
Builds and runs a differential graph with LibraryML.
"""
x, y, z = Input(), Input(), Input()
#inputs, weights, bias = Input(), Input(), Input()
X, W, b = Input(), Input(), Input()
y, y_hat = Input(), Input()

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

#y_ = np.array(([0.01, 0.9, 0.8],[0.7, 0.2, 0.1]))
#y_hat_ = np.array(([0.01, 1., 1.], [1, 1, 0.02]))

y_ = np.array(([1.], [1.]))
y_hat_ = np.array(([0.9], [0.1]))

f = Add(x, y, z)
s = Mul(x, y, z)
#l = Linear(inputs, weights, bias)
l = Linear(X, W, b)
g = Sigmoid(l)
c = cost_mse(y, y_hat)
a = cost_categorical_cross_entropy(y, y_hat)


feed_dict1 = {x: 10, y: 5, z: 1}

# feed_dict2 = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }

feed_dict2 = {X: X_, W: W_, b: b_}

feed_dict3 = {y: y_, y_hat: y_hat_}



sorted_nodes1 = topological_sort(feed_dict1)
sorted_nodes2 = topological_sort(feed_dict2)
sorted_nodes3 = topological_sort(feed_dict3)

add = forward_pass(f, sorted_nodes1)
mul = forward_pass(s, sorted_nodes1)
lin = forward_pass(l, sorted_nodes2)
sig = forward_pass(g, sorted_nodes2)
cost = forward_pass(c, sorted_nodes3)
cross = forward_pass(a, sorted_nodes3)


print("{} + {} + {} = {} ".format(feed_dict1[x], feed_dict1[y], feed_dict1[z], add))

print("{} * {} * {} = {} ".format(feed_dict1[x], feed_dict1[y], feed_dict1[z], mul))

print("Linear Output: {}".format(lin))

print("Sigmoid of Linear Operation: {}".format(sig))

print("Cost: {}".format(cost))

print("Cost: {}".format(cross))

# y = self.inbound_nodes[0].value.reshape(-1, 1)
		# y_hat = self.inbound_nodes[1].value.reshape(-1, 1)
		# m = self.inbound_nodes[0].value.shape[0]
		# probs = (y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))
		# self.value = - np.sum(probs) / m

