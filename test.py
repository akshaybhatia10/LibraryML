from LibraryML import *

"""
For Testing Purposes only
Builds and runs a differential graph with LibraryML.
"""
x, y, z = Input(), Input(), Input()
#inputs, weights, bias = Input(), Input(), Input()
X, W, b = Input(), Input(), Input()

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

f = Add(x, y, z)
s = Mul(x, y, z)
#l = Linear(inputs, weights, bias)
l = Linear(X, W, b)


feed_dict1 = {x: 10, y: 5, z: 1}

# feed_dict2 = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }

feed_dict2 = {X: X_, W: W_, b: b_}



sorted_nodes1 = topological_sort(feed_dict1)
sorted_nodes2 = topological_sort(feed_dict2)


add = forward_pass(f, sorted_nodes1)
mul = forward_pass(s, sorted_nodes1)
lin = forward_pass(l, sorted_nodes2)


print("{} + {} + {} = {} ".format(feed_dict1[x], feed_dict1[y], feed_dict1[z], add))

print("{} * {} * {} = {} ".format(feed_dict1[x], feed_dict1[y], feed_dict1[z], mul))

print("Linear Output: {}".format(lin))