from LibraryML import *

"""
For Testing Purposes only
Builds and runs a differential graph with LibraryML.
"""
x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
s = Mul(x, y, z)

feed_dict = {x: 10, y: 5, z: 1}

sorted_nodes = topological_sort(feed_dict)

add = forward_pass(f, sorted_nodes)
mul = forward_pass(s, sorted_nodes)


print("{} + {} + {} = {} ".format(feed_dict[x], feed_dict[y], feed_dict[z], add))

print("{} * {} * {} = {} ".format(feed_dict[x], feed_dict[y], feed_dict[z], mul))