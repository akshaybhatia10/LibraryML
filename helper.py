def topological_sort(input_nodes):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

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
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def accuracy(node, feed_dict):
    """
    Computes the accuracy of the model. All the weights and data(features, labels) should be in `feed_dict`.
    Arguments:
        `node`: A node in the graph, should be the output node (have no outgoing edges.
        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    """
    input_nodes = [n for n in feed_dict.keys()]
    nodes = topological_sort(input_nodes)
    # doesn't make sense is output node isn't Softmax
    assert node.typname == 'CrossEntropyWithSoftmax'
    assert nodes[-1].typname == 'CrossEntropyWithSoftmax'
    

    # forward pass on all nodes except the last
    for n in nodes[:-1]:
        if n.typname == 'Input':
            v = feed_dict[n]
            n.forward(v)
        else:
            n.forward()

    return nodes[-1]._accuracy()    