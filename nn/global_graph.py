import warnings


try:
    np = __import__('cupy')
except ModuleNotFoundError:
    warnings.warn('Looks like you\'re using Numpy, try to use Cupy to speed up instead!')
    np = __import__('numpy')

inputs = None
outputs = None
graph = None


def topological_sort(inputs, outputs):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    Returns a list of sorted nodes.
    """
    name_dict = dict()
    simple_name_dict = {
        'BatchNormalization': 'bn',
        'LayerNormalization': 'ln',
        'GroupNormalization': 'gn'
    }

    G = {}
    graph = []
    outputs = list([outputs])

    layers = list([inputs])
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.out_bounds:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    S = set(list([inputs]))
    while len(S) > 0:
        n = S.pop()
        graph.append(n)
        if n.name is None:
            if n.__class__.__name__ in name_dict:
                name_dict[n.__class__.__name__] += 1
            else:
                name_dict[n.__class__.__name__] = 0
            if n.__class__.__name__ in simple_name_dict.keys():
                n.name = simple_name_dict[n.__class__.__name__] + str(name_dict[n.__class__.__name__])
            else:
                n.name = n.__class__.__name__.lower() + str(name_dict[n.__class__.__name__])
        if n in outputs:
            continue
        for m in n.out_bounds:

            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)

    return graph


def build_graph():
    global graph
    graph = topological_sort(inputs, outputs)
    return graph


def reset_graph():
    global graph
    global inputs
    global outputs
    del inputs, outputs, graph
    inputs = None
    outputs = None
    graph = None


def reset_node(node):
    node.__init__(node.data)


def delete_node(node):
    del node.cache, node.data, node.grad, node.grad_fn, node.in_bounds, node.out_bounds, node.shape, \
        node.name, node.requires_grad, node.retain
    del node
