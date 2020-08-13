import warnings


try:
    np = __import__('cupy')
except ModuleNotFoundError:
    np = __import__('numpy')
    warnings.warn('Looks like you\'re using Numpy, try to install Cupy to gain GPU acceleration!')

INPUTS = None
OUTPUTS = None
GRAPH = None
IS_TRAINING = True


def topological_sort(ins, outs):
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
    sorted_graph = []
    outs = list([outs])
    layers = list([ins])
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

    S = set(list([ins]))
    while len(S) > 0:
        n = S.pop()
        sorted_graph.append(n)
        if n.name is None:
            if n.__class__.__name__ in name_dict:
                name_dict[n.__class__.__name__] += 1
            else:
                name_dict[n.__class__.__name__] = 0
            if n.__class__.__name__ in simple_name_dict.keys():
                n.name = simple_name_dict[n.__class__.__name__] + str(name_dict[n.__class__.__name__])
            else:
                n.name = n.__class__.__name__.lower() + str(name_dict[n.__class__.__name__])
        if n in outs:
            continue
        for m in n.out_bounds:

            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)

    return sorted_graph


def build_graph():
    global GRAPH
    global INPUTS
    global OUTPUTS
    GRAPH = topological_sort(INPUTS, OUTPUTS)
    return GRAPH


def reset_graph():
    global GRAPH
    global INPUTS
    global OUTPUTS
    del INPUTS, OUTPUTS, GRAPH
    INPUTS = None
    OUTPUTS = None
    GRAPH = None


def reset_node(node):
    retain = node.retain
    grad = node.grad
    node.__init__(data=node.data, name=node.name)
    node.grad = grad
    node.retain = retain


def delete_node(node):
    del node.cache, node.data, node.grad, node.grad_fn, node.in_bounds, node.out_bounds, node.shape, \
        node.name, node.requires_grad, node.retain
    del node
