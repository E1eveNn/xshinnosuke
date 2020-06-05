
np = __import__('cupy')

inputs = None
outputs = None
graph = None


def topological_sort(inputs, outputs):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    Returns a list of sorted nodes.
    """
    name_dict = dict()
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
