from utils.toolkit import topological_sort


inputs = None
outputs = None
graph = None


def build_graph():
    global graph
    graph = topological_sort(inputs, outputs, mode='forward')
    return graph


def reset_graph():
    global graph
    global inputs
    global outputs
    inputs = None
    outputs = None
    for g in graph:
        g.out_bounds.clear()
