import core.__global as GLOBAL
import nn


def backward(outputs, inputs, retain_graph=False):
    graph = topological_sort(inputs, outputs)
    for node in reversed(graph):
        if node.grad_fn is not None:
            node.grad_fn(node)
        for l in node.next_layers:
            l.data = None
        if retain_graph or node.is_leaf or node.retain_grad():
            node.reset_(requires_grad=node.requires_grad, name=node.name, slices=node.slices(),
                        static_graph_tensor=node.is_static, retain_grad=node.retain_grad(), grad=node.grad)
        else:
            node.free_memory()
            node.reset_(requires_grad=node.requires_grad, name=node.name, slices=node.slices(),
                        static_graph_tensor=node.is_static)

    if retain_graph:
        GLOBAL.GRAPH = graph
    else:
        del GLOBAL.INPUTS, GLOBAL.OUTPUTS, GLOBAL.GRAPH, graph[:]
        GLOBAL.INPUTS = None
        GLOBAL.OUTPUTS = None
        GLOBAL.GRAPH = None


def topological_sort(ins, outs):
    topo = []
    visited = set()

    def build_topo(v):
        if ins is not None and v == ins:
            topo.append(v)
            return
        if v not in visited:
            visited.add(v)
            for child in v.in_bounds:
                if isinstance(child, nn.Parameter) or child is None:
                    continue
                build_topo(child)
            topo.append(v)
    build_topo(outs)
    return topo


def reset_graph(graph):
    del graph[:]
    del GLOBAL.INPUTS, GLOBAL.OUTPUTS, GLOBAL.GRAPH
    GLOBAL.INPUTS = None
    GLOBAL.OUTPUTS = None
    GLOBAL.GRAPH = None

