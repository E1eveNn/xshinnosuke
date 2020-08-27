from ..nn.core import Layer, Variable
from xshinnosuke.nn.initializers import get_initializer
from ..nn.functional import embedding
from ..nn.grad_fn import EmbeddingBackward
from ..nn import global_graph as GlobalGraph


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform', mask_zero=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = get_initializer(embeddings_initializer)
        self.mask_zero = mask_zero

    def initial_params(self, input_shape=None):
        w = self.initializer((self.input_dim, self.output_dim), name='xs_variable')
        self.variables.append(w)

    def compute_output_shape(self, input_shape=None):
        return input_shape + (self.output_dim,)

    def __call__(self, inbound, *args, **kwargs):
        assert self.output_dim is not None and self.input_dim is not None
        if isinstance(inbound, Variable):
            if len(self.variables) == 0:
                self.initial_params()
            output = embedding(inbound, self.variables[0], GlobalGraph.IS_TRAINING)
            return output
        super(Embedding, self).__call__(inbound)
        return self

    def forward(self, x: Variable = None, *args):
        if x is not None:
            self.input_data = x
        assert self.input_data.data.ndim == 2

        self.data = embedding(self.input_data, self.variables[0], GlobalGraph.IS_TRAINING)
        self.feed_variable_to_next_layers(self.data)
        return self.data

    def backward(self, gradients=None):
        EmbeddingBackward(self.data)
