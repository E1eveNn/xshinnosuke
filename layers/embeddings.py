from ..nn.core import Layer, Variable
from xshinnosuke.nn.initializers import get_initializer
from ..nn.functional import embedding
from ..nn.grad_fn import EmbeddingBackward


class Embedding(Layer):
    def __init__(self, output_dim, embeddings_initializer='uniform', mask_zero=False, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.initializer = get_initializer(embeddings_initializer)
        self.mask_zero = mask_zero

    def initial_params(self):
        w = Variable(self.initializer((self.input_shape[-1], self.output_dim)), name='embedding_w')
        self.variables.append(w)

    def compute_output_shape(self, input_shape=None):
        return input_shape + (self.output_dim,)

    def __call__(self, inbound):
        assert self.output_dim is not None
        super(Embedding, self).__call__(inbound)

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        assert self.input_data.data.ndim == 2

        self.data = embedding(self.input_data, self.variables[0])
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        EmbeddingBackward(self)
