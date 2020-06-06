from ..nn.core import Layer, Variable
from xshinnosuke.nn.initializers import get_initializer
from ..nn.global_graph import np


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform', mask_zero=False, input_length=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = get_initializer(embeddings_initializer)
        self.mask_zero = mask_zero
        self.input_shape = (input_length, )

    def initial_params(self):
        w = Variable(self.initializer((self.input_dim, self.output_dim)), name='embedding_w')
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
        w, = self.variables
        # to one-hot
        self.data = w.data[self.input_data.data]
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        w, = self.variables
        if w.requires_grad:
            np.add.at(w.grad, self.input_data.data, self.data.grad)
