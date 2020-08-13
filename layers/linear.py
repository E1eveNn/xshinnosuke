from ..nn.core import Layer, Variable
from functools import reduce
from ..nn.initializers import get_initializer
from .activators import get_activator
from ..nn.functional import dense, flatten
from ..nn.grad_fn import DenseBackward, FlattenBackward
from ..nn import global_graph as GlobalGraph


class Flatten(Layer):
    def __init__(self, start=1, **kwargs):
        # output dimensions
        if start < 1:
            raise ValueError('start must be > 0')
        self.start = start
        super(Flatten, self).__init__(**kwargs)

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            if GlobalGraph.INPUTS is None:
                GlobalGraph.INPUTS = inbound

            output = flatten(inbound, self.start, training=GlobalGraph.IS_TRAINING)
            # output是一个Variable
            return output
        super().__call__(inbound)
        return self

    def compute_output_shape(self, input_shape=None):
        assert len(input_shape) >= 3
        flatten_shape = reduce(lambda x, y: x * y, input_shape[self.start - 1:])
        output_shape = input_shape[: self.start - 1] + (flatten_shape,)
        return output_shape

    def forward(self, x: Variable = None, *args):
        if x is not None:
            self.input_data = x
        self.data = flatten(self.input_data, self.start, training=GlobalGraph.IS_TRAINING)
        self.feed_variable_to_next_layers(self.data)
        return self.data

    def backward(self, gradients=None):
        FlattenBackward(self.data)


class Dense(Layer):
    def __init__(self, out_features, activation=None, use_bias=True, kernel_initializer='normal',
                 bias_initializer='zeros', kernel_regularizer=None, **kwargs):
        self.out_features = out_features
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.activation = get_activator(activation) if activation is not None else None
        self.kernel_regularizer = kernel_regularizer
        super(Dense, self).__init__(**kwargs)

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            if GlobalGraph.INPUTS is None:
                GlobalGraph.INPUTS = inbound

            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = dense(inbound, self.variables[0], self.variables[1], GlobalGraph.IS_TRAINING)
            if self.activation is not None:
                output = self.activation.forward(output)
            # output是一个Variable
            return output

        super().__call__(inbound)
        return self

    def initial_params(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        w = self.kernel_initializer(self.input_shape + (self.out_features, ), name='xs_variable')
        if self.use_bias:
            b = self.bias_initializer((1, self.out_features), name='xs_variable')
        else:
            b = None
        self.variables.append(w)
        self.variables.append(b)

    def compute_output_shape(self, input_shape=None):
        return tuple([self.out_features, ])

    def forward(self, x: Variable = None, *args):
        if x is not None:
            self.input_data = x
        w, b = self.variables
        self.data = dense(self.input_data, w, b, GlobalGraph.IS_TRAINING)
        if self.activation is not None:
            output = self.activation.forward(self.data)
            self.feed_variable_to_next_layers(output)
            return output
        else:
            self.feed_variable_to_next_layers(self.data)
            return self.data

    def backward(self, gradients=None):
        if self.activation is not None:
            self.activation.backward()
        DenseBackward(self.data)
