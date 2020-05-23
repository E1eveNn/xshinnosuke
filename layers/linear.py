from nn.core import Layer, Variable
from utils.initializers import get_initializer
from utils.activators import get_activator
from nn.functional import dense, flatten
from nn.grad_fn import DenseBackward, FlattenBackward
import numpy as np
import nn.global_graph as GlobalGraph


class Flatten(Layer):
    def __init__(self, start=1, **kwargs):
        # output dimensions
        if start < 1:
            raise ValueError('start must be > 0')
        self.start = start
        super(Flatten, self).__init__(**kwargs)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            if GlobalGraph.inputs is None:
                GlobalGraph.inputs = inbound

            output = flatten(inbound, self.start)
            # output是一个Variable
            return output
        super().__call__(inbound)
        return self

    def compute_output_shape(self, input_shape=None):
        assert len(input_shape) >= 3
        flatten_shape = np.prod(input_shape[self.start - 1:])
        output_shape = input_shape[: self.start - 1] + (flatten_shape,)
        return output_shape

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = flatten(self.input_data, self.start)
        self.connect_init(self.data, is_training)
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
        self.timedist_grad = None
        super(Dense, self).__init__(**kwargs)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            if GlobalGraph.inputs is None:
                GlobalGraph.inputs = inbound

            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = dense(inbound, self.variables[0], self.variables[1])
            if self.activation is not None:
                output = self.activation.forward(output)
            # output是一个Variable
            return output

        super().__call__(inbound)
        return self

    def initial_params(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        w = Variable(self.kernel_initializer(self.input_shape + (self.out_features, )), name='variable')
        if self.use_bias:
            b = Variable(self.bias_initializer((1, self.out_features)), name='variable')
        else:
            b = None
        self.variables.append(w)
        self.variables.append(b)

    def compute_output_shape(self, input_shape=None):
        return tuple([self.out_features, ])

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        w, b = self.variables
        self.data = dense(self.input_data, w, b)
        if self.activation is not None:
            output = self.activation.forward(self.data)
            self.connect_init(output, is_training)
            return output
        else:
            self.connect_init(self.data, is_training)
            return self.data

    def backward(self, gradients=None):
        if self.activation is not None:
            self.activation.backward()
        DenseBackward(self.data)
