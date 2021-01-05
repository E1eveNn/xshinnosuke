from layers.activators import get_activator
from .base import *


class Dense(Layer):
    def __init__(self, out_features: int, activation: str = None, use_bias: bool = True, kernel_initializer: str = 'normal',
                 bias_initializer: str = 'zeros', **kwargs):
        self.out_features = out_features
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.activation = get_activator(activation) if activation is not None else None
        super().__init__(**kwargs)

    @property
    def data(self):
        return self._data if self.activation is None else self.activation.data

    @data.setter
    def data(self, v):
        if self.activation is None:
            self._data = v
        else:
            self.activation.data = v

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return (self.out_features, )

    def init_params(self, input_shape=None, *args):
        if input_shape is not None:
            self._input_shape = input_shape
        weight = F.Parameter(self.kernel_initializer(self._input_shape + (self.out_features, ), requires_grad=True))
        self._parameters.append(weight)
        if self.use_bias:
            bias = F.Parameter(self.bias_initializer((1, self.out_features), requires_grad=True))
            self._parameters.append(bias)

    def init_layer_out_tensor(self, x: F.Tensor = None):
        super(Dense, self).init_layer_out_tensor(x)
        if self.activation is not None:
            self.activation.compute_output_shape(self.out_features)
            self.activation.init_layer_out_tensor(self._data)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        if self.use_bias:
            weight, bias = self._parameters
            self._data = F.addmm(bias, x, weight, self._data)
        else:
            weight,  = self._parameters
            self._data = F.mm(x, weight, self._data)
        if self.activation is not None:
            return self.activation.forward(self._data)
        return self._data

    def backward(self, gradients: F.Tensor = None):
        if self.activation is not None:
            self.activation.backward()
        super().backward()


class Flatten(Layer):
    def __init__(self, start_dim=1, **kwargs):
        if start_dim < 1:
            raise ValueError('start_dim must be > 0')
        self.start_dim = start_dim
        super(Flatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape=None):
        assert len(input_shape) >= 3
        flatten_shape = reduce(lambda x, y: x * y, input_shape[self.start_dim - 1:])
        output_shape = input_shape[: self.start_dim - 1] + (flatten_shape,)
        return output_shape

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.flatten(x, self.start_dim,  self._data)
        return self._data
