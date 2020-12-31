from .base import *
from nn.initializers import Initializer
from layers.activators import get_activator
import nn.functional as F


class Conv2D(Layer):
    def __init__(self, out_channels: int, kernel_size: Union[int, Tuple], use_bias: bool = True,
                 stride: Union[int, Tuple] = 1, padding: Union[int, str] = 0, activation: str = None, kernel_initializer: Union[str, Initializer] = 'Normal',
                 bias_initializer: Union[str, Initializer] = 'zeros', **kwargs):
        self.out_channels = out_channels
        self.kernel_size = self.__check_size(kernel_size)
        self.use_bias = use_bias
        self.stride = self.__check_size(stride)
        self.padding = padding
        self.activation = get_activator(activation) if activation is not None else None
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.pad_size = padding
        super(Conv2D, self).__init__(**kwargs)

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        elif isinstance(kernel_size, (Tuple, List)):
            return kernel_size

    def init_params(self, input_shape: Tuple = None, *args):
        if input_shape is not None:
            self._input_shape = input_shape
        weight = F.Parameter(self.kernel_initializer((self.out_channels, self._input_shape[0], self.kernel_size[0], self.kernel_size[1]), requires_grad=True))
        self._parameters.append(weight)
        if self.use_bias:
            bias = F.Parameter(self.bias_initializer((1, self.out_channels), requires_grad=True))
            self._parameters.append(bias)
        else:
            self._parameters.append(None)

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        assert len(input_shape) == 3
        n_C_prev, n_H_prev, n_W_prev = input_shape
        filter_h, filter_w = self.kernel_size
        if self.padding.__class__.__name__ == 'str':
            padding = self.padding.upper()
            if padding == 'SAME':
                n_H = n_H_prev
                n_W = n_W_prev
                pad_h = (self.stride[0] * (n_H_prev - 1) - n_H_prev + filter_h) // 2
                self.pad_size = pad_h
            elif padding == 'VALID':
                n_H = (n_H_prev - filter_h) // self.stride[0] + 1
                n_W = (n_W_prev - filter_w) // self.stride[1] + 1
                self.pad_size = 0
            else:
                raise TypeError('Unknown padding type!plz inputs SAME or VALID or an integer')
        else:
            assert isinstance(self.padding, int)
            n_H = (n_H_prev - filter_h + 2 * self.padding) // self.stride[0] + 1
            n_W = (n_W_prev - filter_w + 2 * self.padding) // self.stride[1] + 1
            self.pad_size = self.padding
        return self.out_channels, n_H, n_W

    def init_layer_out_tensor(self, x: F.Tensor = None):
        super(Conv2D, self).init_layer_out_tensor(x)
        if self.activation is not None:
            self.activation.compute_output_shape(self._shape)
            self.activation.init_layer_out_tensor(self._data)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        if self.use_bias:
            weight, bias = self._parameters
        else:
            weight, bias = self._parameters[0], None

        self._data = F.conv2d(x, weight, bias, self.stride, self.pad_size,  self._data)
        if self.activation is not None:
            return self.activation.forward(self._data)
        return self._data

    def backward(self, gradients: F.Tensor = None):
        if self.activation is not None:
            self.activation.backward()
        super().backward()
