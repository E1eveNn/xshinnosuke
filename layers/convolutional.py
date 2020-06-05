from ..utils.initializers import get_initializer
from ..utils.activators import get_activator
from ..nn.core import Layer, Variable
from ..nn import global_graph as GlobalGraph
from ..nn.functional import conv2d, max_pool2d, avg_pool2d
from ..nn.grad_fn import Conv2DBackward, Maxpool2DBackward, Avgpool2DBackward
from typing import List, Tuple, Union


class Conv2D(Layer):
    def __init__(self, out_channels: int, kernel_size: Union[int, Tuple], use_bias: bool = False, stride: Union[int, Tuple] = 1, padding: Union[int, str] = 0,
                 activation: str = None, kernel_initializer: str = 'Normal', bias_initializer: str = 'zeros',
                 input_shape: Tuple = None,
                 **kwargs):
        self.out_channels = out_channels
        self.kernel_size = self.__check_size(kernel_size)
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.stride = self.__check_size(stride)
        self.padding = padding
        self.activation = get_activator(activation) if activation is not None else None
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.pad_size = padding
        self.cols = None
        super(Conv2D, self).__init__(input_shape=input_shape, **kwargs)

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        return kernel_size

    def initial_params(self, input_shape: Tuple = None):
        if input_shape is not None:
            self.input_shape = input_shape
        w = Variable(self.kernel_initializer((self.out_channels, self.input_shape[0], self.kernel_size[0],
                                              self.kernel_size[1])), name='variable')
        if self.use_bias:
            b = Variable(self.bias_initializer(1, self.out_channels), name='variable')
        else:
            b = None
        self.variables.append(w)
        self.variables.append(b)

    def compute_output_shape(self, input_shape: Tuple = None):
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

    def __call__(self, inbound):
        # if isinstance(inbound, Variable):
        if inbound.data is not None:
            if GlobalGraph.inputs is None:
                GlobalGraph.inputs = inbound

            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = conv2d(inbound, self.variables[0], self.variables[1], self.stride, self.pad_size)
            if self.activation is not None:
                output = self.activation.__call__(output)
            # output是一个Variable
            return output

        super(Conv2D, self).__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        w, b = self.variables
        self.data = conv2d(self.input_data, w, b, self.stride, self.pad_size)
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
        Conv2DBackward(self.data)


class MaxPooling2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = self.__check_size(kernel_size) if stride is None else self.__check_size(stride)
        self.padding = padding
        self.mode = 'reshape'
        super(MaxPooling2D, self).__init__()

    def compute_output_shape(self, input_shape=None):
        n_c, n_h_prev, n_w_prev = input_shape
        if self.kernel_size == self.stride:
            self.mode = 'reshape'
        else:
            self.mode = 'im2col'
        n_h, n_w = (n_h_prev - self.kernel_size + 2 * self.padding) // self.stride[0] + 1, (n_w_prev - self.kernel_size
                                                                                            + 2 * self.padding) // self.stride[1] + 1
        return n_c, n_h, n_w

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        return kernel_size

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            output = max_pool2d(inbound, self.kernel_size, self.stride, self.padding)
            # output是一个Variable
            return output
        super(MaxPooling2D, self).__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = max_pool2d(self.input_data, self.kernel_size, self.stride, self.padding)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Maxpool2DBackward(self.data)


class AvgPooling2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = self.__check_size(kernel_size) if stride is None else self.__check_size(stride)
        self.padding = padding
        super(AvgPooling2D, self).__init__()

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        return kernel_size

    def compute_output_shape(self, input_shape=None):
        n_c, n_h_prev, n_w_prev = input_shape
        n_h, n_w = (n_h_prev - self.kernel_size + 2 * self.padding) // self.stride[0] + 1, \
                   (n_w_prev - self.kernel_size + 2 * self.padding) // self.stride[1] + 1
        return n_c, n_h, n_w

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            output = avg_pool2d(inbound, self.kernel_size, self.stride, self.padding)
            # output是一个Variable
            return output
        super(AvgPooling2D, self).__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = avg_pool2d(self.input_data, self.kernel_size, self.stride, self.padding)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Avgpool2DBackward(self.data)
