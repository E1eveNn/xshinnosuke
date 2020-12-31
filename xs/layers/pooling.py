from .base import *
import nn.functional as F


class MaxPooling2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0, **kwargs):
        self.kernel_size = kernel_size
        self.stride = self.__check_size(kernel_size) if stride is None else self.__check_size(stride)
        self.padding = padding
        self.__mode = 'reshape'
        super(MaxPooling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        n_c, n_h_prev, n_w_prev = input_shape
        if self.kernel_size == self.stride:
            self.__mode = 'reshape'
        else:
            self.__mode = 'im2col'
        n_h, n_w = (n_h_prev - self.kernel_size + 2 * self.padding) // self.stride[0] + 1, (n_w_prev - self.kernel_size
                                                                                            + 2 * self.padding) // self.stride[1] + 1
        return n_c, n_h, n_w

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        elif isinstance(kernel_size, (Tuple, List)):
            return kernel_size

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,  self._data)
        return self._data


class AvgPooling2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0, **kwargs):
        self.kernel_size = kernel_size
        self.stride = self.__check_size(kernel_size) if stride is None else self.__check_size(stride)
        self.padding = padding
        super(AvgPooling2D, self).__init__(**kwargs)

    def __check_size(self, kernel_size):
        if isinstance(kernel_size, (int, float)):
            return int(kernel_size), int(kernel_size)
        elif isinstance(kernel_size, (Tuple, List)):
            return kernel_size

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        n_c, n_h_prev, n_w_prev = input_shape
        n_h, n_w = (n_h_prev - self.kernel_size + 2 * self.padding) // self.stride[0] + 1, \
                   (n_w_prev - self.kernel_size + 2 * self.padding) // self.stride[1] + 1
        return n_c, n_h, n_w

    def call(self, x, *args, **kwargs) -> F.Tensor:
        self._data = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,  self._data)
        return self._data


class ChannelMaxPooling(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0, **kwargs):
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.__mode = 'reshape'
        super(ChannelMaxPooling, self).__init__(**kwargs)
        self.__data = None

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        n_c_prev, n_h, n_w = input_shape
        if self.kernel_size == self.stride:
            self.__mode = 'reshape'
        else:
            self.__mode = 'im2col'
        n_c = (n_c_prev - self.kernel_size + 2 * self.padding) / self.stride + 1
        return n_c, n_h, n_w

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self.__data = F.channel_max_pool(x, self.kernel_size, self.stride, self.padding,  self.__data)
        return self.__data


class ChannelAvgPooling(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.__mode = 'reshape'
        super(ChannelAvgPooling, self).__init__()
        self.__data = None

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        n_c_prev, n_h, n_w = input_shape
        if self.kernel_size == self.stride:
            self.__mode = 'reshape'
        else:
            self.__mode = 'im2col'
        n_c = (n_c_prev - self.kernel_size + 2 * self.padding) / self.stride + 1
        return n_c, n_h, n_w

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self.__data = F.channel_avg_pool(x, self.kernel_size, self.stride, self.padding,  self.__data)
        return self.__data