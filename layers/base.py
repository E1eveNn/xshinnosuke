from ..nn.core import Layer, Variable
from ..nn.global_graph import np
from ..nn.functional import pad_2d
from ..nn.grad_fn import Pad2DBackward, NegBackward, MultiplyBackward, MatmulBackward, LogBackward, ExpBackward, SumBackward, MeanBackward, AbsBackward, PowBackward
from typing import Tuple, Union, List


class Input(Layer):
    def __init__(self, input_shape: Union[List, Tuple], **kwargs):
        super(Input, self).__init__(input_shape=input_shape, **kwargs)
        self.input_shape = input_shape
        self.shape = self.compute_output_shape(input_shape)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            return inbound
        super().__call__(inbound)

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        if input_shape is None:
            return self.input_shape
        return input_shape

    def forward(self, x: Variable = None, is_training:bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = self.input_data
        self.connect_init(self.data, is_training)
        return self.data


class ZeroPadding2D(Layer):
    def __init__(self, pad_size: Union[int, float, Tuple], **kwargs):
        self.pad_h, self.pad_w = self.__check_pad_size(pad_size)
        super(ZeroPadding2D, self).__init__(**kwargs)

    def __check_pad_size(self, pad_size: Union[int, float, Tuple]):
        if isinstance(pad_size, (int, float)):
            return int(pad_size), int(pad_size)
        return pad_size

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            output = pad_2d(inbound, (self.pad_h, self.pad_w))
            # output是一个Variable
            return output
        super(ZeroPadding2D, self).__call__(inbound)
        return self

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) ->  Union[List, Tuple]:
        c, h, w = input_shape
        return c, h + 2 * self.pad_h, w + 2 * self.pad_w

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = pad_2d(self.input_data, (self.pad_h, self.pad_w))
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Pad2DBackward(self.data)


class Add(Layer):
    def __call__(self, inbounds: List[Layer]):
        # inbounds只能是Layer数组
        for inbound in inbounds:
            inbound.out_bounds.append(self)
            self.in_bounds.append(inbound)
            self.shape = inbound.shape
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        data = 0
        for in_bound in self.in_bounds:
            data += in_bound.data.data
        self.data = Variable(data, in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        for in_bound in self.in_bounds:
            if in_bound.data.requires_grad:
                in_bound.data.grad += self.data.grad


class Negative(Layer):
    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=-self.input_data.data, in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        NegBackward(self.data)


class Multiply(Layer):
    def __call__(self, inbounds: List[Layer]):
        for inbound in inbounds:
            super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        data = 1
        for in_bound in self.in_bounds:
            data = np.multiply(data, in_bound.data)
        self.data = Variable(data=data, in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        MultiplyBackward(self.data)


class Matmul(Layer):
    def __call__(self, inbounds: List[Layer]):
        """
        只支持两个Layer做矩阵乘
        """
        assert len(inbounds) == 2
        for inbound in inbounds:
            inbound.out_bounds.append(self)
            self.in_bounds.append(inbound)
        self.shape = inbounds[0].shape[:-1] + inbounds[-1].shape[1:]
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        data = self.in_bounds[0].data.dot(self.in_bounds[1].data)
        self.data = Variable(data=data, in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        MatmulBackward(self.data)


class Log(Layer):
    def __init__(self, base: Union[int, str]='e'):
        self.base = self.__check_base(base)
        super().__init__()

    def __check_base(self, base: Union[int, str]):
        if base == 'e':
            return np.e
        elif base in [2, 10]:
            return base
        else:
            raise ValueError('unknown base value {}'.format(base))

    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x

        if self.base == 2:
            data = np.log2(self.in_bounds[0].data)
        elif self.base == 10:
            data = np.log10(self.in_bounds[0].data)
        else:
            data = np.log(self.in_bounds[0].data)
        self.data = Variable(data=data, in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        LogBackward(self.data)


class Exp(Layer):
    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=np.exp(self.in_bounds[0].data), in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        ExpBackward(self.data)


class Sum(Layer):
    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return tuple([1])

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=np.sum(self.in_bounds[0].data), in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        SumBackward(self.data)


class Mean(Layer):
    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return tuple([1])

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=np.mean(self.in_bounds[0].data), in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        MeanBackward(self.data)


class Abs(Layer):
    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=np.abs(self.in_bounds[0].data), in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        AbsBackward(self.data)


class Pow(Layer):
    def __init__(self, exponent:int = 2):
        self.exponent = exponent
        super().__init__()

    def __call__(self, inbound: Layer):
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training: bool = True, *args):
        if x is not None:
            self.input_data = x
        self.data = Variable(data=np.power(self.in_bounds[0].data, self.exponent), in_bounds=self.in_bounds)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        PowBackward(self.data)
