from .base import *


class Input(Layer):
    def __init__(self, input_shape: Union[List, Tuple], **kwargs):
        super(Input, self).__init__(input_shape=input_shape, **kwargs)
        self._shape = input_shape

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = x
        return self._data


class Reshape(Layer):
    def __init__(self, shape: Tuple, **kwargs):
        super().__init__(shape=shape, **kwargs)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.view(x, (-1, ) + self._shape, self._data)
        return self._data

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return self._shape


class ZeroPadding2D(Layer):
    def __init__(self, padding, **kwargs):
        self.padding = padding
        super(ZeroPadding2D, self).__init__(**kwargs)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.pad2d(x, self.padding, self._data)
        return self._data

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        self._shape = (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1])
        return self._shape


class Add(Layer):
    def __call__(self, inbounds: List[Layer], *args, **kwargs):
        for inbound in inbounds:
            self._in_bounds.append(inbound)
            inbound.add_out_bounds(self)
            self._shape = inbound.shape
        return self

    def init_layer_out_tensor(self, x : F.Tensor = None):
        x = self._in_bounds[0].data if x is None else x
        if self._data is None or x.shape[0] > self._data.shape_capacity[0]:
            self._data = Zeros()((x.shape[0],) + self.shape, requires_grad=self.trainable)
            self._data.to('static')
            for in_bound in self._in_bounds:
                self._data.add_in_bounds(in_bound.data)
        elif x.shape[0] < self._data.shape_capacity[0]:
            if GLOBAL.TRAINING:
                self._data.slices(slice(None, x.shape[0], None))
            else:
                self._data = Zeros()((x.shape[0],) + self.shape, requires_grad=self.trainable)
                self._data.to('static')
                for in_bound in self._in_bounds:
                    self._data.add_in_bounds(in_bound.data)
        else:
            self._data.slices(slice(None, None, None))

    def forward(self, x: F.Tensor = None, *args, **kwargs) -> F.Tensor:
        self._data.zero_()
        for in_bound in self._in_bounds:
            GLOBAL.np.add(self._data.eval, in_bound.data.eval, out=self._data.eval)
            if GLOBAL.TRAINING and in_bound.data.requires_grad:
                initialize_ops_grad(in_bound.data)
            self._data.requires_grad = self._data.requires_grad or in_bound.data.requires_grad
        return self._data

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return self._shape

    def backward(self, gradients: F.Tensor = None):
        for in_bound in self._in_bounds:
            if in_bound.data.requires_grad:
                GLOBAL.np.add(in_bound.data.grad.eval, self._data.grad.eval, out=in_bound.data.grad.eval)
        self._data.zero_grad()
