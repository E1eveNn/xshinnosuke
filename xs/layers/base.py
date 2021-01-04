from utils.common import *
from nn.initializers import Zeros, get_initializer
import nn.functional as F


class Layer:
    def __init__(self, trainable: bool = True, name: str = None, in_bounds: List = None,
                 input_shape: Union[List, Tuple] = None, shape: Union[List, Tuple] = None,
                 parameters: List[F.Tensor] = None,  **kwargs):
        self.trainable = trainable
        self._in_bounds = in_bounds if in_bounds is not None else []
        self._out_bounds = []
        self._input_shape = input_shape
        self.name = str(name)
        self._shape = shape
        self._parameters = parameters if parameters is not None else []
        self._data = None
        super(Layer, self).__init__(**kwargs)

    def __call__(self, x, *args, **kwargs):
        if isinstance(x, F.Tensor):
            if len(self._parameters) == 0:
                self.init_params(x.shape[1:])
            x.next_layers.append(self)
            return self.call(x, *args, **kwargs)
        # elif isinstance(x, Layer):
        else:
            self._shape = self.compute_output_shape(x.shape)
            self._in_bounds.append(x)
            self._input_shape = x.shape
            x.add_out_bounds(self)
            return self
        # else:
        #     raise TypeError('unknown type for {}'.format(x))

    def connect(self, inbound=None):
        if inbound is None:
            if self._input_shape is None:
                raise ValueError('must specify input_shape')
        else:
            self._input_shape = inbound.shape
        self._shape = self.compute_output_shape(self._input_shape)
        if inbound is not None:
            self._in_bounds.append(inbound)
            inbound.add_out_bounds(self)

    def params_count(self) -> int:
        total_params = 0
        for v in self._parameters:
            if v is not None:
                total_params += v.eval.size
        return total_params

    @property
    def shape(self):
        return self._shape

    @property
    def out_bounds(self):
        return self._out_bounds

    @property
    def in_bounds(self):
        return self._in_bounds

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v

    def parameters(self, parameters: List[F.Tensor] = None):
        if parameters is None:
            return self._parameters
        else:
            self._parameters = parameters

    def add_out_bounds(self, *outs):
        for out in outs:
            if isinstance(out, Layer):
                self._out_bounds.append(out)

    def init_params(self, input_shape: Tuple = None, *args):
        pass
    
    def init_layer_out_tensor(self, x: F.Tensor = None):
        x = self._in_bounds[0].data if x is None else x
        if self._data is None or x.shape[0] > self._data.shape_capacity[0]:
            self._data = Zeros()((x.shape[0],) + self.shape, requires_grad=self.trainable)
            self._data.to('static')
            self._data.add_in_bounds(x)
            self._data.add_in_bounds(*self.parameters())
        elif x.shape[0] < self._data.shape_capacity[0]:
            if GLOBAL.TRAINING:
                self._data.slices(slice(None, x.shape[0], None))
            else:
                self._data = Zeros()((x.shape[0],) + self.shape, requires_grad=self.trainable)
                self._data.to('static')
                self._data.add_in_bounds(x)
                self._data.add_in_bounds(*self.parameters())
        else:
            self._data.slices(slice(None, None, None))

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        if isinstance(input_shape, int):
            self._shape = (input_shape, )
        else:
            self._shape = tuple(input_shape)
        return self._shape

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        raise NotImplemented

    def forward(self, x: F.Tensor = None, *args, **kwargs) -> F.Tensor:
        if x is None:
            x = self._in_bounds[-1]
        if len(self._parameters) == 0:
            self.init_params(x.shape[1:])
        if isinstance(x, F.Tensor):
            if x.is_dynamic:
                x.next_layers.append(self)
            return self.call(x, *args, **kwargs)
        else:
            return self.call(x.data, *args, **kwargs)

    def backward(self, gradients: F.Tensor = None):
        if gradients is not None:
            assert isinstance(gradients, F.Tensor) and gradients.shape == self._data.shape
            self._data.grad = gradients
        if self._data.grad_fn is not None:
            self._data.grad_fn(self._data)
            self._data.zero_grad()
