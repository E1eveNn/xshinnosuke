from .base import *


class Activation(Layer):
    def __init__(self, act_name: str = 'relu', **kwargs):
        self.activation = get_activator(act_name)
        super(Activation, self).__init__(**kwargs)

    def init_layer_out_tensor(self, x: F.Tensor = None):
        super().init_layer_out_tensor(x)
        self.activation._data = self._data

    def call(self, x, *args, **kwargs) -> F.Tensor:
        self._data = self.activation.call(x)
        return self._data


class ReLU(Layer):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def init_layer_out_tensor(self, x = None):
        if self.inplace:
            self._data = x
            self._data.cache['mask'] = np.empty(self._data.shape, dtype=np.bool)
        else:
            if self._data is None or x.shape[0] > self._data.shape_capacity[0]:
                self._data = Zeros()((x.shape[0],) + self.shape, requires_grad=self.trainable)
                self._data.to('static')
                self._data.add_in_bounds(x)
            elif x.shape[0] < self._data.shape_capacity[0]:
                self._data.slices(slice(None, x.shape[0], None))
            else:
                self._data.slices(slice(None, None, None))

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        if self.inplace:
            self._data = x
            if GLOBAL.TRAINING and x.requires_grad:
                if 'grad_fn' not in self._data.cache.keys():
                    self._data.cache['grad_fn'] = []
                self._data.cache['grad_fn'].append(self._data.grad_fn)
                self._data.cache['mask'] = x.eval < 0
                self._data.grad_fn = F.ReLUBackward
        self._data = F.relu(x, self.inplace, self._data)
        if GLOBAL.TRAINING and x.requires_grad:
            self._data.cache['inplace'] = self.inplace
            initialize_ops_grad(x)
        return self._data

    def backward(self, gradients: F.Tensor = None):
        if gradients is not None:
            assert isinstance(gradients, F.Tensor) and gradients.shape == self._data.shape
            self._data.grad = gradients
        if self._data.grad_fn is not None:
            self._data.grad_fn(self._data)
            if not self.inplace:
                self._data.zero_grad()


class Sigmoid(Layer):
    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.sigmoid(x,  self._data)
        return self._data


class Tanh(Layer):
    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self._data = F.tanh(x,  self._data)
        return self._data


def get_activator(activator):
    if activator.__class__.__name__ == 'str':
        activator = activator.lower()
        if activator == 'relu':
            return ReLU()
        elif activator == 'sigmoid':
            return Sigmoid()
        elif activator == 'tanh':
            return Tanh()
        return None
    elif isinstance(activator, Activation):
        return F.copy.deepcopy(activator)
    else:
        raise TypeError('unknown activator type!')
