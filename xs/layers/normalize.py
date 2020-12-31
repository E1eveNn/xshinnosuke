from layers.activators import get_activator
from nn.initializers import get_initializer
from .base import *
import nn.functional as F


class Dropout(Layer):
    def __init__(self, keep_prob: float, **kwargs):
        # prob :probability of keeping a unit active.
        self.keep_prob = keep_prob
        self.__data = None
        super(Dropout, self).__init__(**kwargs)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        self.__data = F.dropout2d(x, self.keep_prob,  self.__data)
        return self.__data


class BatchNormalization(Layer):
    def __init__(self, epsilon: float = 1e-6, momentum: float = 0.99, axis: int = 1, gamma_initializer: str = 'ones',
                 beta_initializer: str = 'zeros', moving_mean_initializer: str = 'zeros',
                 moving_variance_initializer: str = 'ones', **kwargs):
        # axis=1 when input Fully Connected Layers(data shape:(M,N),where M donotes Batch-size,and N represents feature nums)  ---also axis=-1 is the same
        # axis=1 when input Convolution Layers(data shape:(M,C,H,W),represents Batch-size,Channels,Height,Width,respectively)
        self.epsilon = epsilon
        self.axis = axis
        self.momentum = momentum
        self.gamma_initializer = get_initializer(gamma_initializer)
        self.beta_initializer = get_initializer(beta_initializer)
        self.moving_mean_initializer = get_initializer(moving_mean_initializer)
        self.moving_variance_initializer = get_initializer(moving_variance_initializer)
        self.moving_mean = None
        self.moving_variance = None
        super(BatchNormalization, self).__init__(**kwargs)

    def init_params(self, input_shape: Tuple = None, *args):
        if input_shape is not None:
            self._input_shape = input_shape
        assert len(self._input_shape) >= 1
        n_in = self._input_shape[self.axis - 1]
        gamma = F.Parameter(self.gamma_initializer(n_in, requires_grad=True))
        beta = F.Parameter(self.beta_initializer(n_in, requires_grad=True))
        self.moving_mean = self.moving_mean_initializer(n_in)
        self.moving_variance = self.moving_variance_initializer(n_in)
        self._parameters.append(gamma)
        self._parameters.append(beta)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        gamma, beta = self._parameters
        self._data = F.batch_norm(x, gamma, beta, self.moving_mean, self.moving_variance, self.axis, GLOBAL.TRAINING,
                                  self.epsilon, self.momentum, self._data)
        return self._data


class LayerNormalization(Layer):
    def __init__(self, epsilon: float = 1e-10, gamma_initializer: str = 'ones', beta_initializer: str = 'zeros'):
        self.epsilon = epsilon
        self.gamma_initializer = get_initializer(gamma_initializer)
        self.beta_initializer = get_initializer(beta_initializer)
        self.__data = None
        super(LayerNormalization, self).__init__()

    def init_params(self, input_shape: Tuple = None, *args):
        if input_shape is not None:
            self.input_shape = input_shape
        gamma = self.gamma_initializer(self.input_shape)
        beta = self.beta_initializer(self.input_shape)
        self._parameters.append(gamma)
        self._parameters.append(beta)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        gamma, beta = self._parameters
        self.__data = F.layernorm2d(x, gamma, beta,
                                 self.epsilon, self.__data)
        return self.__data


class GroupNormalization(Layer):
    def __init__(self, epsilon: float = 1e-5, groups: int = 16, gamma_initializer: str = 'ones', beta_initializer: str = 'zeros'):
        self.epsilon = epsilon
        self.G = groups
        self.gamma_initializer = get_initializer(gamma_initializer)
        self.beta_initializer = get_initializer(beta_initializer)
        self.__data = None
        super(GroupNormalization, self).__init__()

    def init_params(self, input_shape: Tuple = None, *args):
        if input_shape is not None:
            self.input_shape = input_shape
        c = self.input_shape[0]
        assert c % self.G == 0

        gamma = self.gamma_initializer((1, c, 1, 1))
        beta = self.beta_initializer((1, c, 1, 1))
        self._parameters.append(gamma)
        self._parameters.append(beta)

    def call(self, x: F.Tensor, *args, **kwargs) -> F.Tensor:
        gamma, beta = self._parameters
        self.__data = F.groupnorm2d(x, gamma, beta,
                                 self.epsilon, self.G, self.__data)
        return self.__data
