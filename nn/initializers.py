from .global_graph import np
from .core import Variable
import copy
from typing import Tuple


class Initializer:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def decompose_size(self, shape):
        shape = np.array(shape)
        if shape.ndim == 2:
            fan_in, fan_out = shape
        elif shape.ndim == 4 or shape.ndim == 5:
            respective_field_size = np.prod(shape[2:])
            fan_in = shape[1] * respective_field_size
            fan_out = shape[0] * respective_field_size
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(shape)))
        return fan_in, fan_out


class Uniform(Initializer):
    def __init__(self, scale=0.05, seed=None):
        self.scale = scale
        super(Uniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.random.uniform(-self.scale, self.scale, size=shape), name=name)


class LecunUniform(Initializer):
    def __init__(self, seed=None):
        super(LecunUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Uniform(np.sqrt(3. / fan_in))(shape, name=name)


class GlorotUniform(Initializer):
    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Uniform(np.sqrt(6. / (fan_in + fan_out)))(shape, name=name)


class HeUniform(Initializer):
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Uniform(np.sqrt(6. / fan_in))(shape, name=name)


class Normal(Initializer):
    def __init__(self, std=0.1, mean=0.0, seed=None):
        self.std = std
        self.mean = mean
        super(Normal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.random.normal(loc=self.mean, scale=self.std, size=shape), name=name)


class LecunNormal(Initializer):
    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Normal(np.sqrt(1. / fan_in))(shape, name=name)


class GlorotNormal(Initializer):
    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Normal(np.sqrt(2. / (fan_in + fan_out)))(shape, name=name)


class HeNormal(Initializer):
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        name = kwargs.pop('name', None)
        return Normal(np.sqrt(2. / fan_in))(shape, name=name)


class Orthogonal(Initializer):
    def __init__(self, gain=1.0, seed=None):
        if gain == 'relu':
            gain = np.sqrt(2)
        self.gain = gain
        super(Orthogonal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        shape = np.array(shape)
        flat_shape = (shape[0].tolist(), np.prod(shape[1:]).tolist())
        name = kwargs.pop('name', None)
        a = Normal(1.)(flat_shape, name=name)
        u, _, v = np.linalg.svd(a.data, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape.tolist())
        a.data = self.gain * q
        return a


class Zeros(Initializer):
    def __call__(self, shape, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.zeros(shape), name=name)


class Ones(Initializer):
    def __call__(self, shape, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.ones(shape), name=name)


class Matrix(Initializer):
    def __init__(self, value: float, seed=None):
        super().__init__(seed=seed)
        self.value = value

    def __call__(self, shape: Tuple, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.ones(shape) * self.value, name=name)
    

class RandN(Initializer):
    def __call__(self, shape: Tuple, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.random.randn(*shape), name=name)


class Rand(Initializer):
    def __call__(self, shape: Tuple, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.random.rand(*shape), name=name)


class RandInt(Initializer):
    def __call__(self, low, high=None, shape=None, **kwargs):
        name = kwargs.pop('name', None)
        return Variable(np.random.randint(low=low, high=high, size=shape), name=name)


def ones(*shape, **kwargs):
    return Ones()(shape, **kwargs)


def ones_like(a, **kwargs):
    return np.ones_like(a, **kwargs)


def zeros(*shape, **kwargs):
    return Zeros()(shape, **kwargs)


def zeros_like(a, **kwargs):
    return np.zeros_like(a, **kwargs)


def rand(*shape, **kwargs):
    return Rand()(shape, **kwargs)


def randn(*shape, **kwargs):
    return RandN()(shape, **kwargs)


def randint(low, high=None, shape=None, **kwargs):
    return RandInt()(low, high, shape, **kwargs)


def get_initializer(initializer):
    if initializer.__class__.__name__ == 'str':
        initializer = initializer.lower()
        if initializer == 'zeros':
            return Zeros()
        elif initializer == 'ones':
            return Ones()
        elif initializer in ['heuniform', 'he_uniform']:
            return HeUniform()
        elif initializer == 'uniform':
            return Uniform()
        elif initializer == 'normal':
            return Normal()
        elif initializer in ['henormal', 'he_normal']:
            return HeNormal()
        elif initializer in ['lecunnormal', 'lecun_normal']:
            return LecunNormal()
        elif initializer in ['lecununiform', 'lecun_uniform']:
            return LecunUniform()
        elif initializer =='orthogonal':
            return Orthogonal()
        elif initializer in ['glorotnoraml', 'glorot_normal']:
            return GlorotNormal()
        elif initializer in ['glorotuniform', 'glorot_uniform']:
            return GlorotUniform()

    elif isinstance(initializer, Initializer):
        return copy.deepcopy(initializer)
    else:
        raise TypeError('unknown initialization type!')
