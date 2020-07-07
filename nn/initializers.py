from .global_graph import np
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

    def __call__(self, shape):
        return np.random.uniform(-self.scale, self.scale, shape=shape)


class LecunUniform(Initializer):
    def __init__(self, seed=None):
        super(LecunUniform, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(np.sqrt(3. / fan_in))(shape)


class GlorotUniform(Initializer):
    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(np.sqrt(6. / (fan_in + fan_out)))(shape)


class HeUniform(Initializer):
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(np.sqrt(6. / fan_in))(shape)


class Normal(Initializer):
    def __init__(self, std=0.1, mean=0.0, seed=None):
        self.std = std
        self.mean = mean
        super(Normal, self).__init__(seed)

    def __call__(self, shape):
        return np.random.normal(loc=self.mean, scale=self.std, shape=shape)


class LecunNormal(Initializer):
    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(np.sqrt(1. / fan_in))(shape)


class GlorotNormal(Initializer):
    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(np.sqrt(2. / (fan_in + fan_out)))(shape)


class HeNormal(Initializer):
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed)

    def __call__(self, shape):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(np.sqrt(2. / fan_in))(shape)


class Orthogonal(Initializer):
    def __init__(self, gain=1.0, seed=None):
        if gain == 'relu':
            gain = np.sqrt(2)
        self.gain = gain
        super(Orthogonal, self).__init__(seed)

    def __call__(self, shape):
        shape = np.array(shape)
        flat_shape = (shape[0].tolist(), np.prod(shape[1:]).tolist())
        a = Normal(1.)(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape.tolist())
        q = self.gain * q
        return q


class Zeros(Initializer):
    def __call__(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    def __call__(self, shape):
        return np.ones(shape)


class Matrix(Initializer):
    def __init__(self, value: float, seed=None):
        super().__init__(seed=seed)
        self.value = value

    def __call__(self, shape: Tuple):
        return np.ones(shape) * self.value
    

class RandN(Initializer):
    def __call__(self, shape: Tuple):
        return np.random.randn(shape)


class Rand(Initializer):
    def __call__(self, shape: Tuple):
        return np.random.rand(shape)


class RandInt(Initializer):
    def __call__(self, low, high=None, shape=None):
        return np.random.randint(low=low, high=high, size=shape)


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
