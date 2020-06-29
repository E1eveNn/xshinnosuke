from .global_graph import np
import copy
from typing import Tuple


class Initializer:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def decompose_size(self, size):
        size = np.array(size)
        if size.ndim == 2:
            fan_in, fan_out = size
        elif size.ndim == 4 or size.ndim == 5:
            respective_field_size = np.prod(size[2:])
            fan_in = size[1] * respective_field_size
            fan_out = size[0] * respective_field_size
        else:
            fan_in = fan_out = int(np.sqrt(np.prod(size)))
        return fan_in, fan_out


class Uniform(Initializer):
    def __init__(self, scale=0.05, seed=None):
        self.scale = scale
        super(Uniform, self).__init__(seed)

    def __call__(self, size):
        return np.random.uniform(-self.scale, self.scale, size=size)


class LecunUniform(Initializer):
    def __init__(self, seed=None):
        super(LecunUniform, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform(Initializer):
    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Uniform(np.sqrt(6. / (fan_in + fan_out)))(size)


class HeUniform(Initializer):
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Uniform(np.sqrt(6. / fan_in))(size)


class Normal(Initializer):
    def __init__(self, std=0.1, mean=0.0, seed=None):
        self.std = std
        self.mean = mean
        super(Normal, self).__init__(seed)

    def __call__(self, size):
        return np.random.normal(loc=self.mean, scale=self.std, size=size)


class LecunNormal(Initializer):
    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Normal(np.sqrt(1. / fan_in))(size)


class GlorotNormal(Initializer):
    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Normal(np.sqrt(2. / (fan_in + fan_out)))(size)


class HeNormal(Initializer):
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed)

    def __call__(self, size):
        fan_in, fan_out = self.decompose_size(size)
        return Normal(np.sqrt(2. / fan_in))(size)


class Orthogonal(Initializer):
    def __init__(self, gain=1.0, seed=None):
        if gain == 'relu':
            gain = np.sqrt(2)
        self.gain = gain
        super(Orthogonal, self).__init__(seed)

    def __call__(self, size):
        size = np.array(size)
        flat_shape = (size[0].tolist(), np.prod(size[1:]).tolist())
        a = Normal(1.)(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(size.tolist())
        q = self.gain * q
        return q


class Zeros(Initializer):
    def __call__(self, size):
        return np.zeros(size)


class Ones(Initializer):
    def __call__(self, size):
        return np.ones(size)


class Constant(Initializer):
    def __init__(self, value: float, seed=None):
        super().__init__(seed=seed)
        self.value = value

    def __call__(self, size: Tuple):
        return np.ones(size) * self.value


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
