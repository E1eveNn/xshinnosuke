import core.base
from utils.common import *


class Initializer:
    def __init__(self, seed=None):
        if seed is not None:
            GLOBAL.np.random.seed(seed)

    def decompose_size(self, shape):
        shape = GLOBAL.np.array(shape)
        if shape.ndim == 2:
            fan_in, fan_out = shape
        elif shape.ndim == 4 or shape.ndim == 5:
            respective_field_size = GLOBAL.np.prod(shape[2:])
            fan_in = shape[1] * respective_field_size
            fan_out = shape[0] * respective_field_size
        else:
            fan_in = fan_out = int(GLOBAL.np.sqrt(GLOBAL.np.prod(shape)))
        return fan_in, fan_out


class Uniform(Initializer):
    def __init__(self, scale=0.05, seed=None):
        self.scale = scale
        super(Uniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        return core.base.Tensor(GLOBAL.np.random.uniform(-self.scale, self.scale, size=shape), **kwargs)


class LecunUniform(Initializer):
    def __init__(self, seed=None):
        super(LecunUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(GLOBAL.np.sqrt(3. / fan_in))(shape, **kwargs)


class GlorotUniform(Initializer):
    def __init__(self, seed=None):
        super(GlorotUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(GLOBAL.np.sqrt(6. / (fan_in + fan_out)))(shape, **kwargs)


class HeUniform(Initializer):
    def __init__(self, seed=None):
        super(HeUniform, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Uniform(GLOBAL.np.sqrt(6. / fan_in))(shape, **kwargs)


class Normal(Initializer):
    def __init__(self, std=0.1, mean=0.0, seed=None):
        self.std = std
        self.mean = mean
        super(Normal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        return core.base.Tensor(GLOBAL.np.random.normal(loc=self.mean, scale=self.std, size=shape), **kwargs)


class LecunNormal(Initializer):
    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(GLOBAL.np.sqrt(1. / fan_in))(shape, **kwargs)


class GlorotNormal(Initializer):
    def __init__(self, seed=None):
        super(GlorotNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(GLOBAL.np.sqrt(2. / (fan_in + fan_out)))(shape, **kwargs)


class HeNormal(Initializer):
    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = self.decompose_size(shape)
        return Normal(GLOBAL.np.sqrt(2. / fan_in))(shape, **kwargs)


class Orthogonal(Initializer):
    def __init__(self, gain=1.0, seed=None):
        if gain == 'relu':
            gain = GLOBAL.np.sqrt(2)
        self.gain = gain
        super(Orthogonal, self).__init__(seed)

    def __call__(self, shape, **kwargs):
        shape = GLOBAL.np.array(shape)
        flat_shape = (shape[0].tolist(), GLOBAL.np.prod(shape[1:]).tolist())
        a = Normal(1.)(flat_shape, **kwargs)
        u, _, v = GLOBAL.np.linalg.svd(a.data, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape.tolist())
        a.data.data = self.gain * q
        return a


class Zeros(Initializer):
    def __call__(self, shape, **kwargs):
        return core.base.Tensor(GLOBAL.np.zeros(shape), **kwargs)


class Ones(Initializer):
    def __call__(self, shape, **kwargs):
        return core.base.Tensor(GLOBAL.np.ones(shape), **kwargs)


class Matrix(Initializer):
    def __init__(self, value: float, seed=None):
        super().__init__(seed=seed)
        self.value = value

    def __call__(self, shape: Tuple, **kwargs):
        return core.base.Tensor(GLOBAL.np.ones(shape) * self.value, **kwargs)


class RandN(Initializer):
    def __call__(self, shape: Tuple, **kwargs):
        return core.base.Tensor(GLOBAL.np.random.randn(*shape), **kwargs)


class Rand(Initializer):
    def __call__(self, shape: Tuple, **kwargs):
        return core.base.Tensor(GLOBAL.np.random.rand(*shape), **kwargs)


class RandInt(Initializer):
    def __call__(self, low, high=None, shape=None, **kwargs):
        return core.base.Tensor(GLOBAL.np.random.randint(low=low, high=high, size=shape), **kwargs)


def matrix(*shape, value=0., **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return Matrix(value=value)(shape, requires_grad=requires_grad, **kwargs)


def ones(*shape, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return Ones()(shape, requires_grad=requires_grad, **kwargs)


def ones_like(a, **kwargs):
    requires_grad = kwargs.pop('requires_grad', a.requires_grad)
    return core.base.Tensor(GLOBAL.np.ones_like(a.data), requires_grad=requires_grad, **kwargs)


def zeros(*shape, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return Zeros()(shape, requires_grad=requires_grad, **kwargs)


def zeros_like(a, **kwargs):
    requires_grad = kwargs.pop('requires_grad', a.requires_grad)
    return core.base.Tensor(GLOBAL.np.zeros_like(a.data), requires_grad=requires_grad, **kwargs)


def rand(*shape, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return Rand()(shape, requires_grad=requires_grad, **kwargs)


def randn(*shape, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return RandN()(shape, requires_grad=requires_grad, **kwargs)


def randint(low, high=None, shape=None, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return RandInt()(low, high, shape, requires_grad=requires_grad, **kwargs)


def tensor(data, **kwargs):
    requires_grad = kwargs.pop('requires_grad', False)
    return core.base.Tensor(data=data, requires_grad=requires_grad, **kwargs)


def manual_seed_all(seeds=None):
    GLOBAL.np.random.seed(seeds)


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
        elif initializer == 'orthogonal':
            return Orthogonal()
        elif initializer in ['glorotnoraml', 'glorot_normal']:
            return GlorotNormal()
        elif initializer in ['glorotuniform', 'glorot_uniform']:
            return GlorotUniform()

    elif isinstance(initializer, Initializer):
        return copy.deepcopy(initializer)
    else:
        raise TypeError('unknown initialization type!')

