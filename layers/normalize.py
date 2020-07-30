from ..nn.core import Layer, Variable
from ..nn.initializers import get_initializer
from ..nn.functional import dropout2d, batchnorm2d, layernorm2d, groupnorm2d
from ..nn.grad_fn import Dropout2DBackward, Batchnorm2DBackward, Layernorm2DBackward, Groupnorm2DBackward


class Dropout(Layer):
    def __init__(self, keep_prob: float):
        # prob :probability of keeping a unit active.
        self.keep_prob = keep_prob
        self.mask = None
        super(Dropout, self).__init__()

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            is_training = kwargs.pop('is_training', True)
            output = dropout2d(inbound, self.keep_prob, training=is_training)
            # output是一个Variable
            return output
        super(Dropout, self).__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = dropout2d(self.input_data, self.keep_prob, is_training)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Dropout2DBackward(self.data)


class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-6, momentum=0.99, axis=1, gamma_initializer='ones', beta_initializer='zeros',
                 moving_mean_initializer='zeros', moving_variance_initializer='ones', **kwargs):
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
        self.cache = None
        super(BatchNormalization, self).__init__(**kwargs)

    def initial_params(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        assert len(self.input_shape) >= 1
        n_in = self.input_shape[self.axis - 1]
        gamma = self.gamma_initializer(n_in, name='xs_variable')
        beta = self.beta_initializer(n_in, name='xs_variable')
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean = self.moving_mean_initializer(n_in)
        self.moving_variance = self.moving_variance_initializer(n_in)

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            is_training = kwargs.pop('is_training', True)
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = batchnorm2d(inbound, self.variables[0], self.variables[1], self.axis, self.epsilon, is_training,
                                 self.momentum, self.moving_mean, self.moving_variance)
            return output
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x

        self.data = batchnorm2d(self.input_data, self.variables[0], self.variables[1], self.axis, self.epsilon,
                                is_training, self.momentum, self.moving_mean, self.moving_variance)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Batchnorm2DBackward(self.data)


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-10, gamma_initializer='ones', beta_initializer='zeros'):
        self.epsilon = epsilon
        self.gamma_initializer = get_initializer(gamma_initializer)
        self.beta_initializer = get_initializer(beta_initializer)
        self.shape_field = None
        self.cache = None
        super(LayerNormalization, self).__init__()

    def initial_params(self, input_shape=None):
        gamma = self.gamma_initializer(self.input_shape, name='xs_variable')
        beta = self.beta_initializer(self.input_shape, name='xs_variable')
        self.variables.append(gamma)
        self.variables.append(beta)

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            is_training = kwargs.pop('is_training', True)
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = layernorm2d(inbound, self.variables[0], self.variables[1], is_training, self.epsilon)
            return output
        Layer.__call__(self, inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x

        self.data = layernorm2d(self.input_data, self.variables[0], self.variables[1], is_training, self.epsilon)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Layernorm2DBackward(self.data)


class GroupNormalization(Layer):
    def __init__(self, epsilon=1e-5, groups=16, gamma_initializer='ones', beta_initializer='zeros'):
        self.epsilon = epsilon
        self.G = groups
        self.gamma_initializer = get_initializer(gamma_initializer)
        self.beta_initializer = get_initializer(beta_initializer)
        self.shape_field = None
        self.cache = None
        super(GroupNormalization, self).__init__()

    def initial_params(self, input_shape=None):
        c = self.input_shape[0]
        assert c % self.G == 0
        gamma = self.gamma_initializer((1, c, 1, 1), name='xs_variable')
        beta = self.beta_initializer((1, c, 1, 1), name='xs_variable')
        self.variables.append(gamma)
        self.variables.append(beta)

    def __call__(self, inbound, *args, **kwargs):
        if isinstance(inbound, Variable):
            is_training = kwargs.pop('is_training', True)
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = groupnorm2d(inbound, self.variables[0], self.variables[1], is_training, self.epsilon, self.G)
            return output
        Layer.__call__(self, inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x

        self.data = groupnorm2d(self.input_data, self.variables[0], self.variables[1], is_training, self.epsilon, self.G)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        Groupnorm2DBackward(self)
