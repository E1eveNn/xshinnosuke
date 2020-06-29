from ..nn.core import Layer, Variable
from ..nn.global_graph import np
from functools import reduce
from ..nn.initializers import get_initializer
from ..nn.functional import dropout2d, batchnorm_2d
from ..nn.grad_fn import Dropout2DBackward, Batchnorm2DBackward


class Dropout(Layer):
    def __init__(self, keep_prob: float):
        # prob :probability of keeping a unit active.
        self.keep_prob = keep_prob
        self.mask = None
        super(Dropout, self).__init__()

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            output = dropout2d(inbound, self.keep_prob)
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
        gamma = Variable(self.gamma_initializer(n_in))
        beta = Variable(self.beta_initializer(n_in))
        self.variables.append(gamma)
        self.variables.append(beta)
        self.moving_mean = self.moving_mean_initializer(n_in)
        self.moving_variance = self.moving_variance_initializer(n_in)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = self.forward(inbound)
            return output
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x

        self.data = batchnorm_2d(self.input_data, self.variables[0], self.variables[1], self.axis, self.epsilon,
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
        gamma = Variable(self.gamma_initializer(self.input_shape))
        beta = Variable(self.beta_initializer(self.input_shape))
        self.variables.append(gamma)
        self.variables.append(beta)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = self.forward(inbound)
            return output
        Layer.__call__(self, inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        inputs = self.input_data.data
        self.shape_field = tuple([i for i in range(1, inputs.ndim)])
        gamma, beta = self.variables
        #calc mean
        mean = np.mean(inputs, axis=self.shape_field, keepdims=True)
        #calc var
        var = np.var(inputs, axis=self.shape_field, keepdims=True)

        #x minus u
        xmu = inputs - mean
        sqrtvar = np.sqrt(var + self.epsilon)
        normalized_x = xmu / sqrtvar
        outputs = gamma.data * normalized_x + beta.data
        self.cache = (xmu, sqrtvar, normalized_x)
        self.data = Variable(data=outputs)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        grad = self.data.grad
        xmu, sqrtvar, normalized_x = self.cache
        std_inv = 1. / sqrtvar

        N = reduce(lambda x, y: x * y, normalized_x.shape[1:])
        gamma, beta = self.variables
        if gamma.requires_grad:
            gamma.grad += np.sum(grad * normalized_x, axis=0)
        if beta.requires_grad:
            beta.grad += np.sum(grad, axis=0)

        dnormalized_x = grad * gamma.data
        dvar = (-0.5) * np.sum(dnormalized_x * xmu, axis=self.shape_field, keepdims=True) * (std_inv ** 3) # (m,1)=(m,c,h,w)*(m,c,h,w)*(m,1)

        dmean = (-1.0) * np.sum(dnormalized_x * std_inv, axis=self.shape_field, keepdims=True) - 2.0 * dvar * np.mean(xmu, axis=self.shape_field, keepdims=True)

        outputs = dnormalized_x * std_inv + (2. / N) * dvar * xmu + (1. / N) * dmean

        if self.input_data.requires_grad:
            self.input_data.grad += outputs


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
        gamma = Variable(self.gamma_initializer((1, c, 1, 1)))
        beta = Variable(self.beta_initializer((1, c, 1, 1)))
        self.variables.append(gamma)
        self.variables.append(beta)

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            output = self.forward(inbound)
            return output
        Layer.__call__(self, inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x

        inputs = self.input_data.data
        gamma, beta = self.variables
        n, c, h, w = inputs.shape
        self.shape_field = tuple([i for i in range(2, inputs.ndim)])

        x_group = np.reshape(inputs, (n, self.G, c // self.G, h, w))
        mean = np.mean(x_group, axis=self.shape_field, keepdims=True)
        var = np.var(x_group, axis=self.shape_field, keepdims=True)
        xgmu = x_group - mean
        sqrtvar = np.sqrt(var + self.epsilon)
        x_group_norm = xgmu / sqrtvar

        x_norm = np.reshape(x_group_norm, (n, c, h, w))

        outputs = gamma.data * x_norm + beta.data

        self.cache = (xgmu, sqrtvar, x_norm)
        self.data = Variable(data=outputs)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        grad = self.data.grad
        xgmu, sqrtvar, x_norm = self.cache
        std_inv = 1. / sqrtvar
        gamma, beta = self.variables
        n, c, h, w = grad.shape
        if gamma.requires_grad:
            gamma.grad += np.sum(grad * x_norm, axis=(0, 2, 3), keepdims=True)
        if beta.requires_grad:
            beta.grad += np.sum(grad, axis=(0, 2, 3), keepdims=True)

        # dx_group_norm
        dx_norm = grad * gamma.data  # (N,C,H,W)
        dx_group_norm = np.reshape(dx_norm, (n, self.G, c // self.G, h, w))
        # dvar
        dvar = -0.5 * (std_inv ** 3) * np.sum(dx_group_norm * xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean
        N_GROUP = c // self.G * h * w
        dmean1 = np.sum(dx_group_norm * -std_inv, axis=(2, 3, 4), keepdims=True)
        dmean2_var = dvar * -2.0 / N_GROUP * np.sum(xgmu, axis=(2, 3, 4), keepdims=True)
        dmean = dmean1 + dmean2_var
        # dx_group
        dx_group1 = dx_group_norm * std_inv
        dx_group_var = dvar * 2.0 / N_GROUP * xgmu
        dx_group_mean = dmean * 1.0 / N_GROUP
        dx_group = dx_group1 + dx_group_var + dx_group_mean
        # dx
        outputs = np.reshape(dx_group, (n, c, h, w))

        if self.input_data.requires_grad:
            self.input_data.grad += outputs
