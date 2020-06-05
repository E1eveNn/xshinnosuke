from ..nn.global_graph import np
import copy


class Optimizer:
    def __init__(self, trainable_variables=None, lr=0.1, weight_decay=0.):
        self.trainable_variables = trainable_variables
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations = 0

    def zero_grad(self):
        for v in self.trainable_variables:
            v.grad = np.zeros_like(v.data)

    def step(self):
        self.iterations += 1


class SGD(Optimizer):
    def step(self):
        for v in self.trainable_variables:
            v.data = v.data - self.lr * v.grad
        super(SGD, self).step()


class Momentum(Optimizer):
    def __init__(self, trainable_variables=None, lr=0.01, weight_decay=0., rho=0.9):
        self.rho = rho
        self.velocity = None
        super(Momentum, self).__init__(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.velocity is None:
            # initialize
            self.velocity = [np.zeros_like(p.data) for p in self.trainable_variables]

        for i, (v, var) in enumerate(zip(self.velocity, self.trainable_variables)):
            v = self.rho * v + (1 - self.rho) * var.grad
            var.data -= self.lr * v
            self.velocity[i] = v

        super(Momentum, self).step()


class RMSprop(Optimizer):
    def __init__(self, trainable_variables=None, lr=0.001, weight_decay=0., rho=0.9, epsilon=1e-7):
        self.rho = rho
        self.epsilon = epsilon
        self.ms = None
        super(RMSprop, self).__init__(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            # initialize
            self.ms = [np.zeros_like(p.data) for p in self.trainable_variables]

        for i, (s, var) in enumerate(zip(self.ms, self.trainable_variables)):
            new_s = self.rho * s + (1 - self.rho) * np.square(var.grad)
            var.data -= self.lr * var.grad / np.sqrt(new_s + self.epsilon)
            self.ms[i] = new_s

        super(RMSprop, self).step()


class AdaGrad(Optimizer):
    def __init__(self, trainable_variables=None, lr=0.01, weight_decay=0., epsilon=1e-7):
        self.epsilon = epsilon
        self.ms = None
        super(AdaGrad, self).__init__(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            self.ms = [np.zeros_like(g.data) for g in self.trainable_variables]
        for i, (s, var)in enumerate(zip(self.ms, self.trainable_variables)):
            s += np.power(var.grad, 2)
            var.data -= self.lr * var.grad / np.sqrt(s + self.epsilon)
            self.ms[i] = s
        super(AdaGrad, self).step()


class AdaDelta(Optimizer):
    def __init__(self, trainable_variables=None, lr=1.0, weight_decay=0.0, rho=0.95, epsilon=1e-7):
        self.rho = rho
        self.epsilon = epsilon
        self.ms = None
        self.delta_x = None
        super(AdaDelta, self).__init__(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            self.ms = [np.zeros_like(g.grad) for g in self.trainable_variables]
        if self.delta_x is None:
            self.delta_x = [np.zeros_like(g.grad) for g in self.trainable_variables]

        for i, (s, var, x) in enumerate(zip(self.ms, self.trainable_variables, self.delta_x)):
            s = self.rho * s + (1 - self.rho) * np.power(var.grad, 2)
            g_ = np.sqrt((x + self.epsilon) / (s + self.epsilon)) * var.grad
            var.data -= g_
            x = self.rho * x + (1 - self.rho) * np.power(g_, 2)
            self.ms[i] = s
            self.delta_x[i] = x
        super(AdaDelta,self).step()


class Adam(Optimizer):
    def __init__(self, trainable_variables=None, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.ms = None
        self.vs = None
        super(Adam, self).__init__(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)

    def step(self):
        self.iterations += 1
        if self.ms is None:
            # initialize
            self.ms = [np.zeros_like(p.data) for p in self.trainable_variables]
        if self.vs is None:
            # initialize
            self.vs = [np.zeros_like(p.data) for p in self.trainable_variables]

        for i, (v, m, var) in enumerate(zip(self.vs, self.ms, self.trainable_variables)):
            v = self.beta1 * v + (1 - self.beta1) * var.grad
            m = self.beta2 * m + (1 - self.beta2) * np.square(var.grad)
            v_correct = v / (1 - pow(self.beta1, self.iterations))
            m_correct = m / (1 - pow(self.beta2, self.iterations))
            var.data -= self.lr * (v_correct / (np.sqrt(m_correct) + self.epsilon))
            self.ms[i] = m
            self.vs[i] = v

        super(Adam, self).step()


def get_optimizer(optimizer, **kwargs):
    trainable_variables = kwargs.pop('trainable_variables', None)
    lr = kwargs.pop('lr', 0.1)
    weight_decay = kwargs.pop('weight_decay', 0.)
    if optimizer.__class__.__name__ == 'str':
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            return SGD(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            return Adam(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'rmsprop':
            return RMSprop(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'momentum':
            return Momentum(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            return AdaGrad(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adadelta':
            return AdaDelta(trainable_variables=trainable_variables, lr=lr, weight_decay=weight_decay)
    elif isinstance(optimizer, Optimizer):
        return copy.deepcopy(optimizer)
    else:
        raise ValueError('unknown optimizer type!')
