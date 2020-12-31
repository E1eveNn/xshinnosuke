from optim.optimizer import *


class Adam(Optimizer):
    def __init__(self, parameters=None, lr=0.001, weight_decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.ms = None
        self.vs = None
        super(Adam, self).__init__(parameters=parameters, lr=lr, weight_decay=weight_decay)

    def step(self):
        self.iterations += 1
        if self.ms is None:
            # initialize
            self.ms = [np.zeros_like(p.eval) for p in self._parameters]
        if self.vs is None:
            # initialize
            self.vs = [np.zeros_like(p.eval) for p in self._parameters]

        for i, (v, m, var) in enumerate(zip(self.vs, self.ms, self._parameters)):
            if var.requires_grad:
                v = self.beta1 * v + (1 - self.beta1) * var.grad.eval
                m = self.beta2 * m + (1 - self.beta2) * np.square(var.grad.eval)
                v_correct = v / (1 - pow(self.beta1, self.iterations))
                m_correct = m / (1 - pow(self.beta2, self.iterations))
                var.eval -= self.lr * (v_correct / (np.sqrt(m_correct) + self.epsilon))
                self.ms[i] = m
                self.vs[i] = v

        super(Adam, self).step()
