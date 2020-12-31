from optim.optimizer import *


class AdaGrad(Optimizer):
    def __init__(self, parameters=None, lr=0.01, weight_decay=0., epsilon=1e-7):
        self.epsilon = epsilon
        self.ms = None
        super(AdaGrad, self).__init__(parameters=parameters, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            self.ms = [np.zeros_like(g.eval) for g in self._parameters]
        for i, (s, var)in enumerate(zip(self.ms, self._parameters)):
            if var.requires_grad:
                s += np.power(var.grad.eval, 2)
                var.eval -= self.lr * var.grad.eval / np.sqrt(s + self.epsilon)
                self.ms[i] = s
        super(AdaGrad, self).step()
