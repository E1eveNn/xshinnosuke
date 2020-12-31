from optim.optimizer import *


class SGD(Optimizer):
    def step(self):
        for v in self._parameters:
            if v.requires_grad:
                v.eval -= self.lr * v.grad.eval
        super(SGD, self).step()


class Momentum(Optimizer):
    def __init__(self, parameters=None, lr=0.01, weight_decay=0., rho=0.9):
        self.rho = rho
        self.velocity = None
        super(Momentum, self).__init__(parameters=parameters, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.velocity is None:
            # initialize
            self.velocity = [np.zeros_like(p.eval) for p in self._parameters]

        for i, (v, var) in enumerate(zip(self.velocity, self._parameters)):
            if var.requires_grad:
                v = self.rho * v + (1 - self.rho) * var.grad.eval
                var.eval -= self.lr * v
                self.velocity[i] = v

        super(Momentum, self).step()
