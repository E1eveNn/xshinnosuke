from optim.optimizer import *


class RMSprop(Optimizer):
    def __init__(self, parameters=None, lr=0.001, weight_decay=0., rho=0.9, epsilon=1e-7):
        self.rho = rho
        self.epsilon = epsilon
        self.ms = None
        super(RMSprop, self).__init__(parameters=parameters, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            # initialize
            self.ms = [np.zeros_like(p.eval) for p in self._parameters]

        for i, (s, var) in enumerate(zip(self.ms, self._parameters)):
            if var.requires_grad:
                new_s = self.rho * s + (1 - self.rho) * np.square(var.grad.eval)
                var.eval -= self.lr * var.grad.eval / np.sqrt(new_s + self.epsilon)
                self.ms[i] = new_s

        super(RMSprop, self).step()