from optim.optimizer import *


class AdaDelta(Optimizer):
    def __init__(self, parameters=None, lr=1.0, weight_decay=0.0, rho=0.95, epsilon=1e-7):
        self.rho = rho
        self.epsilon = epsilon
        self.ms = None
        self.delta_x = None
        super(AdaDelta, self).__init__(parameters=parameters, lr=lr, weight_decay=weight_decay)

    def step(self):
        if self.ms is None:
            self.ms = [np.zeros_like(g.grad.eval) for g in self._parameters]
        if self.delta_x is None:
            self.delta_x = [np.zeros_like(g.grad.eval) for g in self._parameters]

        for i, (s, var, x) in enumerate(zip(self.ms, self._parameters, self.delta_x)):
            if var.requires_grad:
                s = self.rho * s + (1 - self.rho) * np.power(var.grad.eval, 2)
                g_ = np.sqrt((x + self.epsilon) / (s + self.epsilon)) * var.grad.eval
                var.eval -= g_
                x = self.rho * x + (1 - self.rho) * np.power(g_, 2)
                self.ms[i] = s
                self.delta_x[i] = x
        super(AdaDelta,self).step()
