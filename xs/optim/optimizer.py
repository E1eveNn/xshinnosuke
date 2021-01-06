from utils.common import copy
from nn.initializers import Zeros


class Optimizer:
    def __init__(self, parameters=None, lr=0.01, weight_decay=0.):
        self._parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations = 0

    def zero_grad(self):
        for v in self._parameters:
            if v.grad is None:
                v.zero_grad()
            else:
                v.grad.zero_()

    def step(self):
        self.iterations += 1
