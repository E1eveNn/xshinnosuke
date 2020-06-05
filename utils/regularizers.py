from ..nn.core import Variable


class Regularizer:
    def __call__(self, x):
        return 0.


class L1L2(Regularizer):
    def __init__(self, l1: float = 0., l2: float = 0.):
        self.l1 = Variable(l1, requires_grad=False)
        self.l2 = Variable(l2, requires_grad=False)

    def __call__(self, x: Variable):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * x.abs().sum()
        if self.l2:
            regularization += self.l2 * (x ** 2).sum()
        return regularization


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)
