from .adam import Adam
from .sgd import SGD, Momentum
from .adagrad import AdaGrad
from .adadelta import AdaDelta
from .rmsprop import RMSprop
from optim.optimizer import Optimizer
import copy


def get_optimizer(optimizer, **kwargs):
    parameters = kwargs.pop('parameters', None)
    lr = kwargs.pop('lr', 0.001)
    weight_decay = kwargs.pop('weight_decay', 0.)
    if optimizer.__class__.__name__ == 'str':
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            return SGD(parameters=parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            return Adam(parameters=parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'rmsprop':
            return RMSprop(parameters=parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'momentum':
            return Momentum(parameters=parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            return AdaGrad(parameters=parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adadelta':
            return AdaDelta(parameters=parameters, lr=lr, weight_decay=weight_decay)
    elif isinstance(optimizer, Optimizer):
        return copy.deepcopy(optimizer)
    else:
        raise ValueError('unknown optimizer type!')