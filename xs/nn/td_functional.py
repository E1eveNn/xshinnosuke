from core import __global as GLOBAL
from utils.common import ndarray


# base math operation
def add(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.add(a, b, out=out)


def sub(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.subtract(a, b, out=out)


def mul(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.multiply(a, b, out=out)


def div(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.divide(a, b, out=out)


def floor_div(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.true_divide(a, b, out=out)


def mm(a: ndarray, b: ndarray, out: ndarray = None):
    return GLOBAL.np.dot(a, b, out=out)


def exp(x: ndarray, out: ndarray = None):
    return GLOBAL.np.exp(x, out=out)


def max(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return GLOBAL.np.max(x, axis=axis, keepdims=keepdims, out=out)


def maximum(x1: ndarray, x2: ndarray, out: ndarray = None):
    return GLOBAL.np.maximum(x1, x2, out=out)


def sum(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return GLOBAL.np.sum(x, axis, keepdims=keepdims, out=out)


def mean(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return GLOBAL.np.mean(x, axis, keepdims=keepdims, out=out)


def norm(x: ndarray, p: int = 2, axis: int = None, keepdims: bool = False, out: ndarray = None):
    x = GLOBAL.np.abs(x)
    x = GLOBAL.np.power(x, p)
    out = GLOBAL.np.sum(x, axis=axis, keepdims=keepdims, out=out)
    out = GLOBAL.np.power(out, 1 / p)
    return out


# base activation
def relu(x: ndarray, out: ndarray = None):
    return GLOBAL.np.maximum(0., x, out=out)


def sigmoid(x: ndarray, out: ndarray = None):
    out = GLOBAL.np.exp(-x, out=out)
    out = GLOBAL.np.add(1., out, out=out)
    return GLOBAL.np.divide(1., out, out=out)


def tanh(x: ndarray, out: ndarray = None):
    return GLOBAL.np.tanh(x, out=out)


def softmax(x: ndarray, out: ndarray = None):
    # more stable softmax
    out = GLOBAL.np.subtract(x, max(x, axis=-1, keepdims=True), out=out)
    GLOBAL.np.exp(out, out=out)
    GLOBAL.np.divide(out, sum(out, axis=-1, keepdims=True), out=out)
    return out


def log_softmax(x: ndarray, out: ndarray = None):
    out = softmax(x, out)
    out = GLOBAL.np.log(out, out=out)
    return out


# base nn function
def flatten(x: ndarray, start: int = 1):
    output_shape = tuple(x.shape[:start]) + (-1,)
    return GLOBAL.np.reshape(x, output_shape)


def long(data: ndarray):
    return data.astype(GLOBAL.np.int64)


def expand_as(inputs: ndarray, target: ndarray):
    new_axis_list = []
    inputs_idx = 0
    for i in range(target.ndim):
        if inputs_idx >= inputs.ndim:
            new_axis_list.append(i)
        elif inputs.shape[inputs_idx] == target.shape[i]:
            inputs_idx += 1
        else:
            new_axis_list.append(i)
    try:
        return GLOBAL.np.expand_dims(inputs, axis=new_axis_list)
    except TypeError:
        for a in new_axis_list:
            inputs = GLOBAL.np.expand_dims(inputs, axis=a)
        return inputs


def nll_loss(pred: ndarray, target: ndarray, reduction: str = 'mean', out: ndarray = None):
    to_sum_dim = GLOBAL.np.prod(GLOBAL.np.asarray(pred.shape[:-1])).item()
    log_probs = pred.reshape(-1, pred.shape[-1])
    y_flat = target.reshape(to_sum_dim).astype(GLOBAL.np.int)
    if reduction == 'sum':
        sum_val = -GLOBAL.np.sum(log_probs[GLOBAL.np.arange(to_sum_dim), y_flat])
        out = GLOBAL.np.multiply(-1, sum_val, out=out)
    elif reduction == 'mean':
        sum_val = -GLOBAL.np.sum(log_probs[GLOBAL.np.arange(to_sum_dim), y_flat])
        out = GLOBAL.np.divide(sum_val, pred.shape[0], out=out)
    else:
        out = GLOBAL.np.multiply(-1, log_probs[GLOBAL.np.arange(to_sum_dim), y_flat], out=out)
    out = GLOBAL.np.abs(out)
    return out


def bce_loss(pred: ndarray, target: ndarray, reduction: str = 'mean', out: ndarray = None):
    if reduction == 'sum':
        loss_val = GLOBAL.np.sum(GLOBAL.np.add(GLOBAL.np.multiply(target, GLOBAL.np.log(pred)),
                           GLOBAL.np.multiply(GLOBAL.np.subtract(1, target), GLOBAL.np.log(GLOBAL.np.subtract(1, pred)))))
        out = GLOBAL.np.multiply(-1, loss_val, out=out)
    elif reduction == 'mean':
        loss_val = GLOBAL.np.mean(GLOBAL.np.add(GLOBAL.np.multiply(target, GLOBAL.np.log(pred)),
                                 GLOBAL.np.multiply(GLOBAL.np.subtract(1, target), GLOBAL.np.log(GLOBAL.np.subtract(1, pred)))))
        out = GLOBAL.np.multiply(-1, loss_val, out=out)
    else:
        out = GLOBAL.np.add(GLOBAL.np.multiply(target, GLOBAL.np.log(pred)),
                           GLOBAL.np.multiply(GLOBAL.np.subtract(1, target), GLOBAL.np.log(GLOBAL.np.subtract(1, pred))), out=out)
        out = GLOBAL.np.multiply(-1, out, out=out)

    return out
