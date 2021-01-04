import numpy as np
from utils.common import ndarray


# base math operation
def add(a: ndarray, b: ndarray, out: ndarray = None):
    return np.add(a, b, out=out)


def sub(a: ndarray, b: ndarray, out: ndarray = None):
    return np.subtract(a, b, out=out)


def mul(a: ndarray, b: ndarray, out: ndarray = None):
    return np.multiply(a, b, out=out)


def div(a: ndarray, b: ndarray, out: ndarray = None):
    return np.divide(a, b, out=out)


def floor_div(a: ndarray, b: ndarray, out: ndarray = None):
    return np.true_divide(a, b, out=out)


def mm(a: ndarray, b: ndarray, out: ndarray = None):
    return np.dot(a, b, out=out)


def exp(x: ndarray, out: ndarray = None):
    return np.exp(x, out=out)


def max(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return np.max(x, axis=axis, keepdims=keepdims, out=out)


def maximum(x1: ndarray, x2: ndarray, out: ndarray = None):
    return np.maximum(x1, x2, out=out)


def sum(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return np.sum(x, axis, keepdims=keepdims, out=out)


def mean(x: ndarray, axis: int = None, keepdims: bool = False, out: ndarray = None):
    return np.mean(x, axis, keepdims=keepdims, out=out)


def norm(x: ndarray, p: int = 2, axis: int = None, keepdims: bool = False, out: ndarray = None):
    x = np.abs(x)
    x = np.power(x, p)
    out = np.sum(x, axis=axis, keepdims=keepdims, out=out)
    out = np.power(out, 1 / p)
    return out


# base activation
def relu(x: ndarray, out: ndarray = None):
    return np.maximum(0., x, out=out)


def sigmoid(x: ndarray, out: ndarray = None):
    out = np.exp(-x, out=out)
    out = np.add(1., out, out=out)
    return np.divide(1., out, out=out)


def tanh(x: ndarray, out: ndarray = None):
    return np.tanh(x, out=out)


def softmax(x: ndarray, out: ndarray = None):
    # more stable softmax
    out = np.subtract(x, max(x, axis=-1, keepdims=True), out=out)
    np.exp(out, out=out)
    np.divide(out, sum(out, axis=-1, keepdims=True), out=out)
    return out


def log_softmax(x: ndarray, out: ndarray = None):
    out = softmax(x, out)
    out = np.log(out, out=out)
    return out


# base nn function
def flatten(x: ndarray, start: int = 1):
    output_shape = tuple(x.shape[:start]) + (-1,)
    return np.reshape(x, output_shape)


def long(data: ndarray):
    return data.astype(np.int64)


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
        return np.expand_dims(inputs, axis=new_axis_list)
    except TypeError:
        for a in new_axis_list:
            inputs = np.expand_dims(inputs, axis=a)
        return inputs


def nll_loss(pred: ndarray, target: ndarray, reduction: str = 'mean', out: ndarray = None):
    to_sum_dim = np.prod(np.asarray(pred.shape[:-1]))
    log_probs = pred.reshape(-1, pred.shape[-1])
    y_flat = target.reshape(to_sum_dim).astype(np.int)
    if reduction == 'sum':
        sum_val = -np.sum(log_probs[np.arange(to_sum_dim), y_flat])
        out = np.multiply(-1, sum_val, out=out)
    elif reduction == 'mean':
        sum_val = -np.sum(log_probs[np.arange(to_sum_dim), y_flat])
        out = np.divide(sum_val, pred.shape[0], out=out)
    else:
        out = np.multiply(-1, log_probs[np.arange(to_sum_dim), y_flat], out=out)
    out = np.abs(out)
    return out


def bce_loss(pred: ndarray, target: ndarray, reduction: str = 'mean', out: ndarray = None):
    if reduction == 'sum':
        loss_val = np.sum(np.add(np.multiply(target, np.log(pred)),
                           np.multiply(np.subtract(1, target), np.log(np.subtract(1, pred)))))
        out = np.multiply(-1, loss_val, out=out)
    elif reduction == 'mean':
        loss_val = np.mean(np.add(np.multiply(target, np.log(pred)),
                                 np.multiply(np.subtract(1, target), np.log(np.subtract(1, pred)))))
        out = np.multiply(-1, loss_val, out=out)
    else:
        out = np.add(np.multiply(target, np.log(pred)),
                           np.multiply(np.subtract(1, target), np.log(np.subtract(1, pred))), out=out)
        out = np.multiply(-1, out, out=out)

    return out
