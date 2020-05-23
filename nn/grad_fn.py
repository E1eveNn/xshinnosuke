# 写关于Node和Layer运算所需要的函数
import numpy as np
from utils.toolkit import col2im


def AddBackward(outputs):
    for in_bound in outputs.in_bounds:
        if in_bound.requires_grad:
            in_bound.grad += outputs.grad


def IAddBackward(outputs):
    if outputs.in_bounds[-1].requires_grad:
        outputs.in_bounds[-1].grad += outputs.grad


def NegBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += -outputs.grad


def MatmulBackward(outputs):
    # z = x * y, dloss/dx = dloss/dz * dz/dx = dloss/dz * y
    x, y = outputs.in_bounds
    if x.requires_grad:
        x.grad += np.dot(outputs.grad, y.data.T)
    if y.requires_grad:
        y.grad += np.dot(x.data.T, outputs.grad)


def TransposeBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad.reshape(inputs.shape)


def MultiplyBackward(outputs):
    length = len(outputs.in_bounds)
    product_except_self_list = [0] * length
    product_except_self_list[0] = 1
    for i in range(1, length):
        product_except_self_list[i] = product_except_self_list[i - 1] * outputs.in_bounds[i - 1].data
    right = 1
    for i in reversed(range(length)):
        product_except_self_list[i] *= right
        right *= outputs.in_bounds[i].data

    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            in_bound.grad += outputs.grad * product_except_self_list[i]


def SumBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += np.ones_like(inputs.data) * outputs.grad


def MeanBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        mean_nums = inputs.shape / outputs.shape
        inputs.grad += np.ones_like(inputs.data) * outputs.grad / mean_nums


def AbsBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        grad_ = np.asarray(outputs.data == inputs.data, dtype=np.int8)
        grad_[grad_ == 0] = -1
        inputs.grad += grad_


def ViewBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad.reshape(inputs.shape)


def LogBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        base = outputs.cache['base'] if outputs.cache['base'] != 'e' else np.e
        inputs.grad += outputs.grad / (inputs.data * np.log(base))


def ExpBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad * outputs.data


def PowBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        n_times = inputs.power
        if n_times < 1:
            inputs.grad += n_times * outputs.grad / np.power(inputs.data, 1 - n_times)
        else:
            inputs.grad += n_times * np.power(inputs.data, n_times - 1) * outputs.grad


def ReluBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        grad = outputs.grad
        grad[outputs.data < 0] = 0
        inputs.grad += grad


def SigmoidBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad * outputs.data * (1 - outputs.data)


def TanhBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad * (1 - np.square(outputs.data))


def FlattenBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad.reshape(inputs.shape)


def DenseBackward(outputs):
    inputs, weight, bias = outputs.in_bounds
    # 正向传播时是 inputs -> dense -> outputs
    if weight.requires_grad:
        weight.grad += inputs.data.T.dot(outputs.grad)
    if bias is not None and bias.requires_grad:
        bias.grad += np.sum(outputs.grad, axis=0, keepdims=True)
    if inputs.requires_grad:
        inputs.grad += outputs.grad.dot(weight.data.T)


def Conv2DBackward(outputs):
    inputs, weight, bias = outputs.in_bounds
    n, in_channels, h, w = inputs.data.shape
    out_channels, _, kernel_h, kernel_w = weight.data.shape
    _, _, out_h, out_w = outputs.grad.shape
    grad_reshaped = outputs.grad.transpose(1, 0, 2, 3).reshape(out_channels, -1)
    if weight.requires_grad:
        weight.grad += grad_reshaped.dot(outputs.cache['col']).reshape(out_channels, in_channels, kernel_h, kernel_w)

    if bias is not None and bias.requires_grad:
        bias.grad += np.sum(outputs.grad, axis=(0, 2, 3))

    dcol = grad_reshaped.T.dot(weight.data.reshape(out_channels, -1))

    if inputs.requires_grad:
        inputs.grad += col2im(inputs.shape, outputs.cache['pad_size'], kernel_h, kernel_w,  outputs.cache['stride'],
                              dcol)


def Maxpool2DBackward(outputs):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        dx_reshaped = np.zeros_like(outputs.cache['x_reshaped'])
        out_newaxis = outputs.data[:, :, :, np.newaxis, :, np.newaxis]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        dout_newaxis = outputs.grad[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        grad = dx_reshaped.reshape(inputs.data.shape)
        if outputs.cache['pad_size']:
            grad = grad[:, :, outputs.cache['pad_size']: -outputs.cache['pad_size'], outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        grad = outputs.grad.transpose(0, 2, 3, 1)
        dmax = np.zeros((grad.size, outputs.cache['kernel_size'] * outputs.cache['kernel_size']))
        dmax[np.arange(outputs.cache['pool_argmax'].size), outputs.cache['pool_argmax'].flatten()] = grad.flatten()
        dmax = dmax.reshape(grad.shape + (outputs.cache['kernel_size'] * outputs.cache['kernel_size'],))

        dcol = dmax.reshape(np.prod(dmax.shape[:3]), -1)
        grad = col2im(inputs.shape, outputs.cache['pad_size'], outputs.cache['kernel_size'],
                         outputs.cache['kernel_size'], outputs.cache['stride'], dcol)

    if inputs.requires_grad:
        inputs.grad += grad


def Avgpool2DBackward(outputs):
    inputs, = outputs.in_bounds
    grad = outputs.grad.transpose(0, 2, 3, 1)
    dmean = np.repeat(grad.flatten(), outputs.cache['pool_argmean'].size) / (outputs.cache['kernel_size'] * outputs.cache['kernel_size'])
    dmean = dmean.reshape(grad.shape + (outputs.cache['kernel_size'] * outputs.cache['kernel_size'],))
    dcol = dmean.reshape(np.prod(dmean.shape[:3]), -1)
    grad = col2im(inputs.shape, outputs.cache['pad_size'], outputs.cache['kernel_size'],
                         outputs.cache['kernel_size'], outputs.cache['stride'], dcol)

    if inputs.requires_grad:
        inputs.grad += grad


def Pad2DBackward(outputs):
    inputs, = outputs.in_bounds
    pad_size = outputs.cache['pad_size']
    if inputs.requires_grad:
        inputs.grad += outputs.grad[:, :, pad_size[0]: -pad_size[0], pad_size[1]: -pad_size[1]]


def Dropout2DBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += outputs.grad * outputs.cache['mask'] * outputs.cache['keep_prob']


def MeanSquaredbackward(outputs):
    y_pred, y_true = outputs.in_bounds
    gradients = outputs.grad * (y_pred.data - y_true.data) / y_pred.shape[0]
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = gradients
        else:
            y_true.grad += gradients

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = gradients
        else:
            y_pred.grad += gradients


def MeanAbsolutebackward(outputs):
    y_pred, y_true = outputs.in_bounds
    pos = np.where((y_pred.data - y_true.data) < 0)
    mask = np.ones_like(y_pred.data)
    mask[pos] = -1

    gradients = mask / y_pred.data.shape[0]
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = gradients
        else:
            y_true.grad += gradients

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = gradients
        else:
            y_pred.grad += gradients


def BinaryCrossEntropyBackward(outputs):
    y_pred, y_true = outputs.in_bounds
    avg = np.prod(np.asarray(y_pred.data.shape[:-1]))
    gradients = (np.divide(1 - y_true.data, 1 - y_pred.data) - np.divide(y_true.data, y_pred.data)) / avg
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = gradients
        else:
            y_true.grad += gradients

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = gradients
        else:
            y_pred.grad += gradients


def SparseCrossEntropyBackward(outputs):
    y_pred, y_true = outputs.in_bounds
    # # before softmax
    before_softmax_y_pred = y_pred.in_bounds[0]
    avg = np.prod(np.asarray(y_pred.data.shape[:-1]))
    gradients = (y_pred.data - y_true.data) / avg
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = gradients
        else:
            y_true.grad += gradients

    if before_softmax_y_pred.requires_grad:
        if before_softmax_y_pred.grad is None:
            before_softmax_y_pred.grad = gradients
        else:
            before_softmax_y_pred.grad += gradients


def CrossEntropyBackward(outputs):
    y_pred, y_true = outputs.in_bounds
    # # before softmax
    before_softmax_y_pred = y_pred.in_bounds[0]
    to_sum_dim = np.prod(y_pred.data.shape[:-1])
    last_dim = y_pred.data.shape[-1]
    n = y_pred.data.shape[0]
    probs = y_pred.data.reshape(-1, last_dim)
    y_flat = y_true.data.reshape(to_sum_dim)
    probs[np.arange(to_sum_dim), y_flat] -= 1
    gradients = probs.reshape(y_pred.data.shape) / n
    # if y_true.requires_grad:
    #     y_true.grad += gradients
    if before_softmax_y_pred.requires_grad:
        if before_softmax_y_pred.grad is None:
            before_softmax_y_pred.grad = gradients
        else:
            before_softmax_y_pred.grad += gradients
