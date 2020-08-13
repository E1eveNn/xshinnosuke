# 写关于Node和Layer运算所需要的函数
from .global_graph import np
from .toolkit import col2im
from functools import reduce


def AddBackward(outputs):
    for in_bound in outputs.in_bounds:
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                in_bound.grad += np.sum(outputs.grad, axis=not_equal_axis[0])
            else:
                in_bound.grad += outputs.grad


def IAddBackward(outputs):
    inputs = outputs.in_bounds.pop()
    if inputs.requires_grad:
        inputs.grad += outputs.grad
    outputs.grad_fn = outputs.cache['grad_fn'].pop()
    outputs.grad_fn(outputs)


def SubBackward(outputs):
    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            if i == 0:
                in_bound.grad += outputs.grad
            else:
                in_bound.grad -= outputs.grad


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
        right = right * outputs.in_bounds[i].data

    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                in_bound.grad += np.sum(outputs.grad * product_except_self_list[i], axis=not_equal_axis[0])
            else:
                in_bound.grad += outputs.grad * product_except_self_list[i]


def SumBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad += np.ones_like(inputs.data) * outputs.grad


def MeanBackward(outputs):
    inputs, = outputs.in_bounds
    grad = outputs.grad
    if inputs.requires_grad:
        mean_nums = inputs.data.size / outputs.data.size
        if outputs.data.ndim < inputs.data.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = np.expand_dims(grad, axis)
        inputs.grad += np.ones_like(inputs.data) * grad / mean_nums


def MaxBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        grad = outputs.grad
        if outputs.data.ndim < inputs.data.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = np.expand_dims(grad, axis)
        mask = (inputs.data == np.max(inputs.data))
        inputs.grad += mask * grad


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
        n_times = outputs.cache['power']
        if n_times < 1:
            inputs.grad += n_times * outputs.grad / np.power(inputs.data, 1 - n_times)
        else:
            inputs.grad += n_times * np.power(inputs.data, n_times - 1) * outputs.grad


def ReluBackward(outputs):
    if outputs.cache['inplace']:
        mask = outputs.cache['mask']
        outputs.grad[mask] = 0
        outputs.grad_fn = outputs.cache['grad_fn'].pop()
        outputs.grad_fn(outputs)
    else:
        inputs, = outputs.in_bounds
        if inputs.requires_grad:
            grad = outputs.grad
            grad[inputs.data < 0] = 0
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
    inputs, = outputs.in_bounds
    weight, bias = outputs.get_variables()
    # 正向传播时是 inputs -> dense -> outputs
    if weight.requires_grad:
        weight.grad += inputs.data.T.dot(outputs.grad)
    if bias is not None and bias.requires_grad:
        bias.grad += np.sum(outputs.grad, axis=0, keepdims=True)
    if inputs.requires_grad:
        inputs.grad += outputs.grad.dot(weight.data.T)


def EmbeddingBackward(outputs):
    inputs,  = outputs.in_bounds
    weight, = outputs.get_variables()
    if weight.requires_grad:
        weight.grad[inputs.data.astype(np.int)] += outputs.grad


def Conv2DBackward(outputs):
    inputs, = outputs.in_bounds
    weight, bias = outputs.get_variables()
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
        inputs.grad += col2im(inputs.shape, outputs.cache['pad_size'], kernel_h, kernel_w, outputs.cache['stride'],
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
            grad = grad[:, :, outputs.cache['pad_size']: -outputs.cache['pad_size'],
                   outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        grad = outputs.grad.transpose(0, 2, 3, 1)
        dmax = np.zeros((grad.size, outputs.cache['kernel_size'] * outputs.cache['kernel_size']))
        dmax[np.arange(outputs.cache['pool_argmax'].size), outputs.cache['pool_argmax'].flatten()] = grad.flatten()
        dmax = dmax.reshape(grad.shape + (outputs.cache['kernel_size'] * outputs.cache['kernel_size'],))

        dcol = dmax.reshape(reduce(lambda x, y: x * y, dmax.shape[:3]), -1)
        grad = col2im(inputs.shape, outputs.cache['pad_size'], outputs.cache['kernel_size'],
                      outputs.cache['kernel_size'], outputs.cache['stride'], dcol)

    if inputs.requires_grad:
        inputs.grad += grad


def ChannelMaxpoolBackward(outputs):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        # （n, c // kernel_size, kernel_size, h, w）
        dx_reshaped = np.zeros_like(outputs.cache['x_reshaped'])
        # （n, c // kernel_size, 1, h, w）
        out_newaxis = outputs.data[:, :, np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        # （n, c // kernel_size, 1, h, w）
        dout_newaxis = outputs.grad[:, :, np.newaxis, :, :]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=2, keepdims=True)
        grad = dx_reshaped.reshape(inputs.data.shape)
        if outputs.cache['pad_size']:
            grad = grad[:, outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        raise NotImplemented

    if inputs.requires_grad:
        inputs.grad += grad


def ChannelAvgpoolBackward(outputs):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        dx_reshaped = np.zeros_like(outputs.cache['x_reshaped'])
        out_newaxis = outputs.data[:, :, np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        dout_newaxis = outputs.grad[:, :, np.newaxis, :, :]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.mean(mask, axis=2, keepdims=True)
        grad = dx_reshaped.reshape(inputs.data.shape)
        if outputs.cache['pad_size']:
            grad = grad[:, outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        raise NotImplemented

    if inputs.requires_grad:
        inputs.grad += grad


def Avgpool2DBackward(outputs):
    inputs, = outputs.in_bounds
    grad = outputs.grad.transpose(0, 2, 3, 1)
    dmean = np.repeat(grad.flatten(), outputs.cache['pool_argmean'].size) / (
            outputs.cache['kernel_size'] * outputs.cache['kernel_size'])
    dmean = dmean.reshape(grad.shape + (outputs.cache['kernel_size'] * outputs.cache['kernel_size'],))
    dcol = dmean.reshape(reduce(lambda x, y: x * y, dmean.shape[:3]), -1)
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


def Batchnorm2DBackward(outputs):
    inputs,  = outputs.in_bounds
    gamma, beta = outputs.get_variables()
    if inputs.requires_grad:
        grad = outputs.grad
        ndim = grad.ndim
        axis = outputs.cache['axis']
        xmu = outputs.cache['xmu']
        sqrtvar = outputs.cache['sqrtvar']
        normalized_x = outputs.cache['normalized_x']
        if not (axis == -1 or axis == ndim - 1):
            # for instance,inputs:(N,C,H,W),self.axis=1,after swapaxes,Inputs:(N,W,H,C)
            grad = np.swapaxes(grad, axis, -1)

        # (N,W,H,C) / (N,M)
        before_reshape_shape = grad.shape
        # (N*W*H,C) /(N,M)
        grad = grad.reshape(-1, inputs.data.shape[axis])

        if gamma.requires_grad:
            gamma.grad += np.sum(grad * normalized_x, axis=0)

        if beta.requires_grad:
            beta.grad += np.sum(grad, axis=0)

        N = normalized_x.shape[0]
        dnormalized_x = grad * gamma.data
        dvar = np.sum(np.power(- 1. / sqrtvar, 3) * xmu * dnormalized_x * 0.5, axis=0)
        dmean = np.sum(- dnormalized_x / sqrtvar, axis=0) - 2 * dvar * np.mean(xmu, axis=0)
        grad = dnormalized_x / sqrtvar + dvar * 2 * xmu / N + dmean / N
        grad = grad.reshape(before_reshape_shape)

        if not (axis == -1 or axis == ndim - 1):
            # for instance,outputs:(N,W,H,C),self.axis=1,after swapaxes,outputs:(N,C,H,W)
            grad = np.swapaxes(grad, axis, -1)

        inputs.grad += grad


def Layernorm2DBackward(outputs):
    inputs, = outputs.in_bounds
    gamma, beta = outputs.get_variables()
    grad = outputs.grad
    if gamma.requires_grad:
        normalized_x = inputs.cache['normalized_x']
        gamma.grad += np.sum(grad * normalized_x, axis=0)
    if beta.requires_grad:
        beta.grad += np.sum(grad, axis=0)

    if inputs.requires_grad:
        xmu = outputs.cache['xmu']
        sqrtvar = outputs.cache['sqrtvar']
        normalized_x = outputs.cache['normalized_x']
        shape_field = outputs.cache['shape_field']
        std_inv = 1. / sqrtvar
        N = reduce(lambda x, y: x * y, normalized_x.shape[1:])

        dnormalized_x = grad * gamma.data
        dvar = (-0.5) * np.sum(dnormalized_x * xmu, axis=shape_field, keepdims=True) * (
                    std_inv ** 3)  # (m,1)=(m,c,h,w)*(m,c,h,w)*(m,1)

        dmean = (-1.0) * np.sum(dnormalized_x * std_inv, axis=shape_field, keepdims=True) - 2.0 * dvar * np.mean(xmu,
                                                                                                                 axis=shape_field, keepdims=True)

        grad = dnormalized_x * std_inv + (2. / N) * dvar * xmu + (1. / N) * dmean

        inputs.grad += grad


def Groupnorm2DBackward(outputs):
    inputs, = outputs.in_bounds
    gamma, beta = outputs.get_variables()
    grad = outputs.grad
    if gamma.requires_grad:
        x_norm = outputs.cache['x_norm']
        gamma.grad += np.sum(grad * x_norm, axis=(0, 2, 3), keepdims=True)
    if beta.requires_grad:
        beta.grad += np.sum(grad, axis=(0, 2, 3), keepdims=True)

    if inputs.requires_grad:
        n, c, h, w = grad.shape
        groups = outputs.cache['groups']
        sqrtvar = outputs.cache['sqrtvar']
        xgmu = outputs.cache['xgmu']
        std_inv = 1. / sqrtvar
        # dx_group_norm
        dx_norm = grad * gamma.data  # (N,C,H,W)
        dx_group_norm = np.reshape(dx_norm, (n, groups, c // groups, h, w))
        # dvar
        dvar = -0.5 * (std_inv ** 3) * np.sum(dx_group_norm * xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean
        N_GROUP = c // groups * h * w
        dmean1 = np.sum(dx_group_norm * -std_inv, axis=(2, 3, 4), keepdims=True)
        dmean2_var = dvar * -2.0 / N_GROUP * np.sum(xgmu, axis=(2, 3, 4), keepdims=True)
        dmean = dmean1 + dmean2_var
        # dx_group
        dx_group1 = dx_group_norm * std_inv
        dx_group_var = dvar * 2.0 / N_GROUP * xgmu
        dx_group_mean = dmean * 1.0 / N_GROUP
        dx_group = dx_group1 + dx_group_var + dx_group_mean
        # dx
        grad = np.reshape(dx_group, (n, c, h, w))
        inputs.grad += grad


def ReshapeBackward(outputs):
    if outputs.cache['inplace']:
        outputs.grad = np.reshape(outputs.grad, outputs.cache['input_shape'])
    else:
        inputs, = outputs.in_bounds
        inputs.grad += np.reshape(outputs.grad, inputs.shape)


def MeanSquaredBackward(outputs):
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


def MeanAbsoluteBackward(outputs):
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
    to_sum_dim = reduce(lambda x, y: x * y, y_pred.data.shape[:-1])
    # to_sum_dim = np.prod(y_pred.data.shape[:-1])
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


def CopySlicesBackward(outputs):
    for pos, slices in outputs.cache.items():
        if slices.requires_grad:
            slices.grad += outputs.grad[pos]


def SlicesBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        pos = outputs.cache['pos']
        inputs.grad[pos] += outputs.grad
