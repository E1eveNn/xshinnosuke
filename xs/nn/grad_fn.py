from utils.common import *
from core.base import Tensor, Parameter
import nn.td_functional


def AddBackward(outputs: Tensor):
    for in_bound in outputs.in_bounds:
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                nn.td_functional.add(in_bound.grad.eval, np.sum(outputs.grad.eval, axis=not_equal_axis[0]), in_bound.grad.eval)
                # in_bound.grad.eval += np.sum(outputs.grad.eval, axis=not_equal_axis[0])
            else:
                nn.td_functional.add(in_bound.grad.eval, outputs.grad.eval,
                                     in_bound.grad.eval)
                # in_bound.grad.eval += outputs.grad.eval


def SubBackward(outputs: Tensor):
    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                # grad = np.sum(outputs.grad.eval, axis=not_equal_axis[0])
                grad = np.sum(outputs.grad.eval, axis=not_equal_axis[0])
            else:
                # grad = outputs.grad.eval
                grad = outputs.grad.eval
            if i != 0:
                grad = -grad

            # in_bound.grad.eval += grad
            nn.td_functional.add(in_bound.grad.eval, grad, in_bound.grad.eval)


def MultiplyBackward(outputs: Tensor):
    length = len(outputs.in_bounds)
    product_except_self_list = [0] * length
    product_except_self_list[0] = 1
    for i in range(1, length):
        # product_except_self_list[i] = product_except_self_list[i - 1] * outputs.in_bounds[i - 1].eval
        product_except_self_list[i] = product_except_self_list[i - 1] * outputs.in_bounds[i - 1].eval
    right = 1
    for i in reversed(range(length)):
        product_except_self_list[i] *= right
        # right = right * outputs.in_bounds[i].eval
        right = right * outputs.in_bounds[i].eval

    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                # in_bound.grad.eval += np.sum(outputs.grad.eval * product_except_self_list[i], axis=not_equal_axis[0])
                nn.td_functional.add(in_bound.grad.eval, np.sum(outputs.grad.eval * product_except_self_list[i], axis=not_equal_axis[0]), in_bound.grad.eval)
            else:
                # in_bound.grad.eval += outputs.grad.eval * product_except_self_list[i]
                nn.td_functional.add(in_bound.grad.eval, outputs.grad.eval * product_except_self_list[i], in_bound.grad.eval)
                

def MMBackward(outputs):
    # z = x * y, dloss/dx = dloss/dz * dz/dx = dloss/dz * y
    x, y = outputs.in_bounds
    if x.requires_grad:
        # np.add(x.grad.eval, np.dot(outputs.grad.eval, y.eval.T), out=x.grad.eval)
        nn.td_functional.add(x.grad.eval, np.dot(outputs.grad.eval, y.eval.T), x.grad.eval)
        # x.grad.eval += np.dot(outputs.grad.eval, y.eval.T)
    if y.requires_grad:
        # np.add(y.grad.eval, np.dot(x.eval.T, outputs.grad.eval), out=y.grad.eval)
        nn.td_functional.add(y.grad.eval, np.dot(x.eval.T, outputs.grad.eval), y.grad.eval)
        # y.grad.eval += np.dot(x.eval.T, outputs.grad.eval)


def AddmmBackward(outputs: Tensor):
    x1, x2, b = outputs.in_bounds
    if b.requires_grad:
        not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != b.shape[k]]
        if not_equal_axis:
            # b.grad.eval += np.sum(outputs.grad.eval, axis=not_equal_axis[0])
            b.grad.eval += np.sum(outputs.grad.eval, axis=not_equal_axis[0])
        else:
            # b.grad.eval += outputs.grad.eval
            b.grad.eval += outputs.grad.eval
    if x1.requires_grad:
        x1.grad.eval += np.dot(outputs.grad.eval, x2.eval.T)
    if x2.requires_grad:
        x2.grad.eval += np.dot(x1.eval.T, outputs.grad.eval)


def DivBackward(outputs):
    pass


def MaxBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        grad = outputs.grad.eval
        if outputs.eval.ndim < inputs.eval.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = np.expand_dims(grad, axis)
        mask = (inputs.eval == np.max(inputs.eval))
        inputs.grad.eval += mask * grad


def MaximumBackward(outputs):
    pass


def SumBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad.eval += np.multiply(np.ones_like(inputs.eval), nn.td_functional.expand_as(outputs.grad.eval, inputs.eval))


def MeanBackward(outputs):
    inputs, = outputs.in_bounds
    grad = outputs.grad.eval
    if inputs.requires_grad:
        mean_nums = inputs.eval.size / outputs.eval.size
        if outputs.eval.ndim < inputs.eval.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = np.expand_dims(grad, axis)
        inputs.grad.eval += np.ones_like(inputs.eval) * grad / mean_nums


def TransposeBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad.eval += outputs.grad.eval.T


def ReLUBackward(outputs: Tensor):
    if 'inplace' in outputs.cache and outputs.cache['inplace']:
        mask = outputs.cache['mask']
        outputs.grad.eval[mask] = 0
        outputs.grad_fn = outputs.cache['grad_fn'].pop()
        outputs.grad_fn(outputs)
    else:
        inputs, = outputs.in_bounds
        if inputs.requires_grad:
            grad = outputs.grad.eval.copy()
            grad[inputs.eval < 0] = 0
            np.add(inputs.grad.eval, grad, out=inputs.grad.eval)



def SigmoidBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        np.add(inputs.grad.eval, np.multiply(np.multiply(outputs.grad.eval, outputs.eval), np.subtract(1, outputs.eval)), out=inputs.grad.eval)


def TanhBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        np.add(inputs.grad.eval, np.multiply(outputs.grad.eval, np.subtract(1, np.square(outputs.eval))), out=inputs.grad.eval)


def FlattenBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        np.add(inputs.grad.eval, outputs.grad.eval.reshape(inputs.shape), out=inputs.grad.eval)


def EmbeddingBackward(outputs: Tensor):
    inputs, weight = outputs.in_bounds
    if weight.requires_grad:
        mask = inputs.eval.astype(np.int)
        np.add(weight.grad.eval[mask], outputs.grad.eval, out=weight.grad.eval[mask])


def Conv2DBackward(outputs: Tensor):
    inputs, weight, bias = outputs.in_bounds
    n, in_channels, h, w = inputs.eval.shape
    out_channels, _, kernel_h, kernel_w = weight.eval.shape
    _, _, out_h, out_w = outputs.grad.eval.shape
    grad_reshaped = outputs.grad.eval.transpose(1, 0, 2, 3).reshape(out_channels, -1)
    if weight.requires_grad:
        col = outputs.cache['col']
        np.add(weight.grad.eval, np.dot(grad_reshaped, col).reshape(out_channels, in_channels, kernel_h, kernel_w), out=weight.grad.eval)

    if bias is not None and bias.requires_grad:
        np.add(bias.grad.eval, np.sum(outputs.grad.eval, axis=(0, 2, 3)), out=bias.grad.eval)

    dcol = grad_reshaped.T.dot(weight.eval.reshape(out_channels, -1))

    if inputs.requires_grad:
        np.add(inputs.grad.eval, col2im(inputs.shape, outputs.cache['padding'], kernel_h, kernel_w,
                                             outputs.cache['stride'], dcol), out=inputs.grad.eval)


def Maxpool2DBackward(outputs: Tensor):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        if mode == 'reshape':
            dx_reshaped = outputs.cache['x_reshaped']
            out_newaxis = outputs.eval[:, :, :, np.newaxis, :, np.newaxis]
            mask = (dx_reshaped == out_newaxis)
            dx_reshaped[:] = 0
            dout_newaxis = outputs.grad.eval[:, :, :, np.newaxis, :, np.newaxis]
            dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
            dx_reshaped[mask] = dout_broadcast[mask]
            np.divide(dx_reshaped, np.sum(mask, axis=(3, 5), keepdims=True), out=dx_reshaped)
            grad = dx_reshaped.reshape(inputs.shape)
            pad_size = outputs.cache['padding']
            if pad_size != 0:
                grad = grad[:, :, pad_size: -pad_size, pad_size: -pad_size]
        else:
            kernel_size = outputs.cache['kernel_size']
            pool_argmax = outputs.cache['pool_argmax']
            grad = outputs.grad.eval.transpose(0, 2, 3, 1)
            dmax = np.zeros((grad.size, kernel_size * kernel_size))
            dmax[np.arange(pool_argmax.size), pool_argmax.flatten()] = grad.flatten()
            dmax = dmax.reshape(grad.shape + (kernel_size * kernel_size,))

            dcol = dmax.reshape((np.prod(np.asarray(dmax.shape[:3])), -1))
            grad = col2im(inputs.shape, outputs.cache['padding'], kernel_size,
                          kernel_size, outputs.cache['stride'], dcol)

        np.add(inputs.grad.eval, grad, out=inputs.grad.eval)



def ChannelMaxpoolBackward(outputs: Tensor):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        # （n, c // kernel_size, kernel_size, h, w）
        dx_reshaped = np.zeros_like(outputs.cache['x_reshaped'])
        # （n, c // kernel_size, 1, h, w）
        out_newaxis = outputs.eval[:, :, np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        # （n, c // kernel_size, 1, h, w）
        dout_newaxis = outputs.grad.eval.eval[:, :, np.newaxis, :, :]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=2, keepdims=True)
        grad = dx_reshaped.reshape(inputs.eval.shape)
        if outputs.cache['pad_size']:
            grad = grad[:, outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        raise NotImplemented

    if inputs.requires_grad:
        inputs.grad.eval.eval += grad


def ChannelAvgpoolBackward(outputs: Tensor):
    mode = outputs.get_cache('mode')
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        dx_reshaped = np.zeros_like(outputs.cache['x_reshaped'])
        out_newaxis = outputs.eval[:, :, np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        dout_newaxis = outputs.grad.eval.eval[:, :, np.newaxis, :, :]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.mean(mask, axis=2, keepdims=True)
        grad = dx_reshaped.reshape(inputs.eval.shape)
        if outputs.cache['pad_size']:
            grad = grad[:, outputs.cache['pad_size']: -outputs.cache['pad_size']]
    else:
        raise NotImplemented

    if inputs.requires_grad:
        inputs.grad.eval.eval += grad


def Avgpool2DBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    padding = outputs.cache['padding']
    if inputs.requires_grad:
        if outputs.cache['mode'] == 'reshape':
            reshaped_shape = outputs.cache['reshaped_shape']
            dout_newaxis = np.divide(outputs.grad.eval[:, :, :, np.newaxis, :, np.newaxis], outputs.cache['stride'] * outputs.cache['stride'])
            dout_broadcast = np.broadcast_to(dout_newaxis, reshaped_shape)
            grad = dout_broadcast.reshape(inputs.shape)
        else:
            grad = outputs.grad.eval.transpose(0, 2, 3, 1)
            pool_argmean = outputs.cache['pool_argmean']
            kernel_size = outputs.cache['kernel_size']

            stride = outputs.cache['stride']
            dmean = np.repeat(grad.flatten(), pool_argmean.size)
            np.divide(dmean, kernel_size * kernel_size, out=dmean)
            dmean = dmean.reshape(grad.shape + (kernel_size * kernel_size,))
            dmean = np.reshape(dmean, (np.prod(np.asarray(dmean.shape[:3])), -1))

            grad = col2im(inputs.shape, padding, kernel_size,
                          kernel_size, stride, dmean)
        if padding != 0:
            grad = grad[:, :, padding: -padding, padding: -padding]

        inputs.grad.eval += grad


def Pad2DBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        pad_size = outputs.cache['padding']
        np.add(inputs.grad.eval, outputs.grad.eval[:, :, pad_size[0]: -pad_size[0], pad_size[1]: -pad_size[1]], out=inputs.grad.eval)


def Dropout2DBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        mask = outputs.cache['mask']
        np.multiply(mask, outputs.cache['keep_prob'], out=mask)
        np.multiply(mask, outputs.grad.eval, out=mask)
        np.add(inputs.grad.eval, mask, out=inputs.grad.eval)


# def Batchnorm2DBackward(outputs: Tensor):
#     inputs, gamma, beta = outputs.in_bounds
#     if inputs.requires_grad:
#         grad = outputs.grad.eval
#         ndim = grad.ndim
#         axis = outputs.cache['axis']
#         xmu = outputs.cache['xmu']
#         sqrtvar = outputs.cache['sqrtvar']
#         normalized_x = outputs.cache['normalized_x']
#         if not (axis == -1 or axis == ndim - 1):
#             # for instance,inputs:(N,C,H,W),self.axis=1,after swapaxes,Inputs:(N,W,H,C)
#             grad = np.swapaxes(grad, axis, -1)
#
#         # (N,W,H,C) / (N,M)
#         before_reshape_shape = grad.shape
#         # (N*W*H,C) /(N,M)
#         grad = grad.reshape(-1, inputs.eval.shape[axis])
#
#         if gamma.requires_grad:
#             np.add(gamma.grad.eval, np.sum(grad * normalized_x, axis=0),out=gamma.grad.eval)
#
#         if beta.requires_grad:
#             np.add(beta.grad.eval, np.sum(grad, axis=0), out=beta.grad.eval)
#
#         N = normalized_x.shape[0]
#         grad = np.multiply(grad, gamma.eval, out=grad)
#
#         dvar = np.sum(np.multiply(0.5, np.multiply(grad, np.multiply(xmu, np.power(np.divide(-1., sqrtvar), 3)))), axis=0)
#         dmean = np.sum(np.divide(-grad, sqrtvar), axis=0) - np.multiply(np.multiply(2, dvar), np.mean(xmu, axis=0))
#         np.divide(grad, sqrtvar, out=grad)
#
#         np.multiply(2, xmu, out=xmu)
#         np.multiply(dvar, xmu, out=xmu)
#         np.divide(xmu, N, out=xmu)
#
#         np.divide(dmean, N, out=dmean)
#
#         np.add(grad, xmu, out=grad)
#
#         np.add(grad, dmean, out=grad)
#
#         # dvar = np.sum(np.power(- 1. / sqrtvar, 3) * xmu * dnormalized_x * 0.5, axis=0)
#         # dmean = np.sum(- dnormalized_x / sqrtvar, axis=0) - 2 * dvar * np.mean(xmu, axis=0)
#         # grad = dnormalized_x / sqrtvar + dvar * 2 * xmu / N + dmean / N
#         grad = grad.reshape(before_reshape_shape)
#
#         if not (axis == -1 or axis == ndim - 1):
#             # for instance,outputs:(N,W,H,C),self.axis=1,after swapaxes,outputs:(N,C,H,W)
#             grad = np.swapaxes(grad, axis, -1)
#
#         inputs.grad.eval += grad
def BatchNormBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    grad = outputs.grad.eval
    axis_field = outputs.cache['axis_field']
    sqrtvar = outputs.cache['sqrtvar']
    normalized_x = outputs.cache['normalized_x']
    if gamma.requires_grad:
        np.add(gamma.grad.eval, np.sum(np.multiply(grad, normalized_x), axis=axis_field), out=gamma.grad.eval)

    if beta.requires_grad:
        np.add(beta.grad.eval, np.sum(grad, axis=axis_field), out=beta.grad.eval)
    # N = normalized_x.shape[0]
    N = np.prod([normalized_x.shape[axis] for axis in axis_field])
    if inputs.requires_grad:
        # dx_ = np.matmul(np.ones((N, 1)), gamma.reshape((1, -1))) * dout
        # dx = N * dx_ - np.sum(dx_, axis=0) - x_ * np.sum(dx_ * x_, axis=0)
        # dx *= (1.0 / N) / np.sqrt(var_plus_eps)

        # grad = np.multiply(grad, nn.td_functional.expand_as(gamma.eval, grad))
        # np.multiply(N, sqrtvar, out=sqrtvar)
        # np.divide(1., sqrtvar, out=sqrtvar)
        # dx = 1 / (N * np.sqrt(var_eps)) * (dx_hat * N - np.sum(dx_hat, axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0))
        # np.multiply(normalized_x, np.sum(np.multiply(grad, normalized_x), axis=axis_field, keepdims=True), out=normalized_x)
        # np.subtract(np.subtract(np.multiply(grad, N), np.sum(grad, axis=axis_field, keepdims=True)), normalized_x, out=grad)
        # np.multiply(sqrtvar, grad, out=grad)
        dx_ = np.multiply(grad, nn.td_functional.expand_as(gamma.eval, grad))
        np.multiply(N, sqrtvar, out=sqrtvar)
        np.divide(1., sqrtvar, out=sqrtvar)

        np.multiply(normalized_x, np.sum(np.multiply(dx_, normalized_x), axis=axis_field, keepdims=True), out=normalized_x)
        np.subtract(np.subtract(np.multiply(dx_, N), np.sum(dx_, axis=axis_field, keepdims=True)), normalized_x, out=dx_)
        np.multiply(sqrtvar, dx_, out=dx_)
        np.add(inputs.grad.eval, dx_, out=inputs.grad.eval)


def Layernorm2DBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    dnormalized_x = outputs.grad.eval
    if gamma.requires_grad:
        normalized_x = inputs.cache['normalized_x']
        np.add(gamma.grad.eval, np.sum(dnormalized_x * normalized_x, axis=0), out=gamma.grad.eval)
    if beta.requires_grad:
        np.add(beta.grad.eval, np.sum(dnormalized_x, axis=0), out=beta.grad.eval)

    if inputs.requires_grad:
        xmu = outputs.cache['xmu']
        std_inv = outputs.cache['sqrtvar']
        normalized_x = outputs.cache['normalized_x']
        shape_field = outputs.cache['shape_field']

        np.divide(1., std_inv, out=std_inv)
        N = np.prod(np.asarray(normalized_x.shape[1:]))
        np.multiply(dnormalized_x, gamma.eval, out=dnormalized_x)

        dvar = np.multiply(np.multiply(-0.5, np.sum(np.multiply(dnormalized_x, xmu), axis=shape_field, keepdims=True)), np.power(std_inv, 3))
        # dvar = (-0.5) * np.sum(dnormalized_x * xmu, axis=shape_field, keepdims=True) * (
                    # std_inv ** 3)  # (m,1)=(m,c,h,w)*(m,c,h,w)*(m,1)
        dmean = np.multiply(-1., np.sum(np.multiply(dnormalized_x, std_inv), axis=shape_field, keepdims=True))
        np.subtract(dmean, np.multiply(-2, np.multiply(dvar, np.mean(xmu, axis=shape_field, keepdims=True))), out=dmean)
        # dmean = (-1.0) * np.sum(dnormalized_x * std_inv, axis=shape_field, keepdims=True) - 2.0 * dvar * np.mean(xmu,
        #                                                                                                          axis=shape_field, keepdims=True)
        np.multiply(dnormalized_x, std_inv, out=dnormalized_x)
        np.multiply(dvar, xmu, out=dvar)
        np.multiply((2. / N), dvar, out=dvar)
        np.multiply((1. / N), dmean, out=dmean)
        np.add(dnormalized_x, dvar, out=dnormalized_x)
        np.add(dnormalized_x, dmean, out=dnormalized_x)
        # grad = dnormalized_x * std_inv + (2. / N) * dvar * xmu + (1. / N) * dmean
        np.add(inputs.grad.eval, dnormalized_x, out=inputs.grad.eval)


def Groupnorm2DBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    dx_norm = outputs.grad.eval
    if gamma.requires_grad:
        x_norm = outputs.cache['x_norm']
        np.add(gamma.grad.eval, np.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True), out=gamma.grad.eval)
    if beta.requires_grad:
        np.add(beta.grad.eval, np.sum(dx_norm, axis=(0, 2, 3), keepdims=True), out=beta.grad.eval)

    if inputs.requires_grad:
        n, c, h, w = dx_norm.shape
        groups = outputs.cache['groups']
        std_inv = outputs.cache['sqrtvar']
        xgmu = outputs.cache['xgmu']
        np.divide(1., std_inv, out=std_inv)
        # dx_group_norm
        np.multiply(dx_norm, gamma.eval, out=dx_norm) # (N,C,H,W)
        dx_group_norm = np.reshape(dx_norm, (n, groups, c // groups, h, w))
        # dvar
        dvar = np.multiply(-0.5, np.multiply(np.power(std_inv, 3), np.sum(np.multiply(dx_group_norm, xgmu), axis=(2, 3, 4), keepdims=True)))
        # dvar = -0.5 * (std_inv ** 3) * np.sum(dx_group_norm * xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean
        N_GROUP = c // groups * h * w
        np.multiply(dx_group_norm, std_inv, out=dx_group_norm)

        dmean = np.sum(-dx_group_norm, axis=(2, 3, 4), keepdims=True)
        # dmean1 = np.sum(dx_group_norm * -std_inv, axis=(2, 3, 4), keepdims=True)
        dmean2_var = np.multiply(np.divide(np.multiply(dvar, -2.0), N_GROUP), np.sum(xgmu, axis=(2, 3, 4), keepdims=True))
        np.add(dmean, dmean2_var, out=dmean)
        # dmean2_var = dvar * -2.0 / N_GROUP * np.sum(xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean = dmean1 + dmean2_var
        # dx_group
        np.multiply(dvar, 2.0, out=dvar)
        np.divide(dvar, N_GROUP, out=dvar)
        np.multiply(dvar, xgmu, out=dvar)
        #dx_group_var = dvar * 2.0 / N_GROUP * xgmu
        np.multiply(dmean, 1.0, out=dmean)
        np.divide(dmean, N_GROUP, out=dmean)
        # dx_group_mean = dmean * 1.0 / N_GROUP
        np.add(dx_group_norm, dvar, out=dx_group_norm)
        np.add(dx_group_norm, dmean, out=dx_group_norm)
        # dx_group = dx_group1 + dx_group_var + dx_group_mean
        # dx
        grad = np.reshape(dx_group_norm, (n, c, h, w))
        np.add(inputs.grad.eval, grad, out=inputs.grad.eval)


def ViewBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        np.add(inputs.grad.eval, np.reshape(outputs.grad.eval, inputs.shape), out=inputs.grad.eval)


def MSELossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    gradients = np.divide(np.multiply(np.subtract(y_pred.eval, y_true.eval), outputs.grad.eval),
                          np.prod(y_pred.shape))
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(-gradients)
        else:
            np.add(y_true.grad.eval, -gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def MAEBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    pos = np.where((y_pred.eval - y_true.eval) < 0)
    mask = np.ones_like(y_pred.eval)
    mask[pos] = -1
    np.divide(mask, y_pred.shape[0], out=mask)

    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(mask)
        else:
            np.add(y_true.grad.eval, mask, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(mask)
        else:
            np.add(y_pred.grad.eval, mask, out=y_pred.grad.eval)


def BCELossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    avg = np.prod(y_pred.shape)
    gradients = np.divide(np.subtract(np.divide(np.subtract(1, y_true.eval), np.subtract(1, y_pred.eval)), np.divide(y_true.eval, y_pred.eval)), avg)

    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(gradients)
        else:
            np.add(y_true.grad.eval, gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def BCEWithLogitsLossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    avg = np.prod(y_pred.shape)
    logits = nn.td_functional.sigmoid(y_pred.eval)
    gradients = np.divide(np.subtract(np.divide(np.subtract(1, y_true.eval), np.subtract(1, logits)), np.divide(y_true.eval, logits)), avg)

    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(gradients)
        else:
            np.add(y_true.grad.eval, gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        np.multiply(np.multiply(gradients, logits), np.subtract(1, logits), out=gradients)
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def SparseCrossEntropyBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    # # before softmax
    before_softmax_y_pred = y_pred.in_bounds[0]
    avg = np.prod(np.asarray(y_pred.eval.shape[:-1]))
    gradients = (y_pred.eval - y_true.eval) / avg
    if y_true.requires_grad:
        if y_true.grad.eval.eval is None:
            y_true.grad.eval = gradients
        else:
            y_true.grad.eval += gradients

    if before_softmax_y_pred.requires_grad:
        if before_softmax_y_pred.grad.eval is None:
            before_softmax_y_pred.grad.eval = gradients
        else:
            before_softmax_y_pred.grad.eval += gradients


def NllLossBackward(outputs: Tensor):
    log_softmax_prob, target = outputs.in_bounds



def CrossEntropyBackward(outputs: Tensor):
    before_softmax_y_pred, y_true = outputs.in_bounds
    y_pred = before_softmax_y_pred.cache['softmax']
    # # before softmax
    # before_softmax_y_pred,  = y_pred.in_bounds
    to_sum_dim = np.prod(np.asarray(before_softmax_y_pred.shape[:-1]))
    n = before_softmax_y_pred.eval.shape[0]
    probs = y_pred.eval.reshape(-1,  before_softmax_y_pred.shape[-1])
    y_flat = y_true.eval.reshape(to_sum_dim)
    probs[np.arange(to_sum_dim), y_flat] -= 1
    gradients = np.divide(probs.reshape(before_softmax_y_pred.shape), n)
    gradients = np.multiply(gradients, outputs.grad.eval, out=gradients)
    if before_softmax_y_pred.requires_grad:
        if before_softmax_y_pred.grad is None:
            before_softmax_y_pred.grad = Tensor(gradients)
        else:
            np.add(before_softmax_y_pred.grad.eval, gradients, out=before_softmax_y_pred.grad.eval)


def CopySlicesBackward(outputs: Tensor):
    for pos, slices in outputs.cache.items():
        if slices.requires_grad:
            slices.grad.eval += outputs.grad.eval[pos]


def SlicesBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        pos = outputs.cache['pos']
        inputs.grad.eval[pos] += outputs.grad.eval


def LstmBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    weight, bias = outputs.parameters()
    units = outputs.cache['units']
    time_steps = outputs.cache['time_steps']
    recurrent_activations = outputs.cache['recurrent_activations']
    activations = outputs.cache['activations']
    prev_a = outputs.cache['prev_a']
    c = outputs.cache['c']
    tao_f = outputs.cache['tao_f']
    tao_u = outputs.cache['tao_u']
    tao_o = outputs.cache['tao_o']
    c_tilde = outputs.cache['c_tilde']
    return_sequences = outputs.cache['return_sequences']

    da_next = np.zeros_like(prev_a.eval[:, 0, :])
    dc_next = np.zeros_like(c.eval[:, 0, :])
    if inputs.requires_grad:
        grad = np.zeros_like(inputs.eval)
    if return_sequences:
        for t in reversed(range(time_steps)):
            da = outputs.grad.eval.eval[:, t, :] + da_next
            dtao_o = da * np.tanh(c.eval[:, t + 1, :])
            do = recurrent_activations[3 * (t + 1) - 1].backward(dtao_o)
            dc = dc_next
            dc += da * tao_o.eval[:, t, :] * (1 - np.square(np.tanh(c.eval[:, t + 1, :])))
            dc_tilde = dc * tao_u.eval[:, t, :]
            dc_tilde_before_act = activations[t].backward(dc_tilde)
            dtao_u = dc * c_tilde.eval[:, t, :]
            du = recurrent_activations[3 * (t + 1) - 2].backward(dtao_u)
            dtao_f = dc * c.eval[:, t, :]
            df = recurrent_activations[3 * (t + 1) - 3].backward(dtao_f)
            dgrad = np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
            if weight.requires_grad:
                weight.grad.eval.eval += np.dot(inputs.eval[:, t, :].T, dgrad)
            if bias.requires_grad:
                bias.grad.eval.eval += np.sum(dgrad, axis=0, keepdims=True)

            dz = dgrad.dot(weight.eval.T)

            da_next = dz[:, :units]
            dc_next = dc * tao_f.eval[:, t, :]
            if inputs.requires_grad:
                grad[:, t, :] = dz[:, units:]
    else:
        da = outputs.grad.eval.eval + da_next
        for t in reversed(range(time_steps)):
            dtao_o = da * np.tanh(c.eval[:, t + 1, :])
            recurrent_activations[3 * (t + 1) - 1].eval.grad.eval.eval = dtao_o
            recurrent_activations[3 * (t + 1) - 1].backward()
            do = recurrent_activations[3 * (t + 1) - 1].input_data.grad.eval.eval

            dc = dc_next
            dc += da * tao_o.eval[:, t, :] * (1 - np.square(np.tanh(c.eval[:, t + 1, :])))

            dc_tilde = dc * tao_u.eval[:, t, :]
            activations[t].eval.grad.eval.eval = dc_tilde
            activations[t].backward()
            dc_tilde_before_act = activations[t].input_data.grad.eval.eval

            dtao_u = dc * c_tilde.eval[:, t, :]
            recurrent_activations[3 * (t + 1) - 2].eval.grad.eval.eval = dtao_u
            recurrent_activations[3 * (t + 1) - 2].backward()
            du = recurrent_activations[3 * (t + 1) - 2].input_data.grad.eval.eval

            dtao_f = dc * c.eval[:, t, :]
            recurrent_activations[3 * (t + 1) - 3].eval.grad.eval.eval = dtao_f
            recurrent_activations[3 * (t + 1) - 3].backward()
            df = recurrent_activations[3 * (t + 1) - 3].input_data.grad.eval.eval

            dgrad = np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
            if weight.requires_grad:
                zt = np.concatenate((prev_a.eval[:, t - 1, :], inputs.eval[:, t - 1, :]), axis=1)
                weight.grad.eval.eval += np.dot(zt.T, dgrad)
            if bias.requires_grad:
                bias.grad.eval.eval += np.sum(dgrad, axis=0, keepdims=True)

            dz = dgrad.dot(weight.eval.T)

            da = dz[:, :units]
            dc_next = dc * tao_f.eval[:, t, :]
            if inputs.requires_grad:
                grad[:, t, :] = dz[:, units:]
    if inputs.requires_grad:
        inputs.grad.eval.eval += grad