from utils.common import *
from core.base import Tensor, Parameter
import nn.td_functional


def AddBackward(outputs: Tensor):
    for in_bound in outputs.in_bounds:
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                nn.td_functional.add(in_bound.grad.eval, GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0]), in_bound.grad.eval)
                # in_bound.grad.eval += GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0])
            else:
                nn.td_functional.add(in_bound.grad.eval, outputs.grad.eval,
                                     in_bound.grad.eval)
                # in_bound.grad.eval += outputs.grad.eval


def SubBackward(outputs: Tensor):
    for i, in_bound in enumerate(outputs.in_bounds):
        if in_bound.requires_grad:
            not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != in_bound.shape[k]]
            if not_equal_axis:
                # grad = GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0])
                grad = GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0])
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
                # in_bound.grad.eval += GLOBAL.np.sum(outputs.grad.eval * product_except_self_list[i], axis=not_equal_axis[0])
                nn.td_functional.add(in_bound.grad.eval, GLOBAL.np.sum(outputs.grad.eval * product_except_self_list[i], axis=not_equal_axis[0]), in_bound.grad.eval)
            else:
                # in_bound.grad.eval += outputs.grad.eval * product_except_self_list[i]
                nn.td_functional.add(in_bound.grad.eval, outputs.grad.eval * product_except_self_list[i], in_bound.grad.eval)


def PowBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        exp = outputs.cache['exp']
        if exp == 1:
            nn.td_functional.add(inputs.grad.eval, outputs.grad.eval, inputs.grad.eval)
        else:
            gradients = GLOBAL.np.power(inputs.eval, exp - 1)
            GLOBAL.np.multiply(gradients, exp, out=gradients)
            GLOBAL.np.multiply(outputs.grad.eval, gradients, out=gradients)
            nn.td_functional.add(inputs.grad.eval, gradients, inputs.grad.eval)
                

def MMBackward(outputs):
    # z = x * y, dloss/dx = dloss/dz * dz/dx = dloss/dz * y
    x, y = outputs.in_bounds
    if x.requires_grad:
        # GLOBAL.np.add(x.grad.eval, GLOBAL.np.dot(outputs.grad.eval, y.eval.T), out=x.grad.eval)
        nn.td_functional.add(x.grad.eval, GLOBAL.np.dot(outputs.grad.eval, y.eval.T), x.grad.eval)
        # x.grad.eval += GLOBAL.np.dot(outputs.grad.eval, y.eval.T)
    if y.requires_grad:
        # GLOBAL.np.add(y.grad.eval, GLOBAL.np.dot(x.eval.T, outputs.grad.eval), out=y.grad.eval)
        nn.td_functional.add(y.grad.eval, GLOBAL.np.dot(x.eval.T, outputs.grad.eval), y.grad.eval)
        # y.grad.eval += GLOBAL.np.dot(x.eval.T, outputs.grad.eval)


def AddmmBackward(outputs: Tensor):
    x1, x2, b = outputs.in_bounds
    if b.requires_grad:
        not_equal_axis = [k for k, v in enumerate(outputs.shape) if v != b.shape[k]]
        if not_equal_axis:
            # b.grad.eval += GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0])
            b.grad.eval += GLOBAL.np.sum(outputs.grad.eval, axis=not_equal_axis[0])
        else:
            # b.grad.eval += outputs.grad.eval
            b.grad.eval += outputs.grad.eval
    if x1.requires_grad:
        x1.grad.eval += GLOBAL.np.dot(outputs.grad.eval, x2.eval.T)
    if x2.requires_grad:
        x2.grad.eval += GLOBAL.np.dot(x1.eval.T, outputs.grad.eval)


def DivBackward(outputs):
    pass


def MaxBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        grad = outputs.grad.eval
        if outputs.eval.ndim < inputs.eval.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = GLOBAL.np.expand_dims(grad, axis)
        mask = (inputs.eval == GLOBAL.np.max(inputs.eval))
        inputs.grad.eval += mask * grad


def MaximumBackward(outputs):
    pass


def SumBackward(outputs):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        inputs.grad.eval += GLOBAL.np.multiply(GLOBAL.np.ones_like(inputs.eval), nn.td_functional.expand_as(outputs.grad.eval, inputs.eval))


def MeanBackward(outputs):
    inputs, = outputs.in_bounds
    grad = outputs.grad.eval
    if inputs.requires_grad:
        mean_nums = inputs.eval.size / outputs.eval.size
        if outputs.eval.ndim < inputs.eval.ndim:
            axis = outputs.cache['axis']
            if axis is not None:
                grad = GLOBAL.np.expand_dims(grad, axis)
        inputs.grad.eval += GLOBAL.np.ones_like(inputs.eval) * grad / mean_nums


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
            GLOBAL.np.add(inputs.grad.eval, grad, out=inputs.grad.eval)



def SigmoidBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        GLOBAL.np.add(inputs.grad.eval, GLOBAL.np.multiply(GLOBAL.np.multiply(outputs.grad.eval, outputs.eval), GLOBAL.np.subtract(1, outputs.eval)), out=inputs.grad.eval)


def TanhBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        GLOBAL.np.add(inputs.grad.eval, GLOBAL.np.multiply(outputs.grad.eval, GLOBAL.np.subtract(1, GLOBAL.np.square(outputs.eval))), out=inputs.grad.eval)


def FlattenBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        GLOBAL.np.add(inputs.grad.eval, outputs.grad.eval.reshape(inputs.shape), out=inputs.grad.eval)


def EmbeddingBackward(outputs: Tensor):
    inputs, weight = outputs.in_bounds
    if weight.requires_grad:
        mask = inputs.eval.astype(GLOBAL.np.int)
        GLOBAL.np.add(weight.grad.eval[mask], outputs.grad.eval, out=weight.grad.eval[mask])


def Conv2DBackward(outputs: Tensor):
    inputs, weight, bias = outputs.in_bounds
    n, in_channels, h, w = inputs.eval.shape
    out_channels, _, kernel_h, kernel_w = weight.eval.shape
    _, _, out_h, out_w = outputs.grad.eval.shape
    grad_reshaped = outputs.grad.eval.transpose(1, 0, 2, 3).reshape(out_channels, -1)
    if weight.requires_grad:
        col = outputs.cache['col']
        GLOBAL.np.add(weight.grad.eval, GLOBAL.np.dot(grad_reshaped, col).reshape(out_channels, in_channels, kernel_h, kernel_w), out=weight.grad.eval)

    if bias is not None and bias.requires_grad:
        GLOBAL.np.add(bias.grad.eval, GLOBAL.np.sum(outputs.grad.eval, axis=(0, 2, 3)), out=bias.grad.eval)

    dcol = grad_reshaped.T.dot(weight.eval.reshape(out_channels, -1))

    if inputs.requires_grad:
        GLOBAL.np.add(inputs.grad.eval, col2im(inputs.shape, outputs.cache['padding'], kernel_h, kernel_w,
                                             outputs.cache['stride'], dcol), out=inputs.grad.eval)


def Maxpool2DBackward(outputs: Tensor):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        if mode == 'reshape':
            dx_reshaped = outputs.cache['x_reshaped']
            out_newaxis = outputs.eval[:, :, :, GLOBAL.np.newaxis, :, GLOBAL.np.newaxis]
            mask = (dx_reshaped == out_newaxis)
            dx_reshaped[:] = 0
            dout_newaxis = outputs.grad.eval[:, :, :, GLOBAL.np.newaxis, :, GLOBAL.np.newaxis]
            dout_broadcast, _ = GLOBAL.np.broadcast_arrays(dout_newaxis, dx_reshaped)
            dx_reshaped[mask] = dout_broadcast[mask]
            GLOBAL.np.divide(dx_reshaped, GLOBAL.np.sum(mask, axis=(3, 5), keepdims=True), out=dx_reshaped)
            grad = dx_reshaped.reshape(inputs.shape)
            pad_size = outputs.cache['padding']
            if pad_size != 0:
                grad = grad[:, :, pad_size: -pad_size, pad_size: -pad_size]
        else:
            kernel_size = outputs.cache['kernel_size']
            pool_argmax = outputs.cache['pool_argmax']
            grad = outputs.grad.eval.transpose(0, 2, 3, 1)
            dmax = GLOBAL.np.zeros((grad.size, kernel_size * kernel_size))
            dmax[GLOBAL.np.arange(pool_argmax.size), pool_argmax.flatten()] = grad.flatten()
            dmax = dmax.reshape(grad.shape + (kernel_size * kernel_size,))

            dcol = dmax.reshape((GLOBAL.np.prod(GLOBAL.np.asarray(dmax.shape[:3])).item(), -1))
            grad = col2im(inputs.shape, outputs.cache['padding'], kernel_size,
                          kernel_size, outputs.cache['stride'], dcol)

        GLOBAL.np.add(inputs.grad.eval, grad, out=inputs.grad.eval)



def ChannelMaxpoolBackward(outputs: Tensor):
    mode = outputs.cache['mode']
    inputs, = outputs.in_bounds
    if mode == 'reshape':
        # （n, c // kernel_size, kernel_size, h, w）
        dx_reshaped = GLOBAL.np.zeros_like(outputs.cache['x_reshaped'])
        # （n, c // kernel_size, 1, h, w）
        out_newaxis = outputs.eval[:, :, GLOBAL.np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        # （n, c // kernel_size, 1, h, w）
        dout_newaxis = outputs.grad.eval.eval[:, :, GLOBAL.np.newaxis, :, :]
        dout_broadcast, _ = GLOBAL.np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= GLOBAL.np.sum(mask, axis=2, keepdims=True)
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
        dx_reshaped = GLOBAL.np.zeros_like(outputs.cache['x_reshaped'])
        out_newaxis = outputs.eval[:, :, GLOBAL.np.newaxis, :, :]
        mask = (outputs.cache['x_reshaped'] == out_newaxis)
        dout_newaxis = outputs.grad.eval.eval[:, :, GLOBAL.np.newaxis, :, :]
        dout_broadcast, _ = GLOBAL.np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= GLOBAL.np.mean(mask, axis=2, keepdims=True)
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
            dout_newaxis = GLOBAL.np.divide(outputs.grad.eval[:, :, :, GLOBAL.np.newaxis, :, GLOBAL.np.newaxis], outputs.cache['stride'] * outputs.cache['stride'])
            dout_broadcast = GLOBAL.np.broadcast_to(dout_newaxis, reshaped_shape)
            grad = dout_broadcast.reshape(inputs.shape)
        else:
            grad = outputs.grad.eval.transpose(0, 2, 3, 1)
            pool_argmean = outputs.cache['pool_argmean']
            kernel_size = outputs.cache['kernel_size']

            stride = outputs.cache['stride']
            dmean = GLOBAL.np.repeat(grad.flatten(), pool_argmean.size)
            GLOBAL.np.divide(dmean, kernel_size * kernel_size, out=dmean)
            dmean = dmean.reshape(grad.shape + (kernel_size * kernel_size,))
            dmean = GLOBAL.np.reshape(dmean, (GLOBAL.np.prod(GLOBAL.np.asarray(dmean.shape[:3])).item(), -1))

            grad = col2im(inputs.shape, padding, kernel_size,
                          kernel_size, stride, dmean)
        if padding != 0:
            grad = grad[:, :, padding: -padding, padding: -padding]

        inputs.grad.eval += grad


def Pad2DBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        pad_size = outputs.cache['padding']
        GLOBAL.np.add(inputs.grad.eval, outputs.grad.eval[:, :, pad_size[0]: -pad_size[0], pad_size[1]: -pad_size[1]], out=inputs.grad.eval)


def Dropout2DBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        mask = outputs.cache['mask']
        GLOBAL.np.multiply(mask, outputs.cache['keep_prob'], out=mask)
        GLOBAL.np.multiply(mask, outputs.grad.eval, out=mask)
        GLOBAL.np.add(inputs.grad.eval, mask, out=inputs.grad.eval)


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
#             grad = GLOBAL.np.swapaxes(grad, axis, -1)
#
#         # (N,W,H,C) / (N,M)
#         before_reshape_shape = grad.shape
#         # (N*W*H,C) /(N,M)
#         grad = grad.reshape(-1, inputs.eval.shape[axis])
#
#         if gamma.requires_grad:
#             GLOBAL.np.add(gamma.grad.eval, GLOBAL.np.sum(grad * normalized_x, axis=0),out=gamma.grad.eval)
#
#         if beta.requires_grad:
#             GLOBAL.np.add(beta.grad.eval, GLOBAL.np.sum(grad, axis=0), out=beta.grad.eval)
#
#         N = normalized_x.shape[0]
#         grad = GLOBAL.np.multiply(grad, gamma.eval, out=grad)
#
#         dvar = GLOBAL.np.sum(GLOBAL.np.multiply(0.5, GLOBAL.np.multiply(grad, GLOBAL.np.multiply(xmu, GLOBAL.np.power(GLOBAL.np.divide(-1., sqrtvar), 3)))), axis=0)
#         dmean = GLOBAL.np.sum(GLOBAL.np.divide(-grad, sqrtvar), axis=0) - GLOBAL.np.multiply(GLOBAL.np.multiply(2, dvar), GLOBAL.np.mean(xmu, axis=0))
#         GLOBAL.np.divide(grad, sqrtvar, out=grad)
#
#         GLOBAL.np.multiply(2, xmu, out=xmu)
#         GLOBAL.np.multiply(dvar, xmu, out=xmu)
#         GLOBAL.np.divide(xmu, N, out=xmu)
#
#         GLOBAL.np.divide(dmean, N, out=dmean)
#
#         GLOBAL.np.add(grad, xmu, out=grad)
#
#         GLOBAL.np.add(grad, dmean, out=grad)
#
#         # dvar = GLOBAL.np.sum(GLOBAL.np.power(- 1. / sqrtvar, 3) * xmu * dnormalized_x * 0.5, axis=0)
#         # dmean = GLOBAL.np.sum(- dnormalized_x / sqrtvar, axis=0) - 2 * dvar * GLOBAL.np.mean(xmu, axis=0)
#         # grad = dnormalized_x / sqrtvar + dvar * 2 * xmu / N + dmean / N
#         grad = grad.reshape(before_reshape_shape)
#
#         if not (axis == -1 or axis == ndim - 1):
#             # for instance,outputs:(N,W,H,C),self.axis=1,after swapaxes,outputs:(N,C,H,W)
#             grad = GLOBAL.np.swapaxes(grad, axis, -1)
#
#         inputs.grad.eval += grad
def BatchNormBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    grad = outputs.grad.eval
    axis_field = outputs.cache['axis_field']
    sqrtvar = outputs.cache['sqrtvar']
    normalized_x = outputs.cache['normalized_x']
    if gamma.requires_grad:
        GLOBAL.np.add(gamma.grad.eval, GLOBAL.np.sum(GLOBAL.np.multiply(grad, normalized_x), axis=axis_field), out=gamma.grad.eval)

    if beta.requires_grad:
        GLOBAL.np.add(beta.grad.eval, GLOBAL.np.sum(grad, axis=axis_field), out=beta.grad.eval)
    # N = normalized_x.shape[0]
    N = GLOBAL.np.prod(GLOBAL.np.asarray([normalized_x.shape[axis] for axis in axis_field])).item()
    if inputs.requires_grad:
        # dx_ = GLOBAL.np.matmul(GLOBAL.np.ones((N, 1)), gamma.reshape((1, -1))) * dout
        # dx = N * dx_ - GLOBAL.np.sum(dx_, axis=0) - x_ * GLOBAL.np.sum(dx_ * x_, axis=0)
        # dx *= (1.0 / N) / GLOBAL.np.sqrt(var_plus_eps)

        # grad = GLOBAL.np.multiply(grad, nn.td_functional.expand_as(gamma.eval, grad))
        # GLOBAL.np.multiply(N, sqrtvar, out=sqrtvar)
        # GLOBAL.np.divide(1., sqrtvar, out=sqrtvar)
        # dx = 1 / (N * GLOBAL.np.sqrt(var_eps)) * (dx_hat * N - GLOBAL.np.sum(dx_hat, axis=0) - x_hat * GLOBAL.np.sum(dx_hat * x_hat, axis=0))
        # GLOBAL.np.multiply(normalized_x, GLOBAL.np.sum(GLOBAL.np.multiply(grad, normalized_x), axis=axis_field, keepdims=True), out=normalized_x)
        # GLOBAL.np.subtract(GLOBAL.np.subtract(GLOBAL.np.multiply(grad, N), GLOBAL.np.sum(grad, axis=axis_field, keepdims=True)), normalized_x, out=grad)
        # GLOBAL.np.multiply(sqrtvar, grad, out=grad)
        dx_ = GLOBAL.np.multiply(grad, nn.td_functional.expand_as(gamma.eval, grad))
        GLOBAL.np.multiply(N, sqrtvar, out=sqrtvar)
        GLOBAL.np.divide(1., sqrtvar, out=sqrtvar)

        GLOBAL.np.multiply(normalized_x, GLOBAL.np.sum(GLOBAL.np.multiply(dx_, normalized_x), axis=axis_field, keepdims=True), out=normalized_x)
        GLOBAL.np.subtract(GLOBAL.np.subtract(GLOBAL.np.multiply(dx_, N), GLOBAL.np.sum(dx_, axis=axis_field, keepdims=True)), normalized_x, out=dx_)
        GLOBAL.np.multiply(sqrtvar, dx_, out=dx_)
        GLOBAL.np.add(inputs.grad.eval, dx_, out=inputs.grad.eval)


def LayerNormBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    dnormalized_x = outputs.grad.eval
    if gamma.requires_grad:
        normalized_x = inputs.cache['normalized_x']
        GLOBAL.np.add(gamma.grad.eval, GLOBAL.np.sum(dnormalized_x * normalized_x, axis=0), out=gamma.grad.eval)
    if beta.requires_grad:
        GLOBAL.np.add(beta.grad.eval, GLOBAL.np.sum(dnormalized_x, axis=0), out=beta.grad.eval)

    if inputs.requires_grad:
        xmu = outputs.cache['xmu']
        std_inv = outputs.cache['sqrtvar']
        normalized_x = outputs.cache['normalized_x']
        shape_field = outputs.cache['shape_field']

        GLOBAL.np.divide(1., std_inv, out=std_inv)
        N = GLOBAL.np.prod(GLOBAL.np.asarray(normalized_x.shape[1:]))
        GLOBAL.np.multiply(dnormalized_x, gamma.eval, out=dnormalized_x)

        dvar = GLOBAL.np.multiply(GLOBAL.np.multiply(-0.5, GLOBAL.np.sum(GLOBAL.np.multiply(dnormalized_x, xmu), axis=shape_field, keepdims=True)), GLOBAL.np.power(std_inv, 3))
        # dvar = (-0.5) * GLOBAL.np.sum(dnormalized_x * xmu, axis=shape_field, keepdims=True) * (
                    # std_inv ** 3)  # (m,1)=(m,c,h,w)*(m,c,h,w)*(m,1)
        dmean = GLOBAL.np.multiply(-1., GLOBAL.np.sum(GLOBAL.np.multiply(dnormalized_x, std_inv), axis=shape_field, keepdims=True))
        GLOBAL.np.subtract(dmean, GLOBAL.np.multiply(-2, GLOBAL.np.multiply(dvar, GLOBAL.np.mean(xmu, axis=shape_field, keepdims=True))), out=dmean)
        # dmean = (-1.0) * GLOBAL.np.sum(dnormalized_x * std_inv, axis=shape_field, keepdims=True) - 2.0 * dvar * GLOBAL.np.mean(xmu,
        #                                                                                                          axis=shape_field, keepdims=True)
        GLOBAL.np.multiply(dnormalized_x, std_inv, out=dnormalized_x)
        GLOBAL.np.multiply(dvar, xmu, out=dvar)
        GLOBAL.np.multiply((2. / N), dvar, out=dvar)
        GLOBAL.np.multiply((1. / N), dmean, out=dmean)
        GLOBAL.np.add(dnormalized_x, dvar, out=dnormalized_x)
        GLOBAL.np.add(dnormalized_x, dmean, out=dnormalized_x)
        # grad = dnormalized_x * std_inv + (2. / N) * dvar * xmu + (1. / N) * dmean
        GLOBAL.np.add(inputs.grad.eval, dnormalized_x, out=inputs.grad.eval)


def GroupNormBackward(outputs: Tensor):
    inputs, gamma, beta = outputs.in_bounds
    dx_norm = outputs.grad.eval
    if gamma.requires_grad:
        x_norm = outputs.cache['x_norm']
        GLOBAL.np.add(gamma.grad.eval, GLOBAL.np.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True), out=gamma.grad.eval)
    if beta.requires_grad:
        GLOBAL.np.add(beta.grad.eval, GLOBAL.np.sum(dx_norm, axis=(0, 2, 3), keepdims=True), out=beta.grad.eval)

    if inputs.requires_grad:
        n, c, h, w = dx_norm.shape
        groups = outputs.cache['groups']
        std_inv = outputs.cache['sqrtvar']
        xgmu = outputs.cache['xgmu']
        GLOBAL.np.divide(1., std_inv, out=std_inv)
        # dx_group_norm
        GLOBAL.np.multiply(dx_norm, gamma.eval, out=dx_norm) # (N,C,H,W)
        dx_group_norm = GLOBAL.np.reshape(dx_norm, (n, groups, c // groups, h, w))
        # dvar
        dvar = GLOBAL.np.multiply(-0.5, GLOBAL.np.multiply(GLOBAL.np.power(std_inv, 3), GLOBAL.np.sum(GLOBAL.np.multiply(dx_group_norm, xgmu), axis=(2, 3, 4), keepdims=True)))
        # dvar = -0.5 * (std_inv ** 3) * GLOBAL.np.sum(dx_group_norm * xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean
        N_GROUP = c // groups * h * w
        GLOBAL.np.multiply(dx_group_norm, std_inv, out=dx_group_norm)

        dmean = GLOBAL.np.sum(-dx_group_norm, axis=(2, 3, 4), keepdims=True)
        # dmean1 = GLOBAL.np.sum(dx_group_norm * -std_inv, axis=(2, 3, 4), keepdims=True)
        dmean2_var = GLOBAL.np.multiply(GLOBAL.np.divide(GLOBAL.np.multiply(dvar, -2.0), N_GROUP), GLOBAL.np.sum(xgmu, axis=(2, 3, 4), keepdims=True))
        GLOBAL.np.add(dmean, dmean2_var, out=dmean)
        # dmean2_var = dvar * -2.0 / N_GROUP * GLOBAL.np.sum(xgmu, axis=(2, 3, 4), keepdims=True)
        # dmean = dmean1 + dmean2_var
        # dx_group
        GLOBAL.np.multiply(dvar, 2.0, out=dvar)
        GLOBAL.np.divide(dvar, N_GROUP, out=dvar)
        GLOBAL.np.multiply(dvar, xgmu, out=dvar)
        #dx_group_var = dvar * 2.0 / N_GROUP * xgmu
        GLOBAL.np.multiply(dmean, 1.0, out=dmean)
        GLOBAL.np.divide(dmean, N_GROUP, out=dmean)
        # dx_group_mean = dmean * 1.0 / N_GROUP
        GLOBAL.np.add(dx_group_norm, dvar, out=dx_group_norm)
        GLOBAL.np.add(dx_group_norm, dmean, out=dx_group_norm)
        # dx_group = dx_group1 + dx_group_var + dx_group_mean
        # dx
        grad = GLOBAL.np.reshape(dx_group_norm, (n, c, h, w))
        GLOBAL.np.add(inputs.grad.eval, grad, out=inputs.grad.eval)


def ViewBackward(outputs: Tensor):
    inputs, = outputs.in_bounds
    if inputs.requires_grad:
        GLOBAL.np.add(inputs.grad.eval, GLOBAL.np.reshape(outputs.grad.eval, inputs.shape), out=inputs.grad.eval)


def MSELossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    gradients =GLOBAL.np.multiply(GLOBAL.np.subtract(y_pred.eval, y_true.eval), outputs.grad.eval)
    if outputs.cache['reduction'] == 'mean':
        GLOBAL.np.divide(gradients, GLOBAL.np.prod(y_pred.shape), out=gradients)
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(-gradients)
        else:
            GLOBAL.np.add(y_true.grad.eval, -gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def MAEBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    pos = GLOBAL.np.where((y_pred.eval - y_true.eval) < 0)
    mask = GLOBAL.np.ones_like(y_pred.eval)
    mask[pos] = -1
    if outputs.cache['reduction'] == 'mean':
        GLOBAL.np.divide(mask, y_pred.shape[0], out=mask)
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(mask)
        else:
            GLOBAL.np.add(y_true.grad.eval, mask, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(mask)
        else:
            GLOBAL.np.add(y_pred.grad.eval, mask, out=y_pred.grad.eval)


def BCELossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    gradients = GLOBAL.np.subtract(GLOBAL.np.divide(GLOBAL.np.subtract(1, y_true.eval), GLOBAL.np.subtract(1, y_pred.eval)), GLOBAL.np.divide(y_true.eval, y_pred.eval))
    if outputs.cache['reduction'] == 'mean':
        avg = GLOBAL.np.prod(y_pred.shape)
        GLOBAL.np.divide(gradients, avg, out=gradients)
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(y_true.grad.eval, gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def BCEWithLogitsLossBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    logits = nn.td_functional.sigmoid(y_pred.eval)
    gradients = GLOBAL.np.subtract(GLOBAL.np.divide(GLOBAL.np.subtract(1, y_true.eval), GLOBAL.np.subtract(1, logits)), GLOBAL.np.divide(y_true.eval, logits))
    if outputs.cache['reduction'] == 'mean':
        avg = GLOBAL.np.prod(y_pred.shape)
        GLOBAL.np.divide(gradients, avg, out=gradients)
    if y_true.requires_grad:
        if y_true.grad is None:
            y_true.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(y_true.grad.eval, gradients, out=y_true.grad.eval)

    if y_pred.requires_grad:
        GLOBAL.np.multiply(GLOBAL.np.multiply(gradients, logits), GLOBAL.np.subtract(1, logits), out=gradients)
        if y_pred.grad is None:
            y_pred.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(y_pred.grad.eval, gradients, out=y_pred.grad.eval)


def SparseCrossEntropyBackward(outputs: Tensor):
    y_pred, y_true = outputs.in_bounds
    # # before softmax
    before_softmax_y_pred = y_pred.in_bounds[0]
    avg = GLOBAL.np.prod(GLOBAL.np.asarray(y_pred.eval.shape[:-1]))
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


def CrossEntropyLossBackward(outputs: Tensor):
    before_softmax_y_pred, y_true = outputs.in_bounds
    y_pred = before_softmax_y_pred.cache['softmax']
    to_sum_dim = GLOBAL.np.prod(GLOBAL.np.asarray(before_softmax_y_pred.shape[:-1])).item()
    probs = y_pred.eval.reshape(-1,  before_softmax_y_pred.shape[-1])
    y_flat = y_true.eval.reshape(to_sum_dim)
    probs[GLOBAL.np.arange(to_sum_dim), y_flat] -= 1
    gradients = probs.reshape(before_softmax_y_pred.shape)
    if outputs.cache['reduction'] == 'mean':
        n = before_softmax_y_pred.eval.shape[0]
        GLOBAL.np.divide(gradients, n, out=gradients)
    gradients = GLOBAL.np.multiply(gradients, outputs.grad.eval, out=gradients)
    if before_softmax_y_pred.requires_grad:
        if before_softmax_y_pred.grad is None:
            before_softmax_y_pred.grad = Tensor(gradients)
        else:
            GLOBAL.np.add(before_softmax_y_pred.grad.eval, gradients, out=before_softmax_y_pred.grad.eval)


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
    inputs, weight, bias = outputs.in_bounds
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

    da_next = GLOBAL.np.zeros_like(prev_a.eval[:, 0, :])
    dc_next = GLOBAL.np.zeros_like(c.eval[:, 0, :])
    if inputs.requires_grad:
        grad = GLOBAL.np.zeros_like(inputs.eval)
    if return_sequences:
        for t in reversed(range(time_steps)):
            da = outputs.grad.eval.eval[:, t, :] + da_next
            dtao_o = da * GLOBAL.np.tanh(c.eval[:, t + 1, :])
            do = recurrent_activations[3 * (t + 1) - 1].backward(dtao_o)
            dc = dc_next
            dc += da * tao_o.eval[:, t, :] * (1 - GLOBAL.np.square(GLOBAL.np.tanh(c.eval[:, t + 1, :])))
            dc_tilde = dc * tao_u.eval[:, t, :]
            dc_tilde_before_act = activations[t].backward(dc_tilde)
            dtao_u = dc * c_tilde.eval[:, t, :]
            du = recurrent_activations[3 * (t + 1) - 2].backward(dtao_u)
            dtao_f = dc * c.eval[:, t, :]
            df = recurrent_activations[3 * (t + 1) - 3].backward(dtao_f)
            dgrad = GLOBAL.np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
            if weight.requires_grad:
                weight.grad.eval.eval += GLOBAL.np.dot(inputs.eval[:, t, :].T, dgrad)
            if bias.requires_grad:
                bias.grad.eval.eval += GLOBAL.np.sum(dgrad, axis=0, keepdims=True)

            dz = dgrad.dot(weight.eval.T)

            da_next = dz[:, :units]
            dc_next = dc * tao_f.eval[:, t, :]
            if inputs.requires_grad:
                grad[:, t, :] = dz[:, units:]
    else:
        da = outputs.grad.eval.eval + da_next
        for t in reversed(range(time_steps)):
            dtao_o = da * GLOBAL.np.tanh(c.eval[:, t + 1, :])
            recurrent_activations[3 * (t + 1) - 1].eval.grad.eval.eval = dtao_o
            recurrent_activations[3 * (t + 1) - 1].backward()
            do = recurrent_activations[3 * (t + 1) - 1].input_data.grad.eval.eval

            dc = dc_next
            dc += da * tao_o.eval[:, t, :] * (1 - GLOBAL.np.square(GLOBAL.np.tanh(c.eval[:, t + 1, :])))

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

            dgrad = GLOBAL.np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
            if weight.requires_grad:
                zt = GLOBAL.np.concatenate((prev_a.eval[:, t - 1, :], inputs.eval[:, t - 1, :]), axis=1)
                weight.grad.eval.eval += GLOBAL.np.dot(zt.T, dgrad)
            if bias.requires_grad:
                bias.grad.eval.eval += GLOBAL.np.sum(dgrad, axis=0, keepdims=True)

            dz = dgrad.dot(weight.eval.T)

            da = dz[:, :units]
            dc_next = dc * tao_f.eval[:, t, :]
            if inputs.requires_grad:
                grad[:, t, :] = dz[:, units:]
    if inputs.requires_grad:
        inputs.grad.eval.eval += grad