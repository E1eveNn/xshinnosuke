from nn.core import Variable
from nn.grad_fn import *
from utils.toolkit import im2col, initialize_ops_grad
from typing import List


def relu(inputs: Variable, inplace: bool = False) -> Variable:
    if inplace:
        inputs.data[inputs.data < 0] = 0
        return inputs
    outputs = Variable(data=np.maximum(0, inputs.data), in_bounds=[inputs])
    outputs.grad_fn = ReluBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def sigmoid(inputs: Variable):
    outputs = 1. / (1 + np.exp(-inputs.data))
    outputs = Variable(data=outputs, in_bounds=[inputs])
    outputs.grad_fn = SigmoidBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def tanh(inputs: Variable):
    outputs = np.tanh(inputs.data)
    outputs = Variable(data=outputs, in_bounds=[inputs])
    outputs.grad_fn = TanhBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def softmax(inputs: Variable):
    # more stable softmax
    shiftx = inputs.data - np.max(inputs.data)
    outputs = np.divide(np.exp(shiftx), np.sum(np.exp(shiftx), axis=-1, keepdims=True))
    outputs = Variable(data=outputs, in_bounds=[inputs])
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def dense(inputs: Variable, weight: Variable, bias: Variable = None):
    outputs = inputs.matmul(weight)
    if bias is not None:
        outputs += bias
    else:
        # 这里加入None是为了数组大小能对应
        outputs.in_bounds.append(None)
    outputs.grad_fn = DenseBackward
    initialize_ops_grad(inputs, weight, bias)
    return outputs


def flatten(inputs: Variable, start: int = 1,inplace: bool = False):
    output_shape = tuple(inputs.shape[:start]) + (-1, )
    if inplace:
        inputs.data = inputs.data.reshape(output_shape)
        return inputs
    outputs = Variable(data=inputs.data.reshape(output_shape), in_bounds=[inputs])
    outputs.grad_fn = FlattenBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def conv2d(inputs: Variable, weight: Variable, bias: Variable = None, stride: int = 1, padding: int = 0):
    # before pad size
    batch_nums, n_c_prev, n_h_prev, n_w_prev = inputs.data.shape
    # pad
    pad_data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    out_channels, in_channels, kernel_h, kernel_w = weight.data.shape
    # output size
    n_h = (n_h_prev - kernel_h + 2 * padding) // stride + 1
    n_w = (n_w_prev - kernel_w + 2 * padding) // stride + 1
    col = im2col(pad_data, n_h, n_w, kernel_h, kernel_w, stride)

    col_w = weight.data.reshape(out_channels, -1).T

    outputs = col.dot(col_w)
    if bias is not None:
        outputs += bias.data
    outputs = outputs.reshape(batch_nums, n_h, n_w, -1).transpose(0, 3, 1, 2)
    outputs = Variable(data=outputs, in_bounds=[inputs, weight, bias])
    # store these for bp
    outputs.cache['col'] = col
    outputs.cache['stride'] = stride
    outputs.cache['pad_size'] = padding
    outputs.grad_fn = Conv2DBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs, weight, bias)
    return outputs


def max_pool2d(inputs: Variable, kernel_size: int = 2, stride: int = None, padding: int = 0):
    if stride is None:
        stride = kernel_size
    if padding != 0:
        inputs.data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    n, c, h, w = inputs.data.shape
    if kernel_size == stride:
        mode = 'reshape'

        x_reshaped = inputs.data.reshape((n, c, h // kernel_size, kernel_size, w // kernel_size, kernel_size))
        outputs = Variable(data=x_reshaped.max(axis=3).max(axis=4), in_bounds=[inputs])
        outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        out_h, out_w = (h - kernel_size + 2 * padding) // stride + 1, (w - kernel_size + 2 * padding) // stride + 1
        col = im2col(inputs.data, out_h, out_w, kernel_size, kernel_size, stride)
        col = col.reshape(-1, kernel_size * kernel_size)
        pool_argmax = np.argmax(col, axis=1)
        outputs = np.max(col, axis=1).reshape((n, out_h, out_w, c)).transpose(0, 3, 1, 2)
        outputs = Variable(data=outputs, in_bounds=[inputs])
        outputs.cache['pool_argmax'] = pool_argmax
        outputs.cache['kernel_size'] = kernel_size

    outputs.cache['mode'] = mode
    outputs.cache['pad_size'] = padding
    outputs.cache['stride'] = stride
    outputs.grad_fn = Maxpool2DBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def avg_pool2d(inputs: Variable, kernel_size: int, stride: int = None, padding: int = 0):
    if padding != 0:
        inputs.data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    n, c, h, w = inputs.data.shape
    out_h, out_w = (h - kernel_size + 2 * padding) // stride + 1, (w - kernel_size + 2 * padding) // stride + 1
    col = im2col(inputs.data, out_h, out_w, kernel_size, kernel_size, stride)
    col = col.reshape(-1, kernel_size * kernel_size)
    pool_argmean = np.array([range(col.shape[1])])
    outputs = np.mean(col, axis=1).reshape((n, out_h, out_w, c)).transpose(0, 3, 1, 2)
    outputs = Variable(data=outputs, in_bounds=[inputs])
    outputs.cache['pool_argmean'] = pool_argmean
    outputs.cache['kernel_size'] = kernel_size
    outputs.cache['pad_size'] = padding
    outputs.cache['stride'] = stride
    outputs.grad_fn = Avgpool2DBackward
    inputs.out_bounds.append(outputs)
    initialize_ops_grad(inputs)
    return outputs


def pad_2d(inputs: Variable, padding: tuple, inplace: bool = False):
    if inplace:
        inputs.data = np.pad(inputs.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                             'constant')
        return inputs
    outputs = Variable(data=np.pad(inputs.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                                   'constant'), in_bounds=[inputs])
    outputs.cache['pad_size'] = padding
    outputs.grad_fn = Pad2DBackward
    initialize_ops_grad(inputs)
    return outputs


def dropout2d(inputs: Variable, keep_prob: float = 0.5):
    random_tensor = np.random.binomial(n=1, p=keep_prob, size=inputs.data.shape)
    outputs = inputs.data * random_tensor / keep_prob
    outputs = Variable(data=outputs, in_bounds=[inputs])
    outputs.cache['mask'] = random_tensor
    outputs.cache['keep_prob'] = keep_prob
    outputs.grad_fn = Dropout2DBackward
    initialize_ops_grad(inputs)
    return outputs


def concatenate(*variables: Variable, axis: int, output: Variable = None, name: str = None):
    data = variables[0].data
    for i in range(1, len(variables)):
        data = np.concatenate((data, variables[i].data), axis=axis)
    if output is None:
        output = Variable(data=data, name=name, in_bounds=[variables])
    else:
        output.data = data
        output.shape = data.shape
        output.name = name
        output.in_bounds.append(*variables)
    return output
