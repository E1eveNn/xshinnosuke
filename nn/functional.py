from .core import Variable
from .grad_fn import *
from .toolkit import im2col, initialize_ops_grad
from xshinnosuke.nn.initializers import Zeros, Ones
from typing import Tuple


def relu(inputs: Variable, inplace: bool = False) -> Variable:
    if inplace:
        inputs.data[inputs.data < 0] = 0
        return inputs
    outputs = Variable(data=np.maximum(0, inputs.data), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = ReluBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def sigmoid(inputs: Variable):
    outputs = 1. / (1 + np.exp(-inputs.data))
    outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = SigmoidBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def tanh(inputs: Variable):
    outputs = np.tanh(inputs.data)
    outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = TanhBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def softmax(inputs: Variable):
    # more stable softmax
    shiftx = inputs.data - np.max(inputs.data)
    outputs = np.divide(np.exp(shiftx), np.sum(np.exp(shiftx), axis=-1, keepdims=True))
    outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def dense(inputs: Variable, weight: Variable, bias: Variable = None):
    outputs = inputs.data.dot(weight.data)
    if bias is not None:
        outputs += bias.data
    outputs = Variable(data=outputs, in_bounds=[inputs, weight, bias], requires_grad=inputs.requires_grad or weight.requires_grad or (bias and bias.requires_grad))
    if outputs.requires_grad:
        outputs.grad_fn = DenseBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs, weight, bias)
    return outputs


def flatten(inputs: Variable, start: int = 1,inplace: bool = False):
    output_shape = tuple(inputs.shape[:start]) + (-1, )
    if inplace:
        inputs.data = inputs.data.reshape(output_shape)
        return inputs
    outputs = Variable(data=inputs.data.reshape(output_shape), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = FlattenBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def embedding(inputs: Variable, weight: Variable):
    outputs = Variable(weight.data[inputs.data.astype(np.int)], in_bounds=[inputs, weight], requires_grad=inputs.requires_grad or weight.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = EmbeddingBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs, weight)
    return outputs


def conv2d(inputs: Variable, weight: Variable, bias: Variable = None, stride: Tuple = (1, 1), padding: int = 0):
    # before pad size
    batch_nums, n_c_prev, n_h_prev, n_w_prev = inputs.data.shape
    # pad
    pad_data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')

    out_channels, in_channels, kernel_h, kernel_w = weight.data.shape
    # output size
    n_h = (n_h_prev - kernel_h + 2 * padding) // stride[0] + 1
    n_w = (n_w_prev - kernel_w + 2 * padding) // stride[1] + 1
    col = im2col(pad_data, n_h, n_w, kernel_h, kernel_w, stride)

    col_w = weight.data.reshape(out_channels, -1).T

    outputs = col.dot(col_w)
    if bias is not None:
        outputs += bias.data
    outputs = outputs.reshape(batch_nums, n_h, n_w, -1).transpose(0, 3, 1, 2)
    outputs = Variable(data=outputs, in_bounds=[inputs, weight, bias], requires_grad=inputs.requires_grad or weight.requires_grad or (bias and bias.requires_grad))
    if outputs.requires_grad:
        # store these for bp
        outputs.cache['col'] = col
        outputs.cache['stride'] = stride
        outputs.cache['pad_size'] = padding
        outputs.grad_fn = Conv2DBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs, weight, bias)
    return outputs


def max_pool2d(inputs: Variable, kernel_size: int = 2, stride: Tuple = None, padding: int = 0):
    if stride is None:
        stride = kernel_size
    if padding != 0:
        data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        data = inputs.data

    n, c, h, w = data.shape
    if kernel_size == stride:
        mode = 'reshape'

        x_reshaped = data.reshape((n, c, h // kernel_size, kernel_size, w // kernel_size, kernel_size))
        outputs = Variable(data=x_reshaped.max(axis=3).max(axis=4), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'

        out_h, out_w = (h - kernel_size) // stride[0] + 1, (w - kernel_size) // stride[1] + 1

        col = im2col(data, out_h, out_w, kernel_size, kernel_size, stride)
        col = col.reshape(-1, kernel_size * kernel_size)
        pool_argmax = np.argmax(col, axis=1)
        outputs = np.max(col, axis=1).reshape((n, out_h, out_w, c)).transpose(0, 3, 1, 2)
        outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        outputs.cache['pool_argmax'] = pool_argmax
        outputs.cache['kernel_size'] = kernel_size
    if outputs.requires_grad:
        outputs.cache['mode'] = mode
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = Maxpool2DBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def channel_max_pool(inputs: Variable, kernel_size: int = 2, stride: int = 1, padding: int = 0):
    if stride is None:
        stride = kernel_size
    if padding != 0:
        data = np.pad(inputs.data, ((0, 0), (padding, padding), (0, 0), (0, 0)), 'constant')
    else:
        data = inputs.data

    n, c, h, w = data.shape
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c // kernel_size, kernel_size, h, w))
        outputs = Variable(data=x_reshaped.max(axis=2), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        out_c = (c - kernel_size) // stride + 1
        col = np.zeros((n, kernel_size, out_c, h, w))
        for y in range(kernel_size):
            y_max = y + stride * out_c
            col[:, y] = data[:, y: y_max: stride]

        pool_argmax = np.argmax(col, axis=1)
        outputs = np.max(col, axis=1).reshape((n, out_c, h, w))
        outputs = Variable(data=outputs, in_bounds=[inputs], requires_grad=inputs.requires_grad)
        outputs.cache['pool_argmax'] = pool_argmax
        outputs.cache['kernel_size'] = kernel_size
    if outputs.requires_grad:
        outputs.cache['mode'] = mode
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = ChannelMaxpoolBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def avg_pool2d(inputs: Variable, kernel_size: int, stride: Tuple = None, padding: int = 0):
    if padding != 0:
        data = np.pad(inputs.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        data = inputs.data

    n, c, h, w = data.shape
    out_h, out_w = (h - kernel_size) // stride[0] + 1, (w - kernel_size) // stride[1] + 1
    col = im2col(data, out_h, out_w, kernel_size, kernel_size, stride)
    col = col.reshape(-1, kernel_size * kernel_size)
    pool_argmean = np.array([range(col.shape[1])])
    outputs = np.mean(col, axis=1).reshape((n, out_h, out_w, c)).transpose(0, 3, 1, 2)
    outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.cache['pool_argmean'] = pool_argmean
        outputs.cache['kernel_size'] = kernel_size
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = Avgpool2DBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def channel_avg_pool(inputs: Variable, kernel_size: int = 2, stride: int = 1, padding: int = 0):
    if stride is None:
        stride = kernel_size
    if padding != 0:
        data = np.pad(inputs.data, ((0, 0), (padding, padding), (0, 0), (0, 0)), 'constant')
    else:
        data = inputs.data

    n, c, h, w = data.shape
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c // kernel_size, kernel_size, h, w))
        outputs = Variable(data=x_reshaped.mean(axis=2), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        out_c = (c - kernel_size) // stride + 1
        col = np.zeros((n, kernel_size, out_c, h, w))
        for y in range(kernel_size):
            y_max = y + stride * out_c
            col[:, y] = data[:, y: y_max: stride]

        pool_argmean = np.array([range(col.shape[1])])
        outputs = np.mean(col, axis=1).reshape((n, out_c, h, w))
        outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        outputs.cache['pool_argmean'] = pool_argmean
        outputs.cache['kernel_size'] = kernel_size
    if outputs.requires_grad:
        outputs.cache['mode'] = mode
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = ChannelAvgpoolBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def pad_2d(inputs: Variable, padding: tuple, inplace: bool = False):
    if inplace:
        inputs.data = np.pad(inputs.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                             'constant')
        return inputs
    outputs = Variable(data=np.pad(inputs.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                                   'constant'), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.cache['pad_size'] = padding
        outputs.grad_fn = Pad2DBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def dropout2d(inputs: Variable, keep_prob: float = 0.5, training: bool = True):
    if not training:
        return inputs
    random_tensor = np.random.binomial(n=1, p=keep_prob, size=inputs.data.shape)
    outputs = inputs.data * random_tensor / keep_prob
    outputs = Variable(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.cache['mask'] = random_tensor
        outputs.cache['keep_prob'] = keep_prob
        outputs.grad_fn = Dropout2DBackward
        inputs.out_bounds.append(outputs)
        initialize_ops_grad(inputs)
    return outputs


def batchnorm2d(inputs: Variable, gamma: Variable, beta: Variable, axis: int, epsilon: float = 1e-6,
                training: bool = True, momentum: float = 0.99, moving_mean: np.ndarray = None, moving_variance: np.ndarray = None):
    if moving_mean is not None:
        inputs.cache['moving_mean'] = moving_mean
    if moving_variance is not None:
        inputs.cache['moving_variance'] = moving_variance
    if 'moving_mean' not in inputs.cache.keys():
        inputs.cache['moving_mean'] = Zeros()(inputs.shape[axis])
    if 'moving_variance' not in inputs.cache.keys():
        inputs.cache['moving_variance'] = Ones()(inputs.shape[axis])

    inputs_data = inputs.data
    ndim = inputs_data.ndim

    if not (axis == -1 or axis == ndim - 1):
        inputs_data = np.swapaxes(inputs_data, axis, -1)

    before_reshape_shape = inputs_data.shape
    inputs_data = inputs_data.reshape(-1, inputs.data.shape[axis])

    if training:
        # calc mean
        mean = np.mean(inputs_data, axis=0)
        # calc var
        var = np.var(inputs_data, axis=0)
        # x minus u
        xmu = inputs_data - mean
        sqrtvar = np.sqrt(var + epsilon)
        normalized_x = xmu / sqrtvar
        outputs = gamma.data * normalized_x + beta.data

        inputs.cache['xmu'] = xmu
        inputs.cache['axis'] = axis
        inputs.cache['sqrtvar'] = sqrtvar
        inputs.cache['normalized_x'] = normalized_x
        inputs.cache['moving_mean'] = momentum * inputs.cache['moving_mean'].data + (1 - momentum) * mean
        inputs.cache['moving_variance'] = momentum * inputs.cache['moving_variance'].data + (1 - momentum) * var

    else:
        scale = gamma.data / (np.sqrt(inputs.cache['moving_variance'].data + epsilon))
        outputs = inputs_data * scale + (beta.data - inputs.cache['moving_mean'].data * scale)

    outputs = outputs.reshape(before_reshape_shape)

    if not (axis == -1 or axis == ndim - 1):
        # for instance,outputs:(N,W,H,C), self.axis=1, after swapaxes,outputs:(N,C,H,W)
        outputs = np.swapaxes(outputs, axis, -1)

    outputs = Variable(data=outputs, in_bounds=[inputs, gamma, beta], requires_grad=inputs.requires_grad or gamma.requires_grad or beta.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = Batchnorm2DBackward
        initialize_ops_grad(inputs)
        inputs.out_bounds.append(outputs)
    return outputs


def layernorm2d(inputs: Variable, gamma: Variable, beta: Variable, training: bool = True, epsilon: float = 1e-10):
    inputs_data = inputs.data
    shape_field = tuple([i for i in range(1, inputs_data.ndim)])
    # calc mean
    mean = np.mean(inputs_data, axis=shape_field, keepdims=True)
    # calc var
    var = np.var(inputs_data, axis=shape_field, keepdims=True)
    # x minus u
    xmu = inputs_data - mean
    sqrtvar = np.sqrt(var + epsilon)
    normalized_x = xmu / sqrtvar
    outputs = gamma.data * normalized_x + beta.data
    if training:
        inputs.cache['shape_field'] = shape_field
        inputs.cache['xmu'] = xmu
        inputs.cache['sqrtvar'] = sqrtvar
        inputs.cache['normalized_x'] = normalized_x

    outputs = Variable(data=outputs, in_bounds=[inputs, gamma, beta], requires_grad=inputs.requires_grad or gamma.requires_grad or beta.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = Layernorm2DBackward
        initialize_ops_grad(inputs)
        inputs.out_bounds.append(outputs)
    return outputs


def groupnorm2d(inputs: Variable, gamma: Variable, beta: Variable, training: bool = True, epsilon: float = 1e-5,
                groups: int = 16):
    inputs_data = inputs.data
    n, c, h, w = inputs_data.shape
    shape_field = tuple([i for i in range(2, inputs_data.ndim)])
    x_group = np.reshape(inputs_data, (n, groups, c // groups, h, w))
    mean = np.mean(x_group, axis=shape_field, keepdims=True)
    var = np.var(x_group, axis=shape_field, keepdims=True)
    xgmu = x_group - mean
    sqrtvar = np.sqrt(var + epsilon)
    x_group_norm = xgmu / sqrtvar
    x_norm = np.reshape(x_group_norm, (n, c, h, w))
    outputs = gamma.data * x_norm + beta.data
    if training:
        inputs.cache['groups'] = groups
        inputs.cache['xgmu'] = xgmu
        inputs.cache['sqrtvar'] = sqrtvar
        inputs.cache['x_norm'] = x_norm

    outputs = Variable(data=outputs, in_bounds=[inputs, gamma, beta], requires_grad=inputs.requires_grad or gamma.requires_grad or beta.requires_grad)
    if outputs.requires_grad:
        outputs.grad_fn = Layernorm2DBackward
        initialize_ops_grad(inputs)
        inputs.out_bounds.append(outputs)
    return outputs


def reshape(inputs: Variable, shape: Tuple, inplace: bool = True):
    if inplace:
        inputs.cache['inplace'] = inplace
        inputs.cache['input_shape'] = inputs.shape
        inputs.data = np.reshape(inputs.data, shape)
        inputs.shape = inputs.data.shape
        return inputs
    outputs = Variable(data=np.reshape(inputs.data, shape), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
    if outputs.requires_grad:
        outputs.cache['inplace'] = inplace
        outputs.grad_fn = ReshapeBackward
        initialize_ops_grad(inputs)
        inputs.out_bounds.append(outputs)
    return outputs


def concatenate(*variables: Variable, axis: int, output: Variable = None, name: str = None):
    data = variables[0].data
    requires_grad = variables[0].requires_grad
    for i in range(1, len(variables)):
        data = np.concatenate((data, variables[i].data), axis=axis)
        requires_grad = requires_grad or variables[i].requires_grad
    if output is None:
        output = Variable(data=data, name=name, in_bounds=[variables], requires_grad=requires_grad)
    else:
        output.data = data
        output.shape = data.shape
        output.name = name
        output.in_bounds.append(*variables)
        output.requires_grad = requires_grad
    return output
