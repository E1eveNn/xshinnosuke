from .grad_fn import *
from core import __global as GLOBAL


# math opearations
def add(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(lhs.eval))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.add(lhs.eval, rhs.eval, out.eval)
    if out.is_dynamic:
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
        out.add_in_bounds(lhs, rhs)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = AddBackward
    return out


def sub(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(lhs.eval))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.sub(lhs.eval, rhs.eval, out.eval)
    if out.is_dynamic:
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
        out.add_in_bounds(lhs, rhs)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = SubBackward
    return out


def mul(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(lhs.eval))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.mul(lhs.eval, rhs.eval, out.eval)

    if out.is_dynamic:
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
        out.add_in_bounds(lhs, rhs)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = MultiplyBackward
    return out


def pow(x: Tensor, exp: Union[int, float], out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(x.eval))
    out.requires_grad = x.requires_grad
    GLOBAL.np.power(x.eval, exp, out=x.eval)

    if out.is_dynamic:
        x.add_out_bounds(out)
        out.add_in_bounds(x)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(x)
        out.cache['exp'] = exp
        out.grad_fn = PowBackward
    return out


# 矩阵乘法
def mm(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty((lhs.shape[0], rhs.shape[-1])))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.mm(lhs.eval, rhs.eval, out.eval)

    if out.is_dynamic:
        out.add_in_bounds(lhs, rhs)
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = MMBackward
    return out


def addmm(b: Tensor, x1: Tensor, x2: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    requires_grad = x1.requires_grad or x2.requires_grad or b.requires_grad
    if out is not None:
        assert isinstance(out, Tensor)
        nn.td_functional.mm(x1.eval, x2.eval, out.eval)
        nn.td_functional.add(out.eval, b.eval, out.eval)
    else:
        out_data = nn.td_functional.mm(x1.eval, x2.eval)
        nn.td_functional.add(out_data, b.eval, out_data)
        out = Tensor(data=out_data)
    out.requires_grad = requires_grad
    if out.is_dynamic:
        out.add_in_bounds(x1, x2, b)
        b.add_out_bounds(out)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
    if GLOBAL.COMPUTE_GRAD and requires_grad:
        initialize_ops_grad(b, x1, x2)
        out.grad_fn = AddmmBackward
    return out


def truediv(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(lhs.eval))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.div(lhs.eval, rhs.eval, out.eval)

    if out.is_dynamic:
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
        out.add_in_bounds(lhs, rhs)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = DivBackward
    return out


def floordiv(lhs: Tensor, rhs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = lhs
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(lhs.eval))
    out.requires_grad = lhs.requires_grad or rhs.requires_grad
    nn.td_functional.floor_div(lhs.eval, rhs.eval, out.eval)

    if out.is_dynamic:
        lhs.add_out_bounds(out)
        rhs.add_out_bounds(out)
        out.add_in_bounds(lhs, rhs)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        initialize_ops_grad(lhs, rhs)
        out.grad_fn = DivBackward
    return out


def max(x: Tensor, axis: int = None, keepdims: bool = False, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is not None:
        assert isinstance(out, Tensor)
        nn.td_functional.max(x.eval, axis, keepdims, out.eval)
    else:
        out = Tensor(nn.td_functional.max(x.eval, axis, keepdims))

    if out.is_dynamic:
        out.add_in_bounds(x)
        x.add_out_bounds(out)
    out.requires_grad = x.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['axis'] = axis
        out.grad_fn = MaxBackward
        initialize_ops_grad(x)
    return out


def maximum(x1: Union[int, float, ndarray, Tensor], x2: Union[int, float, GLOBAL.np.ndarray, Tensor], out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if isinstance(x1, Tensor):
        if out is None:
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        if isinstance(x2, (int, float, ndarray)):
            nn.td_functional.maximum(x1.eval, x2, out.eval)
            out.requires_grad = x1.requires_grad
        elif isinstance(x2, Tensor):
            nn.td_functional.maximum(x1.eval, x2.eval, out.eval)
            out.requires_grad = x1.requires_grad or x2.requires_grad
        else:
            raise TypeError('unknown type for {}'.format(type(x2)))
    elif isinstance(x2, Tensor):
        if out is None:
            out = Tensor(GLOBAL.np.empty_like(x2.eval))
        if isinstance(x1, (int, float, GLOBAL.np.ndarray)):
            nn.td_functional.maximum(x1, x2.eval, out.eval)
            out.requires_grad = x2.requires_grad
        elif isinstance(x1, Tensor):
            nn.td_functional.maximum(x1.eval, x2.eval, out.eval)
            out.requires_grad = x1.requires_grad or x2.requires_grad
        else:
            raise TypeError('unknown type for {}'.format(type(x1)))
    else:
        assert isinstance(x2, (int, float, GLOBAL.np.ndarray)) and isinstance(x1, (int, float, GLOBAL.np.ndarray))
        if out is None:
            out = Tensor(GLOBAL.np.maximum(x1, x2))
        else:
            GLOBAL.np.maximum(x1, x2, out=out.eval)

    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = MaximumBackward
        initialize_ops_grad(x1, x2)
    return out


def sum(x: Tensor, axis: int = None, keepdims: bool = False, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is not None:
        assert isinstance(out, Tensor)
        nn.td_functional.sum(x.eval, axis, keepdims, out.eval)
    else:
        out = Tensor(nn.td_functional.sum(x.eval, axis, keepdims))

    if out.is_dynamic:
        out.add_in_bounds(x)
        x.add_out_bounds(out)
    out.requires_grad = x.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['axis'] = axis
        out.grad_fn = SumBackward
        initialize_ops_grad(x)
    return out


def mean(x: Tensor, axis: int = None, keepdims: bool = False, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is not None:
        assert isinstance(out, Tensor)
        nn.td_functional.mean(x.eval, axis, keepdims, out.eval)
    else:
        out = Tensor(nn.td_functional.mean(x.eval, axis, keepdims))
    out.requires_grad = x.requires_grad
    if out.is_dynamic:
        out.add_in_bounds(x)
        x.add_out_bounds(out)
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['axis'] = axis
        out.grad_fn = MeanBackward
        initialize_ops_grad(x)
    return out


def norm(x: Tensor, p: int = 2, axis: int = None, keepdims: bool = False, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is not None:
        assert isinstance(out, Tensor)
        nn.td_functional.norm(x.eval, p, axis, keepdims, out.eval)
    else:
        out = Tensor(nn.td_functional.norm(x.eval, p, axis, keepdims))
    if out.is_dynamic:
        out.add_in_bounds(x)
        x.add_out_bounds(out)
    out.requires_grad = x.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = None
        initialize_ops_grad(x)
    return out


def transpose(x: Tensor, out: Tensor = None):
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(x.eval))
    out.eval = GLOBAL.np.transpose(x.eval)
    if out.is_dynamic:
        out.add_in_bounds(x)
        x.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = TransposeBackward
        initialize_ops_grad(x)
    return out


def relu(inputs: Tensor, inplace: bool = False, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if inplace:
        nn.td_functional.relu(inputs.eval, inputs.eval)
        return inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    nn.td_functional.relu(inputs.eval, out.eval)
    out.requires_grad = inputs.requires_grad
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = ReLUBackward
        initialize_ops_grad(inputs)
    return out


def sigmoid(inputs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    nn.td_functional.sigmoid(inputs.eval, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = SigmoidBackward
        initialize_ops_grad(inputs)
    return out


def tanh(inputs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    nn.td_functional.tanh(inputs.eval, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = TanhBackward
        initialize_ops_grad(inputs)
    return out


def softmax(inputs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    out.requires_grad = inputs.requires_grad
    nn.td_functional.softmax(inputs.eval, out.eval)
    if 'softmax' in inputs.cache.keys():
        inputs.cache['softmax'] = out
    return out


def log_softmax(inputs: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    out.requires_grad = inputs.requires_grad
    nn.td_functional.log_softmax(inputs.eval, out.eval)
    return out


def flatten(inputs: Tensor, start: int = 1, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=nn.td_functional.flatten(inputs.eval, start))
    else:
        out.eval = nn.td_functional.flatten(inputs.eval, start)
    out.requires_grad = inputs.requires_grad
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = FlattenBackward
        initialize_ops_grad(inputs)
    return out


def embedding(inputs: Tensor, weight: Tensor, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is not None:
        out.eval = weight.eval[inputs.eval.astype(GLOBAL.np.int)]
    else:
        out = Tensor(data=weight.eval[inputs.eval.astype(GLOBAL.np.int)])
    if out.is_dynamic:
        out.add_in_bounds(inputs, weight)
        inputs.add_out_bounds(out)
        weight.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad or weight.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = EmbeddingBackward
        initialize_ops_grad(inputs, weight)
    return out


def conv2d(inputs: Tensor, weight: Tensor, bias: Tensor = None, stride: Tuple = (1, 1), padding: int = 0,
            out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    # before pad size
    batch_nums, n_c_prev, n_h_prev, n_w_prev = inputs.eval.shape
    # pad
    pad_data = GLOBAL.np.pad(inputs.eval, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    out_channels, in_channels, kernel_h, kernel_w = weight.eval.shape
    # output size
    n_h = (n_h_prev - kernel_h + 2 * padding) // stride[0] + 1
    n_w = (n_w_prev - kernel_w + 2 * padding) // stride[1] + 1
    col = im2col(pad_data, n_h, n_w, kernel_h, kernel_w, stride)
    col_w = weight.eval.reshape(out_channels, -1).T
    requires_grad = inputs.requires_grad or weight.requires_grad or (bias and bias.requires_grad)
    if out is None:
        out = Tensor(data=GLOBAL.np.empty((batch_nums, out_channels, n_h, n_w), dtype=inputs.dtype))

    data = GLOBAL.np.dot(col, col_w)
    if bias is not None:
        GLOBAL.np.add(data, bias.eval, out=data)
    out.eval = GLOBAL.np.reshape(data, (batch_nums, n_h, n_w, -1)).transpose(0, 3, 1, 2)
    out.requires_grad = requires_grad
    if out.is_dynamic:
        out.add_in_bounds(inputs, weight, bias)
        inputs.add_out_bounds(out)
        weight.add_out_bounds(out)
        if bias is not None:
            bias.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        # store these for bp
        out.cache['col'] = col
        out.cache['stride'] = stride
        out.cache['padding'] = padding
        out.grad_fn = Conv2DBackward
        initialize_ops_grad(inputs, weight, bias)
    return out


def max_pool2d(inputs: Tensor, kernel_size: int = 2, stride: int = None, padding: int = 0, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if stride is None:
        stride = (kernel_size, kernel_size)
    elif isinstance(stride, int):
        stride = (stride, stride)
    if padding != 0:
        data = GLOBAL.np.pad(inputs.eval, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        data = inputs.eval
    # after padding's shape if padded
    n, c, h, w = data.shape
    out_h = (h - kernel_size) // stride[0] + 1
    out_w = (w - kernel_size) // stride[1] + 1
    if out is None:
        out = Tensor(GLOBAL.np.empty((n, c, out_h, out_w), dtype=inputs.dtype))
    out.requires_grad = inputs.requires_grad
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c, h // kernel_size, kernel_size, w // kernel_size, kernel_size))
        data = GLOBAL.np.max(x_reshaped, axis=3)
        GLOBAL.np.max(data, axis=4, out=out.eval)
        if GLOBAL.COMPUTE_GRAD and out.requires_grad:
            out.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        col = im2col(data, out_h, out_w, kernel_size, kernel_size, stride)
        col = col.reshape(-1, kernel_size * kernel_size)
        pool_argmax = GLOBAL.np.argmax(col, axis=1)
        col = GLOBAL.np.max(col, axis=1)
        out.eval = GLOBAL.np.reshape(col, (n, out_h, out_w, c)).transpose(0, 3, 1, 2)
        if GLOBAL.COMPUTE_GRAD and out.requires_grad:
            out.cache['pool_argmax'] = pool_argmax
            out.cache['kernel_size'] = kernel_size
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['mode'] = mode
        out.cache['stride'] = stride
        out.cache['padding'] = padding
        out.grad_fn = Maxpool2DBackward
        initialize_ops_grad(inputs)
    return out


def channel_max_pool(inputs: Tensor, kernel_size: int = 2, stride: int = 1, padding: int = 0, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if stride is None:
        stride = (kernel_size, kernel_size)
    elif isinstance(stride, int):
        stride = (stride, stride)
    if padding != 0:
        data = GLOBAL.np.pad(inputs.eval, ((0, 0), (padding, padding), (0, 0), (0, 0)), 'constant')
    else:
        data = inputs.eval

    n, c, h, w = data.shape
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c // kernel_size, kernel_size, h, w))
        outputs = Tensor(data=x_reshaped.max(axis=2), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
            outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        out_c = (c - kernel_size) // stride + 1
        col = GLOBAL.np.zeros((n, kernel_size, out_c, h, w))
        for y in range(kernel_size):
            y_max = y + stride * out_c
            col[:, y] = data[:, y: y_max: stride]

        pool_argmax = GLOBAL.np.argmax(col, axis=1)
        outputs = GLOBAL.np.max(col, axis=1).reshape((n, out_c, h, w))
        outputs = Tensor(data=outputs, in_bounds=[inputs], requires_grad=inputs.requires_grad)
        if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
            outputs.cache['pool_argmax'] = pool_argmax
            outputs.cache['kernel_size'] = kernel_size

    inputs.out_bounds.append(outputs)
    if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
        outputs.cache['mode'] = mode
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = ChannelMaxpoolBackward
        initialize_ops_grad(inputs)
    return outputs


def avg_pool2d(inputs: Tensor, kernel_size: int, stride: int = None, padding: int = 0, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if stride is None:
        stride = (kernel_size, kernel_size)
    elif isinstance(stride, int):
        stride = (stride, stride)
    if padding != 0:
        data = GLOBAL.np.pad(inputs.eval, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        data = inputs.eval

    n, c, h, w = data.shape
    out_h = (h - kernel_size + 2 * padding) // stride[0] + 1
    out_w = (w - kernel_size + 2 * padding) // stride[1] + 1
    if out is None:
        out = Tensor(GLOBAL.np.empty((n, c, out_h, out_w), dtype=inputs.dtype))
    out.requires_grad = inputs.requires_grad
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c, h // kernel_size, kernel_size, w // kernel_size, kernel_size))
        GLOBAL.np.mean(x_reshaped, axis=(3, 5), out=out.eval)
        if GLOBAL.COMPUTE_GRAD and out.requires_grad:
            out.cache['reshaped_shape'] = x_reshaped.shape
    else:
        mode = 'im2col'
        col = im2col(data, out_h, out_w, kernel_size, kernel_size, stride)
        col = col.reshape(-1, kernel_size * kernel_size)
        pool_argmean = GLOBAL.np.array([range(col.shape[1])])
        col = GLOBAL.np.mean(col, axis=1)
        out.eval = GLOBAL.np.reshape(col, (n, out_h, out_w, c)).transpose(0, 3, 1, 2)
        if GLOBAL.COMPUTE_GRAD and out.requires_grad:
            out.cache['pool_argmean'] = pool_argmean
            out.cache['kernel_size'] = kernel_size
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['mode'] = mode
        out.cache['stride'] = stride
        out.cache['padding'] = padding
        out.grad_fn = Avgpool2DBackward
        initialize_ops_grad(inputs)
    return out


def channel_avg_pool(inputs: Tensor, kernel_size: int = 2, stride: int = 1, padding: int = 0, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if stride is None:
        stride = kernel_size
    if padding != 0:
        data = GLOBAL.np.pad(inputs.eval, ((0, 0), (padding, padding), (0, 0), (0, 0)), 'constant')
    else:
        data = inputs.eval

    n, c, h, w = data.shape
    if kernel_size == stride:
        mode = 'reshape'
        x_reshaped = data.reshape((n, c // kernel_size, kernel_size, h, w))
        outputs = Tensor(data=x_reshaped.mean(axis=2), in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
            outputs.cache['x_reshaped'] = x_reshaped

    else:
        mode = 'im2col'
        out_c = (c - kernel_size) // stride + 1
        col = GLOBAL.np.zeros((n, kernel_size, out_c, h, w))
        for y in range(kernel_size):
            y_max = y + stride * out_c
            col[:, y] = data[:, y: y_max: stride]

        pool_argmean = GLOBAL.np.array([range(col.shape[1])])
        outputs = GLOBAL.np.mean(col, axis=1).reshape((n, out_c, h, w))
        outputs = Tensor(data=outputs, in_bounds=[inputs, ], requires_grad=inputs.requires_grad)
        if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
            outputs.cache['pool_argmean'] = pool_argmean
            outputs.cache['kernel_size'] = kernel_size

    inputs.out_bounds.append(outputs)
    if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
        outputs.cache['mode'] = mode
        outputs.cache['pad_size'] = padding
        outputs.cache['stride'] = stride
        outputs.grad_fn = ChannelAvgpoolBackward
        initialize_ops_grad(inputs)
    return outputs


def pad2d(inputs: Tensor, padding: tuple, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        n, c, h, w = inputs.shape
        out_h = h + 2 * padding[0]
        out_w = w + 2 * padding[1]
        out = Tensor(GLOBAL.np.empty((n, c, out_h, out_w), dtype=inputs.dtype))
    out.eval = GLOBAL.np.pad(inputs.eval, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                           'constant')
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['padding'] = padding
        out.grad_fn = Pad2DBackward
        initialize_ops_grad(inputs)
    return out


def dropout2d(inputs: Tensor, keep_prob: float = 0.5, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if not GLOBAL.COMPUTE_GRAD:
        return inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    random_tensor = None
    if GLOBAL.TRAINING:
        random_tensor = GLOBAL.np.random.binomial(n=1, p=keep_prob, size=inputs.eval.shape)
        GLOBAL.np.multiply(inputs.eval, random_tensor, out=out.eval)
        GLOBAL.np.divide(out.eval, keep_prob, out=out.eval)
    else:
        out.eval = inputs.eval
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['mask'] = random_tensor
        out.cache['keep_prob'] = keep_prob
        out.grad_fn = Dropout2DBackward
        initialize_ops_grad(inputs)
    return out


def batch_norm(inputs: Tensor, gamma: Tensor, beta: Tensor, moving_mean: Tensor, moving_variance: Tensor, axis: int = 1,
               training: bool = True, epsilon: float = 1e-5, momentum: float = 0.9, out: Tensor = None) -> Tensor:

    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    axis_field = tuple([i for i in range(inputs.eval.ndim) if i != axis])
    sqrtvar = None
    normalized_x = None
    if training:
        # calc mean
        mean = GLOBAL.np.mean(inputs.eval, axis=axis_field, keepdims=True)
        # calc var
        var = GLOBAL.np.var(inputs.eval, axis=axis_field, keepdims=True)
        # x minus u
        xmu = GLOBAL.np.subtract(inputs.eval, mean)
        sqrtvar = GLOBAL.np.sqrt(var + epsilon)
        normalized_x = GLOBAL.np.divide(xmu, sqrtvar)

        GLOBAL.np.multiply(nn.td_functional.expand_as(gamma.eval, normalized_x), normalized_x, out=out.eval)
        GLOBAL.np.add(out.eval, nn.td_functional.expand_as(beta.eval, out.eval), out=out.eval)

        # update moving mean
        GLOBAL.np.multiply(momentum, moving_mean.eval, out=moving_mean.eval)
        GLOBAL.np.multiply(1 - momentum, mean, out=mean)
        GLOBAL.np.add(moving_mean.eval, mean.ravel(), out=moving_mean.eval)
        # update moving variance
        GLOBAL.np.multiply(momentum, moving_variance.eval, out=moving_variance.eval)
        GLOBAL.np.multiply(1 - momentum, var, out=var)
        GLOBAL.np.add(moving_variance.eval, var.ravel(), out=moving_variance.eval)

    else:
        # moving_variance_data = GLOBAL.np.add(moving_variance.eval, epsilon)
        # GLOBAL.np.sqrt(moving_variance_data, out=moving_variance_data)
        # GLOBAL.np.divide(gamma.eval, moving_variance_data, out=moving_variance_data)
        #
        # moving_mean_data = GLOBAL.np.multiply(moving_variance_data, moving_mean.eval)
        # GLOBAL.np.subtract(beta.eval, moving_mean_data, out=moving_mean_data)
        #
        # GLOBAL.np.multiply(inputs.eval, moving_variance_data, out=moving_variance_data)
        # GLOBAL.np.add(moving_variance_data, moving_mean_data, out=out.eval)
        moving_variance_data = GLOBAL.np.add(moving_variance.eval, epsilon)
        GLOBAL.np.sqrt(moving_variance_data, out=moving_variance_data)
        moving_mean_data = GLOBAL.np.subtract(inputs.eval, nn.td_functional.expand_as(moving_mean.eval, inputs.eval))
        GLOBAL.np.divide(moving_mean_data, nn.td_functional.expand_as(moving_variance_data, moving_mean_data), out=out.eval)
        # GLOBAL.np.multiply(gamma.eval, out.eval, out=out.eval)
        GLOBAL.np.multiply(nn.td_functional.expand_as(gamma.eval, out.eval), out.eval, out=out.eval)
        # GLOBAL.np.add(beta.eval, out.eval, out=out.eval)
        GLOBAL.np.add(nn.td_functional.expand_as(beta.eval, out.eval), out.eval, out=out.eval)

    out.requires_grad = inputs.requires_grad or gamma.requires_grad or beta.requires_grad
    if out.is_dynamic:
        out.add_in_bounds(inputs, gamma, beta)
        inputs.add_out_bounds(out)
        gamma.add_out_bounds(out)
        beta.add_out_bounds(out)

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['axis_field'] = axis_field
        out.cache['sqrtvar'] = sqrtvar
        out.cache['normalized_x'] = normalized_x
        out.grad_fn = BatchNormBackward
        initialize_ops_grad(inputs, gamma, beta)
    return out


def layer_norm(inputs: Tensor, gamma: Tensor, beta: Tensor, epsilon: float = 1e-10, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))
    shape_field = tuple([i for i in range(1, inputs.eval.ndim)])
    # calc mean
    xmu = GLOBAL.np.mean(inputs.eval, axis=shape_field, keepdims=True)
    # calc var
    sqrtvar = GLOBAL.np.var(inputs.eval, axis=shape_field, keepdims=True)
    # x minus u
    GLOBAL.np.subtract(inputs.eval, xmu, out=xmu)
    GLOBAL.np.sqrt(sqrtvar + epsilon, out=sqrtvar)

    normalized_x = GLOBAL.np.divide(xmu, sqrtvar)
    GLOBAL.np.multiply(nn.td_functional.expand_as(gamma.eval, out.eval), normalized_x, out=out.eval)
    GLOBAL.np.add(out.eval, nn.td_functional.expand_as(beta.eval, out.eval), out=out.eval)
    if out.is_dynamic:
        out.add_in_bounds(inputs, gamma, beta)
        inputs.add_out_bounds(out)
        gamma.add_out_bounds(out)
        beta.add_out_bounds(out)

    out.requires_grad = inputs.requires_grad or gamma.requires_grad or beta.requires_grad

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['shape_field'] = shape_field
        out.cache['xmu'] = xmu
        out.cache['sqrtvar'] = sqrtvar
        out.cache['normalized_x'] = normalized_x
        out.grad_fn = LayerNormBackward
        initialize_ops_grad(inputs, gamma, beta)
    return out


def group_norm(inputs: Tensor, gamma: Tensor, beta: Tensor,  epsilon: float = 1e-5, groups: int = 16, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.empty_like(inputs.eval))

    n, c, h, w = inputs.eval.shape
    shape_field = tuple([i for i in range(2, inputs.eval.ndim)])
    data = GLOBAL.np.reshape(inputs.eval, (n, groups, c // groups, h, w))
    xgmu = GLOBAL.np.mean(data, axis=shape_field, keepdims=True)
    sqrtvar = GLOBAL.np.var(data, axis=shape_field, keepdims=True)
    GLOBAL.np.subtract(data, xgmu, out=xgmu)
    GLOBAL.np.sqrt(sqrtvar + epsilon, out=sqrtvar)

    x_group_norm = GLOBAL.np.divide(xgmu, sqrtvar)
    x_norm = GLOBAL.np.reshape(x_group_norm, (n, c, h, w))
    GLOBAL.np.multiply(nn.td_functional.expand_as(gamma.eval, out.eval), x_norm, out=out.eval)
    GLOBAL.np.add(out.eval, nn.td_functional.expand_as(beta.eval, out.eval), out=out.eval)

    if out.is_dynamic:
        out.add_in_bounds(inputs, gamma, beta)
        inputs.add_out_bounds(out)
        gamma.add_out_bounds(out)
        beta.add_out_bounds(out)

    out.requires_grad = inputs.requires_grad or gamma.requires_grad or beta.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['groups'] = groups
        out.cache['xgmu'] = xgmu
        out.cache['sqrtvar'] = sqrtvar
        out.cache['x_norm'] = x_norm
        out.grad_fn = GroupNormBackward
        initialize_ops_grad(inputs, gamma, beta)
    return out


def view(inputs: Tensor, shape: Tuple, out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    if out is None:
        out = Tensor(data=GLOBAL.np.reshape(inputs.eval, shape))
    else:
        out.eval = GLOBAL.np.reshape(inputs.eval, shape)
    if out.is_dynamic:
        out.add_in_bounds(inputs)
        inputs.add_out_bounds(out)
    out.requires_grad = inputs.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = ViewBackward
        initialize_ops_grad(inputs)
    return out


def concat(variables: List[Tensor], axis: int, name: str = None):
    data = variables[0].eval
    requires_grad = variables[0].requires_grad
    for i in range(1, len(variables)):
        data = GLOBAL.np.concatenate((data, variables[i].eval), axis=axis)
        requires_grad = requires_grad or variables[i].requires_grad
    out = Tensor(data=data, name=name, requires_grad=requires_grad)
    if out.is_dynamic:
        out.add_in_bounds(*variables)
    return out


def mse_loss(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    assert x1.shape == x2.shape
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1, )))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')
    if reduction == 'sum':
        loss_val = GLOBAL.np.power(GLOBAL.np.subtract(x1.eval, x2.eval), 2)
        GLOBAL.np.sum(loss_val, out=out.eval)
    elif reduction == 'mean':
        loss_val = GLOBAL.np.sum(GLOBAL.np.power(GLOBAL.np.subtract(x1.eval, x2.eval), 2))
        GLOBAL.np.divide(loss_val, 2, out=out.eval)
        GLOBAL.np.divide(out.eval, GLOBAL.np.prod(x2.shape), out=out.eval)
    else:
        GLOBAL.np.subtract(x1.eval, x2.eval, out=out.eval)
        GLOBAL.np.power(out.eval, 2, out=out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
    out.requires_grad = x1.requires_grad or x2.requires_grad

    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['reduction'] = reduction
        out.grad_fn = MSELossBackward
        initialize_ops_grad(x1, x2)
    return out


def mae_loss(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    assert x1.shape == x2.shape
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1,)))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')
    if reduction == 'sum':
        loss_val = GLOBAL.np.absolute(GLOBAL.np.subtract(x1.eval, x2.eval))
        GLOBAL.np.sum(loss_val, out=out.eval)
    elif reduction == 'mean':
        loss_val = GLOBAL.np.power(GLOBAL.np.subtract(x1.eval, x2.eval), 2)
        GLOBAL.np.sum(loss_val, out=out.eval)
        GLOBAL.np.divide(out.eval, x2.shape[0], out=out.eval)
    else:
        GLOBAL.np.subtract(x1.eval, x2.eval, out=out.eval)
        GLOBAL.np.absolute(out.eval, out=out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
    out.requires_grad = x1.requires_grad or x2.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['reduction'] = reduction
        out.grad_fn = MAEBackward
        initialize_ops_grad(x1, x2)
    return out


def bce_with_logits_loss(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    assert x1.shape == x2.shape
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1,)))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')

    pred = nn.td_functional.sigmoid(x1.eval)
    nn.td_functional.bce_loss(pred, x2.eval, reduction, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
    out.requires_grad = x1.requires_grad or x2.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.grad_fn = BCEWithLogitsLossBackward
        initialize_ops_grad(x1, x2)
    return out


def bce_loss(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    assert x1.shape == x2.shape
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1,)))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')
    nn.td_functional.bce_loss(x1.eval, x2.eval, reduction, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
    out.requires_grad = x1.requires_grad or x2.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['reduction'] = reduction
        out.grad_fn = BCELossBackward
        initialize_ops_grad(x1, x2)
    return out


def nll_loss(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1,)))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')
    x2.long_()
    nn.td_functional.nll_loss(x1.eval, x2.eval, reduction, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
        out.is_leaf = True
    out.requires_grad = x1.requires_grad or x2.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['reduction'] = reduction
        out.grad_fn = NllLossBackward
        initialize_ops_grad(x1, x2)
    return out


def cross_entropy(x1: Tensor, x2: Tensor, reduction: str = 'mean', out: Tensor = None) -> Tensor:
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = x1
    if out is None:
        if reduction == 'mean' or reduction == 'sum':
            out = Tensor(GLOBAL.np.empty((1,)))
        elif reduction == 'none':
            out = Tensor(GLOBAL.np.empty_like(x1.eval))
        else:
            raise TypeError('unknown type for reduction')
    x2.long_()
    if 'softmax' not in x1.cache.keys():
        x1.cache['softmax'] = None
    softmaxed_x1 = softmax(x1, x1.cache['softmax'])
    log_softmaxed_x1_data = GLOBAL.np.log(softmaxed_x1.eval)
    nn.td_functional.nll_loss(log_softmaxed_x1_data, x2.eval, reduction, out.eval)
    if out.is_dynamic:
        out.add_in_bounds(x1, x2)
        x1.add_out_bounds(out)
        x2.add_out_bounds(out)
        out.is_leaf = True
    out.requires_grad = x1.requires_grad or x2.requires_grad
    if GLOBAL.COMPUTE_GRAD and out.requires_grad:
        out.cache['reduction'] = reduction
        out.grad_fn = CrossEntropyLossBackward
        initialize_ops_grad(x1, x2)
    return out


def clip(inputs: Tensor, min_val: float, max_val: float, out: Tensor = None):
    if out is None:
        out = Tensor(GLOBAL.np.empty_like(inputs.eval))
    GLOBAL.np.clip(inputs.eval, min_val, max_val, out=out.eval)
    return out


def lstm(inputs: Tensor, weight: Tensor, bias: Tensor, units: int,
         recurrent_activations: List, activations: List,
         prev_a: Tensor = None, c: Tensor = None, tao_f: Tensor = None,
         tao_u: Tensor = None, tao_o: Tensor = None, c_tilde: Tensor = None,
         return_sequences: bool = False):
    if GLOBAL.INPUTS is None:
        GLOBAL.INPUTS = inputs
    batch_nums, time_steps, n_vec = inputs.eval.shape
    if prev_a is None:
        prev_a = Tensor(GLOBAL.np.zeros((batch_nums, time_steps, n_vec)))
    if c is None:
        c = Tensor(GLOBAL.np.zeros((batch_nums, time_steps, n_vec)))
    if tao_f is None:
        tao_f = Tensor(GLOBAL.np.zeros((batch_nums, time_steps - 1, n_vec)))
    if tao_u is None:
        tao_u = Tensor(GLOBAL.np.zeros((batch_nums, time_steps - 1, n_vec)))
    if tao_o is None:
        tao_o = Tensor(GLOBAL.np.zeros((batch_nums, time_steps - 1, n_vec)))
    if c_tilde is None:
        c_tilde = Tensor(GLOBAL.np.zeros((batch_nums, time_steps - 1, n_vec)))

    z = GLOBAL.np.zeros((batch_nums, time_steps, n_vec + units))
    for t in range(1, time_steps + 1):
        zt = GLOBAL.np.concatenate((prev_a.eval[:, t - 1, :], inputs.eval[:, t - 1, :]), axis=1)
        ot = zt.dot(weight.eval) + bias.eval
        f = recurrent_activations[3 * (t - 1)].forward(Tensor(ot[:, :units]))
        u = recurrent_activations[3 * (t - 1) + 1].forward(Tensor(ot[:, units: units * 2]))
        _c_tilde = activations[t - 1].forward(Tensor(ot[:, units * 2: units * 3]))
        o = recurrent_activations[3 * (t - 1) + 2].forward(Tensor(ot[:, units * 3:]))

        c_tilde.eval[:, t - 1, :] = _c_tilde.eval
        _c = f.eval * c.eval[:, t - 1, :] + u.eval * _c_tilde.eval

        prev_a.eval[:, t, :] = o.eval * GLOBAL.np.tanh(_c)

        tao_f.eval[:, t - 1, :] = f.eval
        tao_u.eval[:, t - 1, :] = u.eval
        tao_o.eval[:, t - 1, :] = o.eval
        c.eval[:, t, :] = _c
        z[:, t - 1, :] = zt
    if return_sequences:
        outputs = Tensor(data=prev_a.eval[:, 1:, :], in_bounds=[inputs, ],
                           requires_grad=inputs.requires_grad or weight.requires_grad or (bias and bias.requires_grad))
    else:
        outputs = Tensor(data=prev_a.eval[:, -1, :], in_bounds=[inputs, ],
                           requires_grad=inputs.requires_grad or weight.requires_grad or (bias and bias.requires_grad))
    outputs.parameters([weight, bias])
    inputs.out_bounds.append(outputs)
    if GLOBAL.COMPUTE_GRAD and outputs.requires_grad:
        outputs.add_cache('units', units)
        outputs.cache['time_steps'] = time_steps
        outputs.cache['recurrent_activations'] = recurrent_activations
        outputs.cache['activations'] = activations
        outputs.cache['prev_a'] = prev_a
        outputs.cache['c'] = c
        outputs.cache['tao_f'] = tao_f
        outputs.cache['tao_u'] = tao_u
        outputs.cache['tao_o'] = tao_o
        outputs.cache['c_tilde'] = c_tilde
        outputs.cache['return_sequences'] = return_sequences
        outputs.grad_fn = LstmBackward
        initialize_ops_grad(inputs, weight, bias)
    return outputs
