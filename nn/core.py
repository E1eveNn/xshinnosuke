import warnings
from .grad_fn import *
from typing import List, Tuple, Union
import gc
from . import global_graph as GlobalGraph
from .toolkit import initialize_ops_grad
import abc


class Node(metaclass=abc.ABCMeta):
    def __init__(self, in_bounds: List = None, out_bounds: List = None, data: GlobalGraph.np.ndarray = None,
                 shape: Union[List, Tuple] = None, name: str = None, requires_grad: bool = False):
        self.in_bounds = [] if in_bounds is None else list(in_bounds)
        self.out_bounds = [] if out_bounds is None else list(out_bounds)
        self.data = data
        self.shape = self.__check_shape(shape)
        self.name = name
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.cache = {}
        self.retain = False
        self.__parameters = []

    def parameters(self):
        return self.__parameters

    def set_parameters(self, variables: List):
        self.__parameters = variables

    def retain_grad(self):
        self.retain = True

    def __repr__(self) -> str:
        if self.grad_fn:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                f'grad_fn={self.grad_fn.__name__})'
        else:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                f'grad_fn={self.grad_fn})'
        return info

    def __str__(self) -> str:
        if self.grad_fn:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                f'grad_fn=<{self.grad_fn.__name__}>)'
        else:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad})'
        return info

    def __check_shape(self, shape: Union[List, Tuple]) -> Union[List, Tuple]:
        if len(shape) == 0:
            return tuple([1])
        return shape

    # overload
    def __add__(self, other):
        # 全局运算图
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        return add(self, other)

    def __iadd__(self, other):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        self.data += other.data
        self.in_bounds.append(other)
        other.out_bounds.append(self)
        if 'grad_fn' not in self.cache:
            self.cache['grad_fn'] = []
        self.cache['grad_fn'].append(self.grad_fn)
        if GlobalGraph.IS_TRAINING and self.requires_grad:
            self.grad_fn = IAddBackward
            initialize_ops_grad(self, other)
        return self

    def __eq__(self, other):
        if self.data.size != other.data.size:
            return False
        return Variable(self.data == other.data, requires_grad=self.requires_grad or other.requires_grad)

    def __hash__(self):
        return hash(id(self))

    def __neg__(self):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        return neg(self)

    def __sub__(self, other):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        return sub(self, other)

    def __mul__(self, other):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        return mul(self, other)

    def __truediv__(self, other):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        return div(self, other)

    def __pow__(self, power: int, modulo=None):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        outputs = Variable(in_bounds=[self, ], data=GlobalGraph.np.power(self.data, power), requires_grad=True)
        # 绑定反向求梯度的函数
        outputs.grad_fn = PowBackward
        outputs.cache['power'] = power
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    def __matmul__(self, other):
        return self.matmul(other)

    def backward(self, gradients: GlobalGraph.np.ndarray = None):
        if self.grad_fn is None:
            raise ValueError('can not solve grad on %s' % self)
        if gradients is not None:
            if isinstance(gradients, Variable):
                gradients = gradients.data
            assert gradients.size == self.data.size and gradients.shape == self.data.shape
            self.grad = gradients
        else:
            if self.data.size == 1:
                if self.grad is None:
                    self.grad = GlobalGraph.np.array(1.)
            else:
                raise ValueError('grad can be implicitly created only for scalar outputs')

        if GlobalGraph.OUTPUTS is None:
            GlobalGraph.OUTPUTS = self
        if GlobalGraph.INPUTS is not None:
            graph = GlobalGraph.build_graph()
            GlobalGraph.INPUTS.retain_grad()
            GlobalGraph.OUTPUTS.retain_grad()
            for node in reversed(graph):
                if node.grad_fn is not None:
                    node.grad_fn(node)
                if node.retain:
                    GlobalGraph.reset_node(node)
                else:
                    GlobalGraph.delete_node(node)

            GlobalGraph.reset_graph()
            gc.collect()

    def size(self, axis: int = None):
        return self.data.shape if axis is None else self.data.shape[axis]

    def zero_grad(self):
        self.grad = GlobalGraph.np.zeros_like(self.data)

    def long(self):
        output = Variable(data=self.data, dtype=GlobalGraph.np.int64, requires_grad=self.requires_grad)
        return output

    def long_(self):
        self.data = self.data.astype(GlobalGraph.np.int64)

    def int_(self):
        self.data = self.data.astype(GlobalGraph.np.int32)

    def l2norm(self):
        output = Variable(data=GlobalGraph.np.sqrt(GlobalGraph.np.sum(GlobalGraph.np.square(self.data))), requires_grad=self.requires_grad)
        return output


    def matmul(self, other):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        outputs_data = GlobalGraph.np.dot(self.data, other.data)
        outputs = Variable(data=outputs_data, in_bounds=[self, other], requires_grad=self.requires_grad or other.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = MatmulBackward
            initialize_ops_grad(self, other)
            self.out_bounds.append(outputs)
            other.out_bounds.append(outputs)
        return outputs

    def dot(self, other):
        return self.matmul(other)

    def t(self):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        outputs = Variable(data=self.data.T, in_bounds=[self, ], requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = TransposeBackward
            self.out_bounds.append(outputs)
            initialize_ops_grad(self)
        return outputs

    def sum(self, axis: int = None, keepdims: bool = False):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        if axis is None:
            sum_value = GlobalGraph.np.sum(self.data, keepdims=keepdims)
        else:
            sum_value = GlobalGraph.np.sum(self.data, axis=axis, keepdims=keepdims)
        outputs = Variable(in_bounds=[self, ], data=sum_value, requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.cache['axis'] = axis
            outputs.grad_fn = SumBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def mean(self, axis: int = None, keepdims: bool = False):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        if axis is None:
            mean_value = GlobalGraph.np.mean(self.data, keepdims=keepdims)
        else:
            mean_value = GlobalGraph.np.mean(self.data, axis=axis, keepdims=keepdims)
        outputs = Variable(in_bounds=[self, ], data=mean_value, requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.cache['axis'] = axis
            outputs.grad_fn = MeanBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def abs(self):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        outputs = Variable(in_bounds=[self, ], data=GlobalGraph.np.abs(self.data), requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = AbsBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def view(self, *shapes):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        outputs = Variable(in_bounds=[self, ], data=self.data.reshape(*shapes), requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = ViewBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def log(self, base: int = 2):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        if base == 2:
            ret_value = GlobalGraph.np.log2(self.data)
        elif base == 10:
            ret_value = GlobalGraph.np.log10(self.data)
        else:
            ret_value = GlobalGraph.np.log(self.data)
        outputs = Variable(in_bounds=[self, ], data=ret_value, requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = LogBackward
            outputs.cache['base'] = base
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def exp(self):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self

        outputs = Variable(in_bounds=[self, ], data=GlobalGraph.np.exp(self.data), requires_grad=self.requires_grad)
        if outputs.requires_grad:
            outputs.grad_fn = ExpBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def max(self, axis=None, **kwargs):
        if GlobalGraph.INPUTS is None:
            GlobalGraph.INPUTS = self
        requires_grad = kwargs.pop('requires_grad', False)
        keepdims = kwargs.pop('keepdims', False)
        outputs = Variable(in_bounds=[self, ], data=GlobalGraph.np.max(self.data, axis=axis, keepdims=keepdims),
                           requires_grad=self.requires_grad or requires_grad, **kwargs)
        if outputs.requires_grad:
            outputs.cache['axis'] = axis
            outputs.grad_fn = MaxBackward
            initialize_ops_grad(self)
            self.out_bounds.append(outputs)
        return outputs

    def argmax(self, axis=None, **kwargs):
        requires_grad = kwargs.pop('requires_grad', False)
        outputs = Variable(in_bounds=[self, ], data=GlobalGraph.np.argmax(self.data, axis=axis), requires_grad=self.requires_grad or requires_grad)
        return outputs

    def numel(self):
        return self.data.size


class Variable(Node):
    def __init__(self, data: Union[GlobalGraph.np.ndarray, int, float], in_bounds: List = None,
                 out_bounds: Union[List, Tuple] = None,
                 name: str = None, requires_grad: bool = True, dtype: str = 'float32'):
        if isinstance(data, Variable):
            # Node.__init__(self,
            #               in_bounds=data.in_bounds,
            #               out_bounds=data.out_bounds,
            #               data=data.data,
            #               shape=data.data.shape,
            #               name=data.name,
            #               requires_grad=data.requires_grad)
            pass
        else:
            dtype_dict = {'int': GlobalGraph.np.int, 'float': GlobalGraph.np.float, 'int8': GlobalGraph.np.int8, 'int16': GlobalGraph.np.int16, 'int32': GlobalGraph.np.int32,
                          'int64': GlobalGraph.np.int64, 'float32': GlobalGraph.np.float32, 'float64': GlobalGraph.np.float64}
            data = GlobalGraph.np.asarray(data, dtype=dtype_dict[dtype])
            Node.__init__(self,
                          in_bounds=in_bounds,
                          out_bounds=out_bounds,
                          data=data,
                          shape=data.shape,
                          name=name,
                          requires_grad=requires_grad)

    def __getitem__(self, item):
        return slices(self, item, GlobalGraph.IS_TRAINING)

    def __setitem__(self, key, value):
        copySlices(self, key, value, GlobalGraph.IS_TRAINING)


class Constant(Node):
    def __init__(self, data: Union[GlobalGraph.np.ndarray, int, float], in_bounds: List = None,
                 out_bounds: Union[List, Tuple] = None,
                 name: str = None, dtype: str = 'float32'):
        dtype_dict = {'int': GlobalGraph.np.int, 'float': GlobalGraph.np.float, 'int8': GlobalGraph.np.int8, 'int16': GlobalGraph.np.int16, 'int32': GlobalGraph.np.int32,
                      'int64': GlobalGraph.np.int64, 'float32': GlobalGraph.np.float32, 'float64': GlobalGraph.np.float64}
        data = GlobalGraph.np.asarray(data, dtype=dtype_dict[dtype])
        Node.__init__(self,
                      in_bounds=in_bounds,
                      out_bounds=out_bounds,
                      data=data,
                      name=name,
                      shape=data.shape,
                      requires_grad=False)

    def __setattr__(self, key, value):
        if key == 'data' and key in self.__dict__:
            warnings.warn('Can not change the value of a Constant!')
        else:
            self.__dict__[key] = value


class Layer:
    def __init__(self, in_bounds: List = None, out_bounds: List = None, input_shape: Union[List, Tuple] = None,
                 input_data: Variable = None, data: Variable = None, shape: Union[List, Tuple] = None,
                 variables: List[Variable] = None, name: str = None):
        self.in_bounds = [] if in_bounds is None else list(in_bounds)
        self.out_bounds = [] if out_bounds is None else list(out_bounds)
        self.data = data
        self.input_shape = input_shape
        self.input_data = input_data
        self.name = name
        self.shape = shape

        self.variables = variables if variables else []

    def __call__(self, inbound, *args, **kwargs):
        self.shape = self.compute_output_shape(inbound.shape)
        self.in_bounds.append(inbound)
        self.input_shape = inbound.shape
        inbound.out_bounds.append(self)
        return self

    def __rshift__(self, other):
        return other.__call__(self)

    def connect(self, inbound=None):
        if inbound is None:
            if self.input_shape is None:
                raise ValueError('must specify input_shape')
        else:
            self.input_shape = inbound.shape
        self.shape = self.compute_output_shape(self.input_shape)
        if inbound is not None:
            self.in_bounds.append(inbound)
            inbound.out_bounds.append(self)

    def initial_params(self, *args):
        pass

    def parameters(self):
        return self.variables

    def set_parameters(self, variables: List):
        self.variables = variables

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return input_shape

    def params_count(self) -> int:
        total_params = 0
        for v in self.variables:
            if v is not None:
                total_params += v.data.size
        return total_params

    def forward(self, x: Variable = None, *args) -> Variable:
        raise NotImplemented

    def feed_variable_to_next_layers(self, data: Variable):
        for out_bound in self.out_bounds:
            out_bound.input_data = data

    def backward(self, gradients: GlobalGraph.np.ndarray = None):
        for inbound in self.in_bounds:
            if inbound.data.requires_grad:
                inbound.data.grad += gradients


def add(*ops: Variable) -> Variable:
    data = 0
    requires_grad = False
    for op in ops:
        data += op.data
        requires_grad = requires_grad or op.requires_grad
    outputs = Variable(in_bounds=[*ops], data=data, requires_grad=requires_grad)
    if outputs.requires_grad:
        initialize_ops_grad(*ops)
        outputs.grad_fn = AddBackward
        for op in ops:
            op.out_bounds.append(outputs)
    return outputs


def sub(*ops: Variable) -> Variable:
    outputs = ops[0].data
    requires_grad = ops[0].requires_grad
    for op in ops[1:]:
        outputs = outputs - op.data
        requires_grad = requires_grad or op.requires_grad
    outputs = Variable(in_bounds=[*ops], data=outputs, requires_grad=requires_grad)
    if outputs.requires_grad:
        initialize_ops_grad(*ops)
        outputs.grad_fn = SubBackward
        for op in ops:
            op.out_bounds.append(outputs)
    return outputs


def neg(x: Variable) -> Variable:
    outputs = Variable(in_bounds=[x], data=-x.data, requires_grad=x.requires_grad)
    if outputs.requires_grad:
        x.out_bounds.append(outputs)
        initialize_ops_grad(x)
        outputs.grad_fn = NegBackward
    return outputs


def mul(x: Variable, y: Variable) -> Variable:
    if isinstance(y, int) or isinstance(y, float):
        y = Variable(y, requires_grad=False)
    outputs = None
    if len(x.shape) == 1 or len(y.shape) == 1:
        # 点乘
        outputs = Variable(in_bounds=[x, y], data=x.data * y.data, requires_grad=x.requires_grad or y.requires_grad)
        outputs.grad_fn = MultiplyBackward
    elif len(x.shape) == len(y.shape):
        matmul_flag = False
        for s1, s2 in zip(x.shape, y.shape):
            if s1 != s2 and s1 != 1 and s2 != 1:
                matmul_flag = True
                break
        if matmul_flag:
            outputs = Variable(in_bounds=[x, y], data=GlobalGraph.np.dot(x.data, y.data), requires_grad=x.requires_grad or y.requires_grad)
            outputs.grad_fn = MatmulBackward
        else:
            outputs = Variable(in_bounds=[x, y], data=x.data * y.data, requires_grad=x.requires_grad or y.requires_grad)
            outputs.grad_fn = MultiplyBackward
    else:
        raise ValueError('can\'t peroform either multiply nor matmul between {} and {}'.format(x, y))
    if outputs.requires_grad:
        initialize_ops_grad(x, y)
        x.out_bounds.append(outputs)
        y.out_bounds.append(outputs)
    return outputs


def div(x: Variable, y: Variable) -> Variable:
    if isinstance(y, Variable):
        data = x.data / y.data
        requires_grad = x.requires_grad or y.requires_grad
    elif isinstance(y, (int, float)):
        data = x.data / y
        requires_grad = x.requires_grad
    else:
        raise ValueError
    outputs = Variable(in_bounds=[x, y], data=data, requires_grad=requires_grad)
    if outputs.requires_grad:
        initialize_ops_grad(x, y)
        outputs.grad_fn = DivBackward
        x.out_bounds.append(outputs)
        y.out_bounds.append(outputs)
    return outputs


def slices(x, item, training):
    if GlobalGraph.INPUTS is None:
        GlobalGraph.INPUTS = x
    ret = Variable(data=x.data[item], in_bounds=[x, ], requires_grad=x.requires_grad)
    if ret.requires_grad:
        x.out_bounds.append(ret)
        ret.cache['pos'] = item
        initialize_ops_grad(x)
        ret.grad_fn = SlicesBackward
    return ret


def copySlices(x, pos, value: Variable, training):
    x.data[pos] = value.data
    value.out_bounds.append(x)
    x.in_bounds.append(value)
    x.cache[pos] = value
    initialize_ops_grad(value)
    x.grad_fn = CopySlicesBackward
