from nn.grad_fn import *
from typing import List, Tuple, Union
from numpy import ndarray
import numpy as np
import nn.global_graph as GlobalGraph
from utils.toolkit import initialize_ops_grad


# 第一种数据类型Node，也就是我们基本的tensor type，每一个tensor都是Node的实例化类型
class Node:
    def __init__(self, in_bounds: List = None, out_bounds: List = None, data: ndarray = None,
                 shape: Union[List, Tuple] = None, name: str = None, requires_grad: bool = False):
        """
        :param in_bounds: 与当前tensor相连的Node或者是Layer，需要重载下：一.可能是Layer数组，二.可能是Node数组（主要以Variable为主）
        :param out_bounds: 输出的Node或Layer，也是需要重载，Layer数组或者Node数组
        :param data: 一个numpy矩阵
        :param name: tensor的名字，字符串
        :param requires_grad: 需不需要梯度， 布尔值
        """

        # 下面第一行这句话的意思是如果传递了in_bounds参数，那就把传递的in_bounds先包装为数组，在赋给类成员in_bounds，否则的话初始化类成员in_bounds为空数组，如果c++写的话大概是这样：
        # this->in_bounds = in_bounds == NULL ?  vector<Template> : vector<Template>(in_bounds)
        self.in_bounds = [] if in_bounds is None else list(in_bounds)
        self.out_bounds = [] if out_bounds is None else list(out_bounds)
        self.data = data
        self.shape = self.__check_shape(shape)
        self.name = name
        self.requires_grad = requires_grad
        # grad也是一个numpy矩阵，默认为None，之后会初始化它，在cpp中这样写？
        # 先声明ndarray grad,然后this->grad = NULL
        self.grad = None
        # tensor求梯度的函数，这个要绑定一个函数，比如绑定的是AddBackward（self.grad_fn = AddBackward），那么调用self.grad_fn和调用AddBackward一样

        self.grad_fn = None

    # 该方法相当于重载对类的print, info就是print类后显示的结果
    def __repr__(self) -> str:
        # {}里填写的变量就相当于是把这个变量的值转换为字符串填在这
        if self.grad_fn:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                   f'grad_fn={self.grad_fn.__name__})'
        else:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                   f'grad_fn={self.grad_fn})'
        return info

    def __str__(self) -> str:
        # 同上，不在解释，在python里要重载类的print，最好是把这两个方法都重载了
        if self.grad_fn:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, ' \
                   f'grad_fn=<{self.grad_fn.__name__}>)'
        else:
            info = f'{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad})'
        return info

    def __check_shape(self, shape: Union[List, Tuple]) -> Union[List, Tuple]:
        # 这个是numpy矩阵问题，当一个numpy array存储的是标量时，默认shape是空，因此要手动设为1
        if len(shape) == 0:
            return tuple([1])
        return shape

    # 重载 + - * / 等运算符， 先默认所有的操作数都是Variable类型吧，之后要重载其他类型再说
    # 加法
    def __add__(self, other):
        # 全局运算图
        if GlobalGraph.inputs is None:
            GlobalGraph.inputs = self
        return add(self, other)

    # 加等于
    def __iadd__(self, other):
        if GlobalGraph.inputs is None:
            GlobalGraph.inputs = self

        self.data += other.data
        self.in_bounds.append(other)
        self.grad_fn = IAddBackward
        return self

    # 取负号
    def __neg__(self):
        return neg(self)

    # 减法
    def __sub__(self, other):
        # a - b = a + (-b)
        return add(self, neg(other))

    # 乘法
    def __mul__(self, other):
        if GlobalGraph.inputs is None:
            GlobalGraph.inputs = self
        return mul(self, other)

    # 幂
    def __pow__(self, power: int, modulo=None):
        outputs = Variable(in_bounds=[self], data=np.power(self.data, power), requires_grad=True)
        # 绑定反向求梯度的函数
        outputs.grad_fn = PowBackward
        outputs.power = power
        initialize_ops_grad(self)
        return outputs

    def backward(self, gradients: ndarray = None):
        if self.grad_fn is None:
            raise ValueError('can not solve grad on %s' % self)
        # 如果是标量，默认该标量梯度为1
        if self.data.size == 1:
            if self.grad is None:
                self.grad = 1.
            self.grad_fn(self)
            if GlobalGraph.inputs is not None and GlobalGraph.outputs is not None:
                graph = GlobalGraph.build_graph()
                for node in reversed(graph):
                    if node.grad_fn is not None:
                        node.grad_fn(node)
                # 这里反向传播计算完就把这个图销毁，在C++中就是释放内存,把GlobalGraph的inputs, outputs和graph的内存都释放掉，或者说重置参数
                GlobalGraph.reset_graph()

        # 如果不是标量，但是手动传入该tensor的梯度，也可以求导，但是一定要让传入的梯度shape和tensor的shape对上
        elif gradients is not None:
            self.grad = gradients
            self.grad_fn(self)
        else:
            raise ValueError('grad can be implicitly created only for scalar outputs')

    # 把数据转换为长整型，返回一个Variable
    def long(self):
        output = Variable(data=self.data, dtype=np.int64)
        return output

    # 方法名后面加一个_指在本身数据上直接修改，没有返回值
    def long_(self):
        self.data = self.data.astype(np.int64)

    # 矩阵乘
    def matmul(self, other):
        outputs_data = np.dot(self.data, other.data)
        outputs = Variable(data=outputs_data, in_bounds=[self, other])
        outputs.grad_fn = MatmulBackward
        initialize_ops_grad(self, other)
        self.out_bounds.append(outputs)
        other.out_bounds.append(outputs)
        return outputs

    # 转置
    def t(self):
        outputs = Variable(data=self.data.T, in_bounds=[self])
        outputs.grad_fn = TransposeBackward
        self.out_bounds.append(outputs)
        return outputs

    # 求和
    def sum(self, axis: int = None, keepdims: bool = False):
        # 在哪个axis（维度）上求和
        if axis is None:
            sum_value = np.sum(self.data, keepdims=keepdims)
        else:
            sum_value = np.sum(self.data, axis=axis, keepdims=keepdims)
        outputs = Variable(in_bounds=[self], data=sum_value)
        # 绑定反向求梯度的函数
        outputs.grad_fn = SumBackward
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    # 求均值
    def mean(self, axis: int = None, keepdims: bool = False):
        if axis is None:
            mean_value = np.mean(self.data, keepdims=keepdims)
        else:
            mean_value = np.mean(self.data, axis=axis, keepdims=keepdims)
        outputs = Variable(in_bounds=[self], data=mean_value)
        # 绑定反向求梯度的函数
        outputs.grad_fn = MeanBackward
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    # 求绝对值
    def abs(self):
        outputs = Variable(in_bounds=[self], data=np.abs(self.data))
        # 绑定反向求梯度的函数
        outputs.grad_fn = AbsBackward
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    # reshape，传入的参数应该是一个list或者tuple
    def view(self, *shapes):
        outputs = Variable(in_bounds=[self], data=self.data.reshape(*shapes))
        # 绑定反向求梯度的函数
        outputs.grad_fn = AbsBackward
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    # 求log
    def log(self, base: int = 2):
        # 这里可能需要重载一下, base可选为2， 10和'e'
        if base == 2:
            ret_value = np.log2(self.data)
        elif base == 10:
            ret_value = np.log10(self.data)
        else:
            ret_value = np.log(self.data)
        outputs = Variable(in_bounds=[self], data=ret_value)
        # 绑定反向求梯度的函数
        outputs.grad_fn = LogBackward
        # 记录下base供反向传播用
        outputs.cache['base'] = base
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs

    # 求exp
    def exp(self):
        outputs = Variable(in_bounds=[self], data=np.exp(self.data))
        # 绑定反向求梯度的函数
        outputs.grad_fn = ExpBackward
        initialize_ops_grad(self)
        self.out_bounds.append(outputs)
        return outputs


class Variable(Node):
    def __init__(self, data: Union[ndarray, int, float], in_bounds: List = None, out_bounds: Union[List, Tuple] = None,
                 name: str = None, requires_grad: bool = True, dtype=np.float64):
        # Variable初始化时必须提供data值，data值可能传入的是一个int或者float，我们需要把它包装成numpy矩阵（cpp中就包装成我们选的矩阵库类型）
        data = np.asarray(data, dtype=dtype)  # 数据类型用float64吧，float32也行
        Node.__init__(self,
                      in_bounds=in_bounds,
                      out_bounds=out_bounds,
                      data=data,
                      shape=data.shape,
                      name=name,
                      requires_grad=requires_grad)
        # 存储一些必要的数据用于反向传播，类型为一个字典，通过key-value形式索引
        self.cache = {}

    def __getitem__(self, item):
        ret = Variable(data=self.data[item])
        return ret

    def __setitem__(self, key, value):
        self.data[key] = value.data
        self.in_bounds.append(value)


class Constant(Node):
    def __init__(self, data: Union[ndarray, int, float], in_bounds: List = None, out_bounds: Union[List, Tuple] = None,
                 name: str = None):
        # Constant初始化时必须提供data值，并且一旦初始化就不可修改，Constant因为值不需要修改，也就没必要计算梯度，默认require_grads为False
        # !!!!!!!!!!!!!!!!注意Constant里的这个data要是const类型的，总之就是Constant的data一旦初始化后没办法被修改
        data = np.asarray(data, dtype=np.float64)
        # 显示调用父类的初始化函数
        Node.__init__(self,
                      in_bounds=in_bounds,
                      out_bounds=out_bounds,
                      data=data,
                      name=name,
                      shape=data.shape,
                      requires_grad=False)


class Layer:
    def __init__(self, in_bounds: List = None, out_bounds: List = None, input_shape: Union[List, Tuple] = None,
                 input_data: Variable = None, data: Variable = None, shape: Union[List, Tuple] = None,
                 variables: List[Variable] = None, name: str = None):

        """
        不同于Node，Layer有输入值(input_data，并且是一个Variable)，有运算后的输出值(data，也是一个Variable)，同样有输入数据的shape(input_shape)和输出数据的shape(shape)，还有每一个Layer有自己所做运算需要的参数，如对于全连接层就是w和b，它们存在variables里，variabels的类型是Variable数组
        :param in_bounds:
        :param out_bounds:
        :param input_shape:
        :param shape:
        :param input_value:
        :param data:
        :param variables:
        :param name:
        """
        self.in_bounds = [] if in_bounds is None else list(in_bounds)
        self.out_bounds = [] if out_bounds is None else list(out_bounds)
        self.data = data
        self.input_shape = input_shape
        self.input_data = input_data
        self.name = name
        self.shape = shape
        # 如果传入的variables不是None，就赋值，否则初始化为空的数组
        self.variables = variables if variables else []

    def __call__(self, inbound):
        # 实际inbound可能是Layer也可能是Variable，需要重载，这里在父类里只实现了是Layer的情况，是Variable的情况在子类里在特殊实现。每个子类都会手动调用父类的__call__方法
        # 输入是Layer
        self.shape = self.compute_output_shape(inbound.shape)
        self.in_bounds.append(inbound)
        self.input_shape = inbound.shape
        inbound.out_bounds.append(self)
        return self

    def connect(self, inbound=None):
        # inbound只能是一个Layer
        if inbound is None:
            if self.input_shape is None:
                raise ValueError('must specify input_shape')
        else:
            self.input_shape = inbound.shape
        self.shape = self.compute_output_shape(self.input_shape)
        if inbound is not None:
            self.in_bounds.append(inbound)
            self.input_shape = inbound.shape
            inbound.out_bounds.append(self)

    # 初始化本层参数，由子类单独实现
    def initial_params(self, *args):
        pass

    # 计算本层输出shape，由子类实现
    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        return input_shape

    # 计算本层参数数量，由子类实现
    def params_count(self) -> int:
        total_params = 0
        for v in self.variables:
            if v is not None:
                total_params += v.data.size
        return total_params

    def forward(self, x: Variable = None, is_training: bool = True, *args) -> Variable:
        # x代表本层的输入variable
        raise NotImplemented

    # 做一些forward后的初始化工作，每个子类forward后都会调用该函数
    def connect_init(self, data: Variable, is_training: bool = True):
        for out_bound in self.out_bounds:
            out_bound.input_data = data

        if self.data.requires_grad and is_training:
            initialize_ops_grad(self.data)
            initialize_ops_grad(*self.variables)

    def backward(self, gradients: ndarray = None):
        for inbound in self.in_bounds:
            if inbound.data.requires_grad:
                inbound.data.grad += gradients


# 加法
def add(*ops: Variable) -> Variable:
    data = 0
    for op in ops:
        data += op.data
    outputs = Variable(in_bounds=[*ops], data=data)
    # 绑定反向求梯度的函数
    outputs.grad_fn = AddBackward
    for op in ops:
        initialize_ops_grad(op)
        op.out_bounds.append(outputs)
    return outputs


# 对自身取负
def neg(x: Variable) -> Variable:
    outputs = Variable(in_bounds=[x], data=-x.data, requires_grad=True)
    initialize_ops_grad(x)
    # 绑定反向求梯度的函数
    outputs.grad_fn = NegBackward
    return outputs


# 乘法
def mul(x: Variable, y: Variable) -> Variable:
    # 检查下x和y的shape，看看是要做矩阵乘法运算还是点乘运算

    # 列举所有做点乘的情况, 1.两个标量， 2.两个矩阵的shape一样。
    # 有一种特殊情况，两个矩阵的shape虽然不一样，但还是可以通过broadcast机制做点乘，比如(1, 3) * (3, 3) = (3, 3)，但是也可以理解为矩阵乘 = (1, 3)，这种情况下，我们默认为矩阵乘。

    if len(x.shape) == 1 and len(y.shape) == 1 or x.shape == y.shape:
        # 点乘
        outputs = Variable(in_bounds=[x, y], data=x.data * y.data, requires_grad=True)
        outputs.grad_fn = MultiplyBackward
    else:
        # 矩阵乘
        outputs = Variable(in_bounds=[x, y], data=np.dot(x.data, y.data), requires_grad=True)
        outputs.grad_fn = MatmulBackward

    initialize_ops_grad(x, y)
    x.out_bounds.append(outputs)
    y.out_bounds.append(outputs)
    return outputs
