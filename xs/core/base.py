from utils.common import GLOBAL, np, Union, List, ndarray, dtype_dict
import core.autograd
import nn.functional as F


class Tensor(object):
    def __new__(cls, *args, **kwargs):
        if len(args) and isinstance(args[0], Tensor):
            return args[0]
        return super().__new__(cls)

    def __init__(self, data: Union[ndarray, int, float, List], requires_grad: bool = False, dtype: str = None,
                 name: str = None, **kwargs):

        if isinstance(data, Tensor):
            return
        self.__data = None
        self.__grad = None
        self.__in_bounds = None
        self.__out_bounds = None
        self.name = None
        self.grad_fn = None
        self.requires_grad = None
        self.device = None
        self.__slice_items = None
        self.__static_graph_tensor = None
        # max storage tensor's shape
        self.__cache = None
        self.__retain_grad = None
        self.__is_leaf = None
        self.next_layers = None
        self.reset_(data, requires_grad, name, dtype, **kwargs)

    def reset_(self, data: Union[ndarray, int, float, List] = None, requires_grad: bool = False, name: str = None, dtype: str = None, **kwargs):
        if data is not None:
            self.__data = np.asarray(data, dtype=dtype)
        self.__grad = kwargs.pop('grad', None)
        self.__in_bounds = []
        self.__out_bounds = []
        self.name = name
        self.grad_fn = None
        self.requires_grad = requires_grad
        self.__slice_items = kwargs.pop('slices', slice(None, None, None))
        self.__static_graph_tensor = kwargs.pop('static_graph_tensor', False)
        # max storage tensor's shape
        self.__cache = {}
        self.__retain_grad = kwargs.pop('retain_grad', False)
        self.__is_leaf = None
        self.next_layers = []

    @property
    def data(self):
        return Tensor(self.__data, slices=self.__slice_items, static_graph_tensor=self.__static_graph_tensor, grad=self.__grad)

    @data.setter
    def data(self, v):
        if isinstance(v, Tensor):
            self.__data[self.__slice_items] = v.eval
        else:
            raise TypeError('Unknown data type {}'.format(type(v)))

    @property
    def eval(self):
        return self.__data if self.__data.size == 1 else self.__data[self.__slice_items]

    @eval.setter
    def eval(self, v):
        self.__data[self.__slice_items] = np.asarray(v)

    @property
    def shape_capacity(self):
        return self.__data.shape

    # truly shape
    @property
    def shape(self):
        return self.eval.shape if len(self.eval.shape) != 0 else (1, )

    @property
    def dtype(self):
        return str(self.__data.dtype)

    def slices(self, slices: slice = None):
        if slices is None:
            return self.__slice_items
        assert isinstance(slices, slice)
        self.__slice_items = slices

    @property
    def is_leaf(self):
        return self.__is_leaf if self.__is_leaf is not None else len(self.__in_bounds) == 0

    @is_leaf.setter
    def is_leaf(self, flag: bool):
        self.__is_leaf = flag

    @property
    def grad(self):
        if self.__grad is not None:
            self.__grad.slices(self.__slice_items)
        return self.__grad

    @grad.setter
    def grad(self, g):
        self.__grad = Tensor(g)

    @property
    def cache(self):
        return self.__cache

    @property
    def is_static(self):
        return self.__static_graph_tensor

    @property
    def is_dynamic(self):
        return not self.__static_graph_tensor

    def retain_grad(self, retain: bool = None):
        if retain is None:
            return self.__retain_grad
        self.__retain_grad = retain

    def to(self, tensor_type: str = 'static'):
        if tensor_type == 'static':
            self.__static_graph_tensor = True
        elif tensor_type == 'dynamic':
            self.__static_graph_tensor = False

    def add_in_bounds(self, *ins):
        for i in ins:
            if isinstance(i, Tensor) or i is None:
                self.__in_bounds.append(i)
            else:
                raise TypeError('Not a Tensor')

    def add_out_bounds(self, *outs):
        for o in outs:
            if isinstance(o, Tensor):
                self.__out_bounds.append(o)
            else:
                raise TypeError('Not a Tensor')

    def requires_grad_(self, flag: bool):
        self.requires_grad = flag

    @property
    def out_bounds(self):
        return self.__out_bounds

    @property
    def in_bounds(self):
        return self.__in_bounds

    def numpy(self):
        return np.asarray(self.eval)

    def free_(self):
        del self.__data, self.__grad, self.__in_bounds[:], self.__out_bounds[:], self.name, self.grad_fn, \
            self.requires_grad, self.device, self.__static_graph_tensor, self.__cache, self.__is_leaf
        del self

    def __getitem__(self, item):
        ret = Tensor(self.data[item], requires_grad=self.requires_grad, slices=item)
        ret.add_out_bounds(self)
        if self.__grad is not None:
            ret.grad = self.__grad[item]
        return ret

    def __setitem__(self, key, value):
        if self.__slice_items is None:
            self.__data[key] = np.asarray(value, dtype=self.__data.dtype)
        else:
            self.__data[self.__slice_items][key] = np.asarray(value, dtype=self.__data.dtype)

    def __repr__(self) -> str:
        requires_grad = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        grad_fn = f', grad_fn=<{self.grad_fn.__name__}>' if self.grad_fn else ""
        info = f'{self.__class__.__name__}({self.__data}{requires_grad}{grad_fn})'
        return info

    def __str__(self) -> str:
        requires_grad = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        grad_fn = f', grad_fn=<{self.grad_fn.__name__}>' if self.grad_fn else ""
        info = f'{self.__class__.__name__}({self.__data}{requires_grad}{grad_fn})'
        return info

    # overload operators
    def __add__(self, other):
        other = Tensor(other)
        return F.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = Tensor(other)
        return F.sub(self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        other = Tensor(other)
        return F.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = Tensor(other)
        return F.mm(self, other)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def dot(self, other):
        return self.__matmul__(other)

    def mm(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        other = Tensor(other)
        return F.truediv(self, other)

    def __floordiv__(self, other):
        other = Tensor(other)
        return F.floordiv(self, other)

    def numel(self):
        return self.__data.size

    def item(self):
        assert self.__data.size == 1
        return self.__data.item()

    def long_(self):
        self.__data = self.__data.astype(np.int64)

    def zero_(self):
        self.__data = np.zeros_like(self.__data)

    def ones_(self):
        self.__data = np.ones_like(self.__data)

    def fill_(self, value):
        self.__data.fill(value)

    def full(self, value):
        return Tensor(np.full(self.shape), value)

    def zero_grad(self):
        if self.__grad is None:
            self.__grad = Tensor(np.zeros(self.shape, dtype=self.dtype))
        else:
            self.__grad.zero_()

    def sum(self, axis: int = None, keepdims: bool = False):
        return F.sum(self, axis, keepdims)

    def mean(self, axis: int = None, keepdims: bool = False):
        return F.mean(self, axis, keepdims)

    def l2norm(self):
        return F.norm(self)

    def t(self):
        return F.transpose(self)

    def sigmoid(self):
        return F.sigmoid(self)

    def view(self, *shapes):
        return F.view(self, shapes)

    def size(self, *dims):
        if len(dims) == 0:
            return self.shape
        elif len(dims) == 1:
            return self.shape[dims[0]]
        return (self.shape[d] for d in dims)

    def backward(self, gradients=None, retain_graph=False):
        if self.grad_fn is None:
            raise ValueError('can not solve grad on %s' % self)
        if gradients is not None:
            assert gradients.size == self.__data.size and gradients.shape == self.shape
            self.__grad = Tensor(np.asarray(gradients, dtype=self.dtype))
        else:
            if self.__data.size == 1:
                if self.__grad is None:
                    self.__grad = Tensor(np.array(1., dtype=self.dtype))
            else:
                raise ValueError('grad can be implicitly created only for scalar outputs')
        GLOBAL.OUTPUTS = self
        # if GLOBAL.INPUTS is not None:
        core.autograd.backward(self, GLOBAL.INPUTS, retain_graph)


class Parameter(Tensor):
    def __new__(cls, *args, **kwargs):
        if len(args) and isinstance(args[0], Parameter):
            return args[0]
        return super().__new__(cls)

    def __init__(self, t: Union[Tensor, ndarray, int, float], **kwargs):
        if isinstance(t, Tensor):
            super(Parameter, self).__init__(t.eval, t.requires_grad, t.dtype, t.name,
                                            **kwargs)
        else:
            super(Parameter, self).__init__(data=t, requires_grad=True)
