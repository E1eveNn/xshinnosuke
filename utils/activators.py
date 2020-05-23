from nn.core import Layer, Variable
from nn.grad_fn import ReluBackward, SigmoidBackward, TanhBackward
from nn.functional import relu, sigmoid, tanh


class ReLU(Layer):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            output = relu(inbound, self.inplace)
            return output
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = relu(self.input_data, inplace=self.inplace)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, output: Variable = None):
        if self.inplace:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        if output is not None:
            self.data = output
        ReluBackward(self.data)


class Linear(Layer):
    def __call__(self, inbound):
        if isinstance(inbound, Variable):
            return inbound
        super().__call__(inbound)
        return self

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = self.input_data
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, output: Variable = None):
        if output is not None:
            self.data = output
        self.input_data.grad = self.data.grad
        return self.input_data


class Sigmoid(Layer):
    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = sigmoid(self.input_data)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, output: Variable = None):
        if output is not None:
            self.data = output
        SigmoidBackward(self.data)


class Tanh(Layer):
    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        self.data = tanh(self.input_data)
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, output: Variable = None):
        if output is not None:
            self.data = output
        TanhBackward(self.data)


def get_activator(activator):
    if activator.__class__.__name__ == 'str':
        activator = activator.lower()
        if activator == 'relu':
            return ReLU()
        elif activator == 'linear':
            return Linear()
        elif activator == 'sigmoid':
            return Sigmoid()
        elif activator == 'tanh':
            return Tanh()
