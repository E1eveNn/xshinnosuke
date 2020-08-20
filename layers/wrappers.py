from ..nn.core import Layer, Variable
from ..nn.global_graph import np
from typing import Union, Tuple, List


class TimeDistributed(Layer):
    def __init__(self, layer: Layer, **kwargs):
        super(TimeDistributed, self).__init__(**kwargs)
        self.layer = layer

    def initial_params(self, *args):
        self.layer.initial_params(self.input_shape[1:])
        self.variables = self.layer.variables

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        time_steps = input_shape[0]
        out_shape = self.layer.compute_output_shape(input_shape[1:])
        return (time_steps, ) + out_shape

    def forward(self, x: Variable = None, *args):
        if x is not None:
            self.input_data = x
        time_steps = self.input_shape[0]
        output = Variable(data=np.zeros((self.input_data.data.shape[0], ) + self.shape))
        for t in range(time_steps):
            output[:, t] = self.layer.forward(self.input_data[:, t])
        self.data = output
        self.feed_variable_to_next_layers(self.data)
        return self.data

    def backward(self, gradients=None):
        time_steps = self.input_shape[0]
        for t in range(time_steps):
            self.data.in_bounds[t].grad = self.data.grad[:, t]
            self.layer.data = self.data.in_bounds[t]
            self.layer.backward()
            self.input_data.grad[:, t] = self.layer.data.in_bounds[0].grad
        super().backward(self.input_data.grad)
