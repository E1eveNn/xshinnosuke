from ..nn.core import Layer, Variable
from ..nn.global_graph import np
from typing import Union, Tuple, List


class TimeDistributed(Layer):
    # 该层主要用于对时间序列数据做操作的，比如对时间序列使用全连接或者卷积
    def __init__(self, layer: Layer, **kwargs):
        super(TimeDistributed, self).__init__(**kwargs)
        # 所有时间步上的layer是共享的
        self.layer = layer

    def initial_params(self, *args):
        self.layer.initial_params(self.input_shape[1:])
        self.variables = self.layer.variables

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        time_steps = input_shape[0]
        out_shape = self.layer.compute_output_shape(input_shape[1:])
        return (time_steps, ) + out_shape

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        time_steps = self.input_shape[0]
        output = Variable(data=np.zeros((self.input_data.data.shape[0], ) + self.shape))
        for t in range(time_steps):
            output[:, t] = self.layer.forward(self.input_data[:, t])
        self.data = output
        self.connect_init(self.data, is_training)
        return self.data

    def backward(self, gradients=None):
        time_steps = self.input_shape[0]
        for t in range(time_steps):
            self.data.in_bounds[t].grad = self.data.grad[:, t]
            self.layer.data = self.data.in_bounds[t]
            self.layer.backward()
            self.input_data.grad[:, t] = self.layer.data.in_bounds[0].grad
        super().backward(self.input_data.grad)
