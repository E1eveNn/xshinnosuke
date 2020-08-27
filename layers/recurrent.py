from ..nn.core import Layer, Variable
from .activators import get_activator
from ..nn.initializers import get_initializer, ones
from ..nn.functional import concatenate, lstm, LstmBackward
from ..nn.toolkit import initialize_ops_grad
from ..nn import global_graph as GlobalGraph
from typing import Union, List, Tuple


class Cell:
    def __init__(self, units: int, activation: str = 'tanh', initializer: str = 'glorotuniform',
                 recurrent_initializer: str = 'orthogonal', **kwargs):
        self.units = units
        self.activation_cls = get_activator(activation).__class__
        self.initializer = get_initializer(initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)
        self.variables = []
        self.activations = []

    def initial_params(self, input_shape=None):
        pass

    def forward(self, x: Variable, stateful: bool, return_sequences: bool):
        raise NotImplemented

    def backward(self, outputs: Variable):
        pass

    def reset_state(self, shape):
        pass


class Recurrent(Layer):
    # Base class for Recurrent layer
    def __init__(self, cell: Cell, return_sequences: bool = False, return_state: bool = False, stateful: bool = False,
                 input_length: int = None, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.input_length = input_length

    def __call__(self, inbound, *args, **kwargs):
        # assert len(inbound.shape) == 2, 'Only support batch input'
        if isinstance(inbound, Variable):
            if len(self.variables) == 0:
                self.initial_params(inbound.shape[1:])
            return self.forward(inbound)
        super(Recurrent, self).__call__(inbound)
        return self

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        time_steps, n_vec = input_shape
        if self.return_sequences:
            return tuple([time_steps, self.cell.units])
        return tuple([self.cell.units])

    def initial_params(self, input_shape=None):
        if input_shape is not None:
            self.input_shape = input_shape
        time_steps, n_vec = self.input_shape
        self.variables = self.cell.initial_params((n_vec, self.cell.units))

    def forward(self, x: Variable = None, *args):
        if x is not None:
            self.input_data = x
        self.data = self.cell.forward(self.input_data, self.stateful, self.return_sequences)
        self.feed_variable_to_next_layers(self.data)
        if self.return_sequences:
            return self.data, self.cell.prev_a
        return self.data

    def backward(self, gradients=None):
        gradients = self.cell.backward(self.data)
        super().backward(gradients)


class SimpleRNNCell(Cell):
    # cell class for SimpleRNN
    def __init__(self, units: int, activation: str = 'tanh', initializer: str = 'glorotuniform',
                 recurrent_initializer: str = 'orthogonal',  **kwargs):
        super(SimpleRNNCell, self).__init__(units=units, activation=activation, initializer=initializer,
                                            recurrent_initializer=recurrent_initializer)

        self.__first_initialize = True  # bool
        self.prev_a = None  # ndarray
        self.time_steps = None  # int

    def initial_params(self, input_shape=None):
        n_in, n_out = input_shape
        Wxa = self.initializer((n_in, n_out), name='xs_variable')
        Waa = self.recurrent_initializer((n_out, n_out), name='xs_variable')
        ba = ones(n_out, name='xs_variable')
        self.variables.append(Wxa)
        self.variables.append(Waa)
        self.variables.append(ba)
        return self.variables

    def forward(self, x: Variable, stateful: bool):
        batch_nums, time_steps, n_vec = x.data.shape
        Wxa, Waa, ba = self.variables
        if GlobalGraph.IS_TRAINING:
            self.time_steps = time_steps
            if self.__first_initialize:
                # first initialize prev_a
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
                self.activations = [self.activation_cls() for _ in range(time_steps)]
                self.__first_initialize = False

            if stateful:
                self.prev_a[:, 0, :] = self.prev_a[:, -1, :]
            else:
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
        else:
            self.reset_state(shape=(batch_nums, time_steps + 1, self.units))

        for t in range(1, time_steps + 1):
            self.prev_a[:, t, :] = self.activations[t-1].forward(x.data[:, t-1, :].dot(Wxa.data) +
                                                                 self.prev_a[:, t-1, :].dot(Waa.data) + ba.data)

        initialize_ops_grad(self.variables)
        return self.prev_a[:, 1:, :]

    def backward(self, outputs: Variable, inputs: Variable, return_sequences: bool):
        Wxa, Waa, ba = self.variables
        grad = GlobalGraph.np.zeros_like(inputs.data)
        if return_sequences:
            da_next = GlobalGraph.np.zeros_like(self.prev_a[:, 0, :])
            for t in reversed(range(self.time_steps)):
                dz = self.activations[t].backward(outputs.grad[:, t, :] + da_next)

                if Waa.requires_grad:
                    Waa.grad += GlobalGraph.np.dot(self.prev_a[:, t - 1, :].T, dz)
                if Wxa.requires_grad:
                    Wxa.grad += GlobalGraph.np.dot(inputs.data[:, t, :].T, dz)
                if ba.requires_grad:
                    ba.grad += GlobalGraph.np.sum(dz, axis=0)

                da_next = GlobalGraph.np.dot(dz, Waa.data.T)
                grad[:, t, :] = GlobalGraph.np.dot(dz, Wxa.data.T)
        else:
            da = outputs.grad
            for t in reversed(range(self.time_steps)):
                da = self.activations[t].backward(da)

                if Waa.requires_grad:
                    Waa.grad += GlobalGraph.np.dot(self.prev_a[:, t-1, :].T, da)
                if Wxa.requires_grad:
                    Wxa.grad += GlobalGraph.np.dot(inputs.data[:, t, :].T, da)
                if ba.requires_grad:
                    ba.grad += GlobalGraph.np.sum(da, axis=0)

                grad[:, t, :] = GlobalGraph.np.dot(da, Wxa.data.T)
                da = GlobalGraph.np.dot(da, Waa.data.T)
        return grad

    def reset_state(self, shape):
        self.prev_a = GlobalGraph.np.zeros(shape)


class SimpleRNN(Recurrent):
    # Fully-connected RNN
    def __init__(self, units: int, activation: str = 'tanh', initializer: str = 'glorotuniform',
                 recurrent_initializer: str = 'orthogonal', return_sequences: bool = False, return_state: bool = False,
                 stateful: bool = False, **kwargs):
        cell = SimpleRNNCell(units=units, activation=activation, initializer=initializer,
                             recurrent_initializer=recurrent_initializer)
        super(SimpleRNN, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,
                                        stateful=stateful, **kwargs)


class LSTMCell(Cell):
    # cell class for SimpleRNN
    def __init__(self, units: int, activation: str = 'tanh', recurrent_activation: str = 'sigmoid',
                 initializer: str = 'glorotuniform', recurrent_initializer: str = 'orthogonal', unit_forget_bias: bool = True, **kwargs):
        super(LSTMCell, self).__init__(units=units, activation=activation, initializer=initializer,
                                       recurrent_initializer=recurrent_initializer)

        self.recurrent_activation_cls = get_activator(recurrent_activation).__class__
        self.unit_forget_bias = unit_forget_bias
        self.recurrent_activations = []
        self.activations = []
        self.__first_initialize = True  # bool
        self.time_steps = None  # int

        self.prev_a = None
        self.c = None
        self.tao_f = None
        self.tao_u = None
        self.tao_o = None
        self.c_tilde = None

    def initial_params(self, input_shape=None):
        n_in, n_out = input_shape
        # Wf_l means forget gate linear weight,Wf_r represents forget gate recurrent weight.
        # forget gate
        Wf_l = self.initializer((n_in, n_out))
        Wf_r = self.recurrent_initializer((n_out, n_out))
        # update gate
        Wu_l = self.initializer((n_in, n_out))
        Wu_r =self.recurrent_initializer((n_out, n_out))
        # update unit
        Wc_l = self.initializer((n_in, n_out))
        Wc_r = self.recurrent_initializer((n_out, n_out))
        # output gate
        Wo_l = self.initializer((n_in, n_out))
        Wo_r = self.recurrent_initializer((n_out, n_out))

        Wf = concatenate(Wf_r, Wf_l, axis=0, name='variable')
        Wu = concatenate(Wu_r, Wu_l, axis=0, name='variable')
        Wc = concatenate(Wc_r, Wc_l, axis=0, name='variable')
        Wo = concatenate(Wo_r, Wo_l, axis=0, name='variable')
        W = concatenate(Wf, Wu, Wc, Wo, axis=1, name='xs_variable')
        if self.unit_forget_bias:
            bf = Variable(GlobalGraph.np.ones((1, n_out)), name='variable')
        else:
            bf = Variable(GlobalGraph.np.zeros((1, n_out)), name='variable')
        bu = Variable(GlobalGraph.np.zeros((1, n_out)), name='variable')
        bc = Variable(GlobalGraph.np.zeros((1, n_out)), name='variable')
        bo = Variable(GlobalGraph.np.zeros((1, n_out)), name='variable')
        b = concatenate(bf, bu, bc, bo, axis=1, name='xs_variable')

        del Wf_r, Wf_l, Wu_r, Wu_l, Wc_r, Wc_l, Wo_r, Wo_l, Wf, Wu, Wc, Wo, bf, bu, bc, bo
        self.variables.append(W)
        self.variables.append(b)
        return self.variables

    def forward(self, x: Variable, stateful: bool, return_sequences: bool):
        weight, bias = self.variables
        batch_nums, time_steps, n_vec = x.data.shape
        if GlobalGraph.IS_TRAINING:
            if self.__first_initialize:
                # first intialize prev_a
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
                self.activations = [self.activation_cls() for _ in range(time_steps)]
                self.recurrent_activations = [self.recurrent_activation_cls() for _ in range(3 * time_steps)]
                self.__first_initialize = False
            if stateful:
                self.prev_a.data[:, 0, :] = self.prev_a.data[:, -1, :]
                self.c.data[:, 0, :] = self.c.data[:, -1, :]
            else:
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
        else:
            self.reset_state(shape=(batch_nums, time_steps + 1, self.units))

        return lstm(x, weight, bias, self.units, self.recurrent_activations, self.activations, self.prev_a, self.c,
                    self.tao_f, self.tao_u, self.tao_o, self.c_tilde, GlobalGraph.IS_TRAINING, return_sequences)

    def backward(self, outputs: Variable):
        LstmBackward(outputs)

    def reset_state(self, shape):
        # timesteps here equals to real timesteps + 1
        batch_nums, time_steps, units = shape
        self.prev_a = Variable(GlobalGraph.np.zeros(shape))
        self.c = Variable(GlobalGraph.np.zeros(shape))
        self.tao_f = Variable(GlobalGraph.np.zeros((batch_nums, time_steps - 1, units)))
        self.tao_u = Variable(GlobalGraph.np.zeros((batch_nums, time_steps - 1, units)))
        self.tao_o = Variable(GlobalGraph.np.zeros((batch_nums, time_steps - 1, units)))
        self.c_tilde = Variable(GlobalGraph.np.zeros((batch_nums, time_steps - 1, units)))


class LSTM(Recurrent):
    # Fully-connected RNN
    def __init__(self, units: int, activation: str = 'tanh', recurrent_activation: str = 'sigmoid', initializer: str = 'glorotuniform', recurrent_initializer: str = 'orthogonal', unit_forget_bias: bool = True, return_sequences: bool = False, return_state: bool = False, stateful: bool = False, **kwargs):
        cell = LSTMCell(units=units, activation=activation, recurrent_activation=recurrent_activation, initializer=initializer, recurrent_initializer=recurrent_initializer, unit_forget_bias=unit_forget_bias)
        super(LSTM, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,
                                   stateful=stateful, **kwargs)
