from ..nn.core import Layer, Variable
from .activators import get_activator
from xshinnosuke.nn.initializers import get_initializer
from ..nn.functional import concatenate
from ..nn.toolkit import initialize_ops_grad
from ..nn.global_graph import np
from typing import Union, List, Tuple


class Cell:
    def __init__(self, units: int, activation: str = 'tanh', initializer: str = 'glorotuniform',
                 recurrent_initializer: str = 'orthogonal', **kwargs):
        self.units = units
        # 不知道c++支持这样吗，比如有个变量a是由Variable这个类实例化的， 那么a.__class__就返回的是a的类，既Variable，然后使用self.activator_cls()和Variable()等价
        self.activation_cls = get_activator(activation).__class__
        self.initializer = get_initializer(initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)
        self.variables = []  # Variable数组
        self.activations = []  # Activation这个类的数组

    # 以下全都由子类实现
    def initial_params(self, input_shape=None):
        pass

    def forward(self, x: Variable, is_training: bool, stateful: bool):
        pass

    def backward(self, outputs: Variable, inputs: Variable, return_sequences: bool):
        pass

    def reset_state(self, shape):
        pass


class Recurrent(Layer):
    # Base class for Recurrent layer
    def __init__(self, cell: Cell, return_sequences: bool = False, return_state: bool = False, stateful: bool = False,
                 input_length: int = None, **kwargs):
        '''
        :param cell: A Cell object
        :param return_sequences: return all output sequences if true,else return output sequences' last output
        :param return_state:if true ,return last state
        :param stateful:if true，the sequences last state will be used as next sequences initial state
        :param input_length: input sequences' length
        :param kwargs:
        '''
        super(Recurrent, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.input_length = input_length

    def __call__(self, inbound):
        assert len(inbound.shape) == 2, 'Only support batch input'
        super(Recurrent, self).__call__(inbound)
        return self

    def compute_output_shape(self, input_shape: Union[List, Tuple] = None) -> Union[List, Tuple]:
        time_steps, n_vec = input_shape
        if self.return_sequences:
            return tuple([time_steps, self.cell.units])
        return tuple([self.cell.units])

    def initial_params(self, *args):
        time_steps, n_vec = self.input_shape
        self.variables = self.cell.initial_params((n_vec, self.cell.units))

    def forward(self, x: Variable = None, is_training=True, *args):
        if x is not None:
            self.input_data = x
        output = self.cell.forward(self.input_data, is_training, self.stateful)
        if self.return_sequences:
            self.data = output
        else:
            self.data = output[:, -1, :]

        self.connect_init(self.data, is_training)
        if self.return_sequences:
            return self.data, self.cell.prev_a
        return self.data

    def backward(self, gradients=None):
        gradients = self.cell.backward(self.data, self.input_data, self.return_sequences)
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
        Wxa = Variable(self.initializer((n_in, n_out)), name='variable')
        Waa = Variable(self.recurrent_initializer((n_out, n_out)), name='variable')
        ba = Variable(np.zeros(n_out), name='variable')
        self.variables.append(Wxa)
        self.variables.append(Waa)
        self.variables.append(ba)
        return self.variables

    def forward(self, x: Variable, is_training: bool, stateful: bool):
        batch_nums, time_steps, n_vec = x.data.shape
        Wxa, Waa, ba = self.variables
        if is_training:
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
        grad = np.zeros_like(inputs.data)
        if return_sequences:
            da_next = np.zeros_like(self.prev_a[:, 0, :])
            for t in reversed(range(self.time_steps)):
                dz = self.activations[t].backward(outputs.grad[:, t, :] + da_next)

                if Waa.requires_grad:
                    Waa.grad += np.dot(self.prev_a[:, t - 1, :].T, dz)
                if Wxa.requires_grad:
                    Wxa.grad += np.dot(inputs.data[:, t, :].T, dz)
                if ba.requires_grad:
                    ba.grad += np.sum(dz, axis=0)

                da_next = np.dot(dz, Waa.data.T)
                grad[:, t, :] = np.dot(dz, Wxa.data.T)
        else:
            da = outputs.grad
            for t in reversed(range(self.time_steps)):
                da = self.activations[t].backward(da)

                if Waa.requires_grad:
                    Waa.grad += np.dot(self.prev_a[:, t-1, :].T, da)
                if Wxa.requires_grad:
                    Wxa.grad += np.dot(inputs.data[:, t, :].T, da)
                if ba.requires_grad:
                    ba.grad += np.sum(da, axis=0)

                grad[:, t, :] = np.dot(da, Wxa.data.T)
                da = np.dot(da, Waa.data.T)
        return grad

    def reset_state(self, shape):
        self.prev_a = np.zeros(shape)


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
        self.recurrent_activations = []  # Activation这个类的数组
        self.activations = []  # Activation这个类的数组
        self.__first_initialize = True  # bool
        self.time_steps = None  # int
        # 以下都是ndarray，具体看reset_state()
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
        Wf_l = Variable(self.initializer((n_in, n_out)), name='variable')
        Wf_r = Variable(self.recurrent_initializer((n_out, n_out)), name='variable')
        # update gate
        Wu_l = Variable(self.initializer((n_in, n_out)), name='variable')
        Wu_r = Variable(self.recurrent_initializer((n_out, n_out)), name='variable')
        # update unit
        Wc_l = Variable(self.initializer((n_in, n_out)), name='variable')
        Wc_r = Variable(self.recurrent_initializer((n_out, n_out)), name='variable')
        # output gate
        Wo_l = Variable(self.initializer((n_in, n_out)), name='variable')
        Wo_r = Variable(self.recurrent_initializer((n_out, n_out)), name='variable')

        Wf = concatenate(Wf_r, Wf_l, axis=0, name='variable')
        Wu = concatenate(Wu_r, Wu_l, axis=0, name='variable')
        Wc = concatenate(Wc_r, Wc_l, axis=0, name='variable')
        Wo = concatenate(Wo_r, Wo_l, axis=0, name='variable')
        W = concatenate(Wf, Wu, Wc, Wo, axis=1)

        if self.unit_forget_bias:
            bf = Variable(np.ones((1, n_out)), name='variable')
        else:
            bf = Variable(np.zeros((1, n_out)), name='variable')
        bu = Variable(np.zeros((1, n_out)), name='variable')
        bc = Variable(np.zeros((1, n_out)), name='variable')
        bo = Variable(np.zeros((1, n_out)), name='variable')
        b = concatenate(bf, bu, bc, bo, axis=1, name='variable')
        del Wf_r, Wf_l, Wu_r, Wu_l, Wc_r, Wc_l, Wo_r, Wo_l, Wf, Wu, Wc, Wo, bf, bu, bc, bo
        self.variables.append(W)
        self.variables.append(b)
        return self.variables

    def forward(self, x: Variable, is_training: bool, stateful: bool):
        batch_nums, time_steps, n_vec = x.data.shape
        W, b = self.variables
        if is_training:
            self.time_steps = time_steps
            if self.__first_initialize:
                # first intialize prev_a
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
                self.activations = [self.activation_cls() for _ in range(time_steps)]
                self.recurrent_activations = [self.recurrent_activation_cls() for _ in range(3 * time_steps)]
                self.__first_initialize = False
            if stateful:
                self.prev_a[:, 0, :] = self.prev_a[:, -1, :]
                self.c[:, 0, :] = self.c[:, -1, :]
            else:
                self.reset_state(shape=(batch_nums, time_steps + 1, self.units))
        else:
            self.reset_state(shape=(batch_nums, time_steps + 1, self.units))

        z = np.zeros((batch_nums, time_steps, n_vec + self.units))
        for t in range(1, time_steps + 1):
            zt = np.concatenate((self.prev_a[:, t-1, :], x.data[:, t-1, :]), axis=1)
            ot = zt.dot(W.data) + b.data
            f = self.recurrent_activations[3 * (t - 1)].forward(ot[:, :self.units])
            u = self.recurrent_activations[3 * (t - 1) + 1].forward(ot[:, self.units: self.units * 2])
            c_tilde = self.activations[t - 1].forward(ot[:, self.units * 2: self.units * 3])
            o = self.recurrent_activations[3 * (t - 1) + 2].forward(ot[:, self.units * 3:])
            self.c_tilde[:, t-1, :] = c_tilde
            c = f * self.c[:, t-1, :] + u * c_tilde
            self.prev_a[:, t, :] = o * np.tanh(c)

            self.tao_f[:, t-1, :] = f
            self.tao_u[:, t-1, :] = u
            self.tao_o[:, t - 1, :] = o
            self.c[:, t, :] = c
            z[:, t-1, :] = zt
        return self.prev_a[:, 1:, :]

    def backward(self, outputs: Variable, inputs: Variable, return_sequences: bool):
        W, b = self.variables
        grad = np.zeros_like(inputs.data)
        grad = grad[:, :, self.units:]
        if return_sequences:
            da_next = np.zeros_like(self.prev_a[:, 0, :])
            dc_next = np.zeros_like(self.c[:, 0, :])
            for t in reversed(range(self.time_steps)):
                da = outputs.grad[:, t, :] + da_next
                dtao_o = da * np.tanh(self.c[:, t + 1, :])
                do = self.recurrent_activations[3 * (t + 1) - 1].backward(dtao_o)
                dc = dc_next
                dc += da * self.tao_o[:, t, :] * (1 - np.square(np.tanh(self.c[:, t+1, :])))
                dc_tilde = dc * self.tao_u[:, t, :]
                dc_tilde_before_act = self.activations[t].backward(dc_tilde)
                dtao_u = dc * self.c_tilde[:, t, :]
                du = self.recurrent_activations[3 * (t + 1) - 2].backward(dtao_u)
                dtao_f = dc * self.c[:, t, :]
                df = self.recurrent_activations[3 * (t + 1) - 3].backward(dtao_f)
                dgrad = np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
                if W.requires_grad:
                    W.grad += np.dot(inputs.data[:, t, :].T, dgrad)
                if b.requires_grad:
                    b.grad += np.sum(dgrad, axis=0, keepdims=True)

                dz = dgrad.dot(W.data.T)

                da_next = dz[:, :self.units]
                dc_next = dc * self.tao_f[:, t, :]

                grad[:, t, :] = dz[:, self.units:]
        else:
            da_next = np.zeros_like(self.prev_a[:, 0, :])
            dc_next = np.zeros_like(self.c[:, 0, :])
            da = outputs.grad + da_next
            for t in reversed(range(self.time_steps)):
                dtao_o = da * np.tanh(self.c[:, t+1, :])
                do = self.recurrent_activations[3 * (t + 1) - 1].backward(dtao_o)

                dc = dc_next
                dc += da * self.tao_o[:, t, :] * (1 - np.square(np.tanh(self.c[:, t+1, :])))

                dc_tilde = dc * self.tao_u[:, t, :]
                dc_tilde_before_act = self.activations[t].backward(dc_tilde)

                dtao_u = dc * self.c_tilde[:, t, :]
                du = self.recurrent_activations[3 * (t + 1) - 2].backward(dtao_u)

                dtao_f = dc * self.c[:, t, :]
                df = self.recurrent_activations[3 * (t + 1) - 3].backward(dtao_f)

                dgrad = np.concatenate((do, dc_tilde_before_act, du, df), axis=1)
                if W.requires_grad:
                    W.grad += np.dot(inputs.data[:, t, :].T, dgrad)
                if b.requires_grad:
                    b.grad += np.sum(dgrad, axis=0, keepdims=True)

                dz = dgrad.dot(W.data.T)

                da = dz[:, :self.units]
                dc_next = dc * self.tao_f[:, t, :]
                grad[:, t, :] = dz[:, self.units:]

        return grad

    def reset_state(self, shape):
        # timesteps here equals to real timesteps+1
        batch_nums, time_steps, units = shape
        self.prev_a = np.zeros(shape)
        self.c = np.zeros(shape)
        self.tao_f = np.zeros((batch_nums, time_steps - 1, units))
        self.tao_u = np.zeros((batch_nums, time_steps - 1, units))
        self.tao_o = np.zeros((batch_nums, time_steps - 1, units))
        self.c_tilde = np.zeros((batch_nums, time_steps - 1, units))


class LSTM(Recurrent):
    # Fully-connected RNN
    def __init__(self, units: int, activation: str = 'tanh', recurrent_activation: str = 'sigmoid', initializer: str = 'glorotuniform', recurrent_initializer: str = 'orthogonal', unit_forget_bias: bool = True, return_sequences: bool = False, return_state: bool = False, stateful: bool = False, **kwargs):
        '''
        :param units: hidden unit nums
        :param activation:  update unit activation
        :param recurrent_activation: forget gate,update gate,and output gate activation
        :param initializer: same to activation
        :param recurrent_initializer:same to recurrent_activation
        :param unit_forget_bias:if True,add one to the forget gate bias,and force bias initialize as zeros
        :param return_sequences: return sequences or last output
        :param return_state: if True ,return output and last state
        :param stateful: same as SimpleRNN
        :param kwargs:
        '''
        cell = LSTMCell(units=units, activation=activation, recurrent_activation=recurrent_activation, initializer=initializer, recurrent_initializer=recurrent_initializer, unit_forget_bias=unit_forget_bias)
        super(LSTM, self).__init__(cell, return_sequences=return_sequences, return_state=return_state,
                                   stateful=stateful, **kwargs)
