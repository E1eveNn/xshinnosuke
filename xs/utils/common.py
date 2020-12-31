import warnings
import copy
from functools import reduce
from typing import *
from core import __global as GLOBAL
import time
import pickle


try:
    np = __import__('cupy')
except ModuleNotFoundError:
    np = __import__('numpy')
    warnings.warn('Looks like you\'re using Numpy, try to install Cupy to gain GPU acceleration!')


ndarray = np.ndarray
dtype_dict = {'int': np.int, 'float': np.float, 'int8': np.int8, 'int16': np.int16, 'int32': np.int32,
                  'int64': np.int64, 'float32': np.float32, 'float64': np.float64}


def overload(func):
    return func


class no_grad:
    def __init__(self):
        self.compute_grad_flag = GLOBAL.COMPUTE_GRAD

    def __enter__(self):
        GLOBAL.COMPUTE_GRAD = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        GLOBAL.COMPUTE_GRAD = self.compute_grad_flag


def initialize_ops_grad(*ops):
    for op in ops:
        if op is not None and op.grad is None and op.requires_grad:
            op.zero_grad()


def im2col(inputs: np.ndarray, out_h: int, out_w: int, kernel_h: int, kernel_w: int, stride: Tuple):
    batch_nums, n_C_prev, n_H_prev, n_W_prev = inputs.shape
    col = np.zeros((batch_nums, n_C_prev, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride[0] * out_h
        for x in range(kernel_w):
            x_max = x + stride[1] * out_w
            col[:, :, y, x, :, :] = inputs[:, :, y:y_max:stride[0], x:x_max:stride[1]]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(batch_nums * out_h * out_w, -1)
    return col


def col2im(inputs_shape: tuple, pad_size: int, kernel_h: int, kernel_w: int, stride: Tuple, dcol: np.ndarray):
    batch_nums, n_C_prev, n_H_prev, n_W_prev = inputs_shape  # 填充前的shape
    n_H = (n_H_prev + 2 * pad_size - kernel_h) // stride[0] + 1
    n_W = (n_W_prev + 2 * pad_size - kernel_w) // stride[1] + 1

    dcol = dcol.reshape((batch_nums, n_H, n_W, n_C_prev, kernel_h, kernel_w)).transpose(0, 3, 4, 5, 1, 2)

    output = np.zeros(
        (batch_nums, n_C_prev, n_H_prev + 2 * pad_size + stride[0] - 1, n_W_prev + 2 * pad_size + stride[1] - 1))

    for y in range(kernel_h):
        y_max = y + stride[0] * n_H
        for x in range(kernel_w):
            x_max = x + stride[1] * n_W
            output[:, :, y:y_max:stride[0], x:x_max:stride[1]] += dcol[:, :, y, x, :, :]

    return output[:, :, pad_size:n_H_prev + pad_size, pad_size:n_W_prev + pad_size]


class ProgressBar:
    def __init__(self,
                 max_iter: int = 1,
                 verbose: int = 1,
                 bar_nums: int = 20,
                 untrained_sign: str = '*',
                 trained_sign: str = '='):
        self.max_iter = max_iter
        self.verbose = verbose
        self._nums = bar_nums - 1
        self._untrained = untrained_sign
        self._trained = trained_sign
        self.iter = 0
        self.times = 0

    def update(self, n_iter: int = 1):
        self.iter += n_iter
        self.times += 1

    def get_bar(self) -> str:
        trained_ratio = self.iter / self.max_iter
        reached_bar_nums = round(trained_ratio * self._nums)
        unreached_bar_nums = self._nums - reached_bar_nums
        if self.verbose == 1:
            bar = reached_bar_nums * self._trained + '>' + unreached_bar_nums * self._untrained
        else:
            percent = str(round(trained_ratio * 100))
            bar = '{black} {percent:>{white}}%'.format(black="\033[40m%s\033[0m" % ' ' * reached_bar_nums,
                                                       percent=percent, white=unreached_bar_nums)
        return bar

    def console(self,
                verbose: int = 0,
                trained_time: float = 0.,
                batch_loss: float = 0.,
                batch_acc: float = 0.,
                validation_loss: float = None,
                validation_acc: float = None):

        if verbose == 0:
            return

        bar = self.get_bar()
        if verbose == 1:
            formated_trained_time = format_time(trained_time)
            formated_per_batch_time = format_time(trained_time / self.times)
            if validation_loss == 0. and validation_acc == 0.:
                print('\r {:d}/{:d} [{}] - {} - {}/batch -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(self.iter,
                                                                                                        self.max_iter,
                                                                                                        bar,
                                                                                                        formated_trained_time,
                                                                                                        formated_per_batch_time,
                                                                                                        batch_loss,
                                                                                                        batch_acc),
                      flush=True, end='')
            else:
                print('\r {:d}/{:d} [{}] - {} - {}/batch'
                      ' -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(
                    self.iter, self.max_iter, bar, formated_trained_time, formated_per_batch_time, batch_loss,
                    batch_acc, validation_loss, validation_acc), flush=True, end='')
        elif verbose == 2:
            sample_time = trained_time / self.iter
            eta = (self.max_iter - self.iter) * sample_time
            formated_eta = format_time(eta)
            if validation_loss == 0. and validation_acc == 0.:
                print('{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f}'.format(bar, formated_eta, batch_loss,
                                                                                 batch_acc))
            else:
                print(
                    '{} -ETA {} -batch_loss: {:.4f} -batch_acc: {:.4f} -validation_loss: {:.4f} -validation_acc: {:.4f}'.format(
                        bar, formated_eta, batch_loss, batch_acc, validation_loss, validation_acc))
        else:
            raise ValueError('Verbose only supports for 0, 1 and 2 ~')


class AverageMeter:
    def __init__(self, name=None, verbose=0):
        self.name = name
        self.val = None
        self.avg = None
        self.sums = None
        self.steps = 0
        self.verbose = verbose
        self.reset()

    def reset(self):
        if self.verbose == 0:
            self.val = 0.
            self.avg = 0.
            self.sums = 0.
        else:
            self.val = []
            self.avg = []
            self.sums = []

    def update(self, val, step=1):
        if val is None:
            self.val = None
            return
        self.steps += step
        if self.verbose == 0:
            self.val = val
            self.sums += val * step
            self.avg = self.sums / self.steps
        else:
            self.val.append(val)
            self.sums.append(self.sums[-1] + val * step)
            self.avg.append(self.sums[-1] / self.steps)


def format_time(second_time: float) -> str:
    if second_time < 1:
        ms = second_time * 1000
        if ms < 1:
            us = second_time * 1000
            return '%dus' % us
        else:
            return '%dms' % ms
    second_time = round(second_time)
    if second_time > 3600:
        # hours
        h = second_time // 3600
        second_time = second_time % 3600
        # minutes
        m = second_time // 60
        second_time = second_time % 60
        return '%dh%dm%ds' % (h, m, second_time)
    elif second_time > 60:
        m = second_time // 60
        second_time = second_time % 60
        return '%dm%ds' % (m, second_time)
    else:
        return '%ds' % second_time


def gradient_check(inputs, target, layer, criterion, epsilon=1e-4):
    variables = layer.parameters()
    variables_mathematical_gradient_list = []
    with no_grad():
        for i in range(len(variables)):
            variable_shape = variables[i].shape
            variable_size = variables[i].numel()
            flat_variables = variables[i].data.reshape(-1, 1)
            mathematical_gradient = np.zeros((variable_size, ))
            new_flat_add_variable = flat_variables.copy()
            new_flat_minus_variable = flat_variables.copy()
            for j in range(variable_size):
                new_flat_add_variable[j] += epsilon
                new_variable = new_flat_add_variable.reshape(variable_shape)
                variables[i].data = new_variable
                layer.set_parameters(variables)
                out = layer(inputs)
                loss1 = criterion(out, target)
                # recover
                new_flat_add_variable[j] -= epsilon

                new_flat_minus_variable[j] -= epsilon
                new_variable = new_flat_minus_variable.reshape(variable_shape)
                variables[i].data = new_variable
                layer.set_parameters(variables)
                out = layer(inputs)
                loss2 = criterion(out, target)
                # recover
                new_flat_minus_variable[j] += epsilon

                mathematical_gradient[j] = (loss1.data - loss2.data) / (2 * epsilon)
            mathematical_gradient = mathematical_gradient.reshape(variable_shape)
            variables_mathematical_gradient_list.append(mathematical_gradient)
    return variables_mathematical_gradient_list