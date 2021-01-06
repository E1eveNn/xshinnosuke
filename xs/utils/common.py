import warnings
import copy
from functools import reduce
from typing import *
from core import __global as GLOBAL
import time


ndarray = GLOBAL.np.ndarray
dtype_dict = {'int': GLOBAL.np.int, 'float': GLOBAL.np.float, 'int8': GLOBAL.np.int8, 'int16': GLOBAL.np.int16, 'int32': GLOBAL.np.int32,
                  'int64': GLOBAL.np.int64, 'float32': GLOBAL.np.float32, 'float64': GLOBAL.np.float64}


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


def im2col(inputs: GLOBAL.np.ndarray, out_h: int, out_w: int, kernel_h: int, kernel_w: int, stride: Tuple):
    batch_nums, n_C_prev, n_H_prev, n_W_prev = inputs.shape
    col = GLOBAL.np.zeros((batch_nums, n_C_prev, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride[0] * out_h
        for x in range(kernel_w):
            x_max = x + stride[1] * out_w
            col[:, :, y, x, :, :] = inputs[:, :, y:y_max:stride[0], x:x_max:stride[1]]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(batch_nums * out_h * out_w, -1)
    return col


def col2im(inputs_shape: tuple, pad_size: int, kernel_h: int, kernel_w: int, stride: Tuple, dcol: GLOBAL.np.ndarray):
    batch_nums, n_C_prev, n_H_prev, n_W_prev = inputs_shape  # 填充前的shape
    n_H = (n_H_prev + 2 * pad_size - kernel_h) // stride[0] + 1
    n_W = (n_W_prev + 2 * pad_size - kernel_w) // stride[1] + 1

    dcol = dcol.reshape((batch_nums, n_H, n_W, n_C_prev, kernel_h, kernel_w)).transpose(0, 3, 4, 5, 1, 2)

    output = GLOBAL.np.zeros(
        (batch_nums, n_C_prev, n_H_prev + 2 * pad_size + stride[0] - 1, n_W_prev + 2 * pad_size + stride[1] - 1))

    for y in range(kernel_h):
        y_max = y + stride[0] * n_H
        for x in range(kernel_w):
            x_max = x + stride[1] * n_W
            output[:, :, y:y_max:stride[0], x:x_max:stride[1]] += dcol[:, :, y, x, :, :]

    return output[:, :, pad_size:n_H_prev + pad_size, pad_size:n_W_prev + pad_size]

