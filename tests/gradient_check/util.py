import numpy as np
import matplotlib.pyplot as plt


def draw_grad_line(torch_grad: np.ndarray, xs_grad: np.ndarray, name: str):
    plt.figure()
    plt.plot(torch_grad.ravel(), label=f'pytorch_{name}')
    plt.plot(xs_grad.ravel(), label=f'xs_{name}')
    plt.legend(loc='best')
    plt.show()


def compute_euclidean_distance(torch_grad: np.ndarray, xs_grad: np.ndarray):
    return np.sqrt(np.sum(np.square(torch_grad.ravel() - xs_grad.ravel())))
