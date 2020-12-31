from utils.common import *
import nn.functional as F
import nn.td_functional
from layers.base import Layer, Zeros


class _Loss(Layer):
    def __init__(self, reduction: str = 'mean', **kwargs):
        self.reduction = reduction
        super(_Loss, self).__init__(**kwargs)

    @overload
    def init_layer_out_tensor(self, inputs: F.Tensor, target: F.Tensor):
        if self._data is None:
            if self.reduction == 'mean' or self.reduction == 'sum':
                self._data = Zeros()((1, ))
            else:
                self._data = Zeros()(inputs.shape)
            self._data.add_in_bounds(inputs, target)
            self._data.to('static')
        elif inputs.shape[0] < self._data.shape_capacity[0]:
            self._data.slices(slice(None, inputs.shape[0], None))
        else:
            self._data.slices(None)

    @overload
    def forward(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        self._data = self.call(inputs, target)
        return self._data

    @overload
    def call(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        raise NotImplemented

    def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor) -> float:
        raise NotImplemented

    def calc_loss(self, y_pred: F.Tensor, y_true: F.Tensor) -> float:
        raise NotImplemented

    def metric(self, y_pred: F.Tensor, y_true: F.Tensor) -> Tuple[float, float]:
        acc = self.calc_acc(y_pred, y_true)
        loss = self.calc_loss(y_pred, y_true)
        return acc, loss


class MSELoss(_Loss):
    def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor) -> float:
        return 0.

    def call(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        return F.mse_loss(inputs, target, self.reduction)


class MAELoss(_Loss):
    def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor):
        return 0.

    def call(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        self._data = F.mae_loss(inputs, target, self.reduction, self._data)
        return self._data


class BCELoss(_Loss):
    def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor):
        pred = y_pred.eval >= 0.5
        return np.mean(pred == y_true.eval)

    def call(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        self._data = F.bce_loss(inputs, target, self.reduction, self._data)
        return self._data


class CrossEntropyLoss(_Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    @overload
    def init_layer_out_tensor(self, inputs: F.Tensor, target: F.Tensor):
        if self._data is None or inputs.shape[0] > self._data.shape_capacity[0]:
            if self.reduction == 'mean' or self.reduction == 'sum':
                self._data = Zeros()((1,))
            else:
                self._data = Zeros()(inputs.shape)
            self._data.add_in_bounds(inputs, target)
            inputs.cache['softmax'] = F.Tensor(data=np.empty_like(inputs.eval))
            self._data.to('static')
        elif inputs.shape[0] < self._data.shape_capacity[0]:
            self._data.slices(slice(None, inputs.shape[0], None))
        else:
            self._data.slices(slice(None, None, None))

        if inputs.shape[0] < inputs.cache['softmax'].shape_capacity[0]:
            inputs.cache['softmax'].slices(slice(None, inputs.shape[0], None))
        else:
            inputs.cache['softmax'].slices(slice(None, None, None))

    def calc_loss(self, y_pred: F.Tensor, y_true: F.Tensor) -> float:
        return nn.td_functional.nll_loss(nn.td_functional.log_softmax(y_pred.eval), y_true.eval, self.reduction)

    def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor):
        acc = (np.argmax(y_pred.eval, axis=-1).ravel() == y_true.eval.ravel())
        return np.mean(acc).tolist()

    def call(self, inputs: F.Tensor, target: F.Tensor) -> F.Tensor:
        self._data = F.cross_entropy(inputs, target, self.reduction, self._data)
        return self._data


# class SparseCrossEntropy(Objective):
#     # used for one-hot label
#     def __init__(self):
#         super().__init__()
#         self.grad_fn = SparseCrossEntropyBackward
#
#     def __call__(self, y_pred: F.Tensor, y_true: F.Tensor):
#         return super().__call__(y_pred, y_true)
#
#     def calc_acc(self, y_pred: F.Tensor, y_true: F.Tensor):
#         calc_acc = np.argmax(y_pred.data, axis=-1) == np.argmax(y_true.data, axis=-1)
#         return np.mean(calc_acc).tolist()
#
#     def calc_loss(self, y_pred: F.Tensor, y_true: F.Tensor):
#         return -np.sum(
#             np.multiply(y_true.data, np.log(y_pred.data))) / np.prod(
#             np.asarray(y_pred.shape[:-1]))
#
#     def forward(self, y_pred: F.Tensor, y_true: F.Tensor):
#         y_pred.retain_grad()
#         y_pred = softmax(y_pred)
#         loss_val = self.calc_loss(y_pred, y_true)
#         loss = F.Tensor(data=loss_val, in_bounds=[y_pred, y_true])
#         y_pred.out_bounds.append(loss)
#         loss.grad_fn = self.grad_fn
#         return loss


def get_objective(objective):
    if objective.__class__.__name__ == 'str':
        objective = objective.lower()
        if objective in ['crossentropy', 'cross_entropy']:
            return CrossEntropyLoss()
        # elif objective in ['sparsecrossentropy', 'sparse_crossentropy', 'sparse_cross_entropy']:
        #     return SparseCrossEntropy()
        elif objective in ['bce', 'binary_cross_entropy', 'binary_crossentropy']:
            return BCELoss()
        elif objective in ['meansquarederror', 'mean_squared_error', 'mse']:
            return MSELoss()
        elif objective in ['meanabsoluteerror', 'mean_absolute_error', 'mae']:
            return MAELoss()
    elif isinstance(objective, _Loss):
        return copy.deepcopy(objective)
    else:
        raise ValueError('unknown objective type!')
