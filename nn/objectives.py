from .core import Variable
from .grad_fn import MeanSquaredBackward, MeanAbsoluteBackward, BinaryCrossEntropyBackward, SparseCrossEntropyBackward, CrossEntropyBackward
import copy
from .functional import softmax
from functools import reduce
from . import global_graph as GlobalGraph


class Objective:
    def __init__(self):
        self.in_bounds = []
        self.grad_fn = None

    def __call__(self, y_pred: Variable, y_true: Variable):
        # if GlobalGraph.outputs is None:
        #     GlobalGraph.outputs = y_pred

        return self.forward(y_pred, y_true)

    def forward(self, y_pred: Variable, y_true: Variable) -> Variable:
        raise NotImplemented

    def acc(self, y_pred: Variable, y_true: Variable):
        raise NotImplemented


class MeanSquaredError(Objective):
    def __init__(self):
        super().__init__()
        self.grad_fn = MeanSquaredBackward

    def acc(self, y_pred: Variable, y_true: Variable):
        return 0.

    def forward(self, y_pred: Variable, y_true: Variable):
        y_pred.retain_grad()
        loss_val = 0.5 * GlobalGraph.np.sum(GlobalGraph.np.power(y_pred.data - y_true.data, 2)) / y_pred.shape[0]
        loss = Variable(data=loss_val, in_bounds=[y_pred, y_true])
        y_pred.out_bounds.append(loss)
        loss.grad_fn = self.grad_fn
        return loss


class MeanAbsoluteError(Objective):
    def __init__(self):
        super().__init__()
        self.grad_fn = MeanAbsoluteBackward

    def acc(self, y_pred: Variable, y_true: Variable):
        return 0.

    def forward(self, y_pred: Variable, y_true: Variable):
        y_pred.retain_grad()
        loss_val = GlobalGraph.np.sum(GlobalGraph.np.absolute(y_pred.data - y_true.data)) / y_pred.data.shape[0]
        loss = Variable(data=loss_val, in_bounds=[y_pred, y_true])
        y_pred.out_bounds.append(loss)
        loss.grad_fn = self.grad_fn
        return loss


class BinaryCrossEntropy(Objective):
    def __init__(self):
        super().__init__()
        self.grad_fn = BinaryCrossEntropyBackward

    def acc(self, y_pred: Variable, y_true: Variable):
        pred = y_pred.data >= 0.5
        return GlobalGraph.np.mean(pred == y_true.data)

    def forward(self, y_pred: Variable, y_true: Variable):
        y_pred.retain_grad()
        loss_val = -GlobalGraph.np.multiply(y_true.data, GlobalGraph.np.log(y_pred.data)) - GlobalGraph.np.multiply(1 - y_true.data, GlobalGraph.np.log(1 - y_pred.data))
        loss_val = GlobalGraph.np.sum(loss_val) / y_pred.data.shape[0]
        loss = Variable(data=loss_val, in_bounds=[y_pred, y_true])
        y_pred.out_bounds.append(loss)
        loss.grad_fn = self.grad_fn
        return loss


class SparseCrossEntropy(Objective):
    # used for one-hot label
    def __init__(self):
        super().__init__()
        self.grad_fn = SparseCrossEntropyBackward

    def __call__(self, y_pred: Variable, y_true: Variable):
        return super().__call__(y_pred, y_true)

    def acc(self, y_pred: Variable, y_true: Variable):
        acc = GlobalGraph.np.argmax(y_pred.data, axis=-1) == GlobalGraph.np.argmax(y_true.data, axis=-1)
        return GlobalGraph.np.mean(acc).tolist()

    def forward(self, y_pred: Variable, y_true: Variable):
        y_pred.retain_grad()
        y_pred = softmax(y_pred)
        avg = GlobalGraph.np.prod(GlobalGraph.np.asarray(y_pred.shape[:-1]))
        loss_val = -GlobalGraph.np.sum(GlobalGraph.np.multiply(y_true.data, GlobalGraph.np.log(y_pred.data))) / avg
        loss = Variable(data=loss_val, in_bounds=[y_pred, y_true])
        y_pred.out_bounds.append(loss)
        loss.grad_fn = self.grad_fn
        return loss


class CrossEntropy(Objective):
    def __init__(self):
        super().__init__()
        self.grad_fn = CrossEntropyBackward

    def __call__(self, y_pred: Variable, y_true: Variable):
        return super().__call__(y_pred, y_true)

    def acc(self, y_pred: Variable, y_true: Variable):
        acc = (GlobalGraph.np.argmax(y_pred.data, axis=-1) == y_true.data)
        return GlobalGraph.np.mean(acc).tolist()

    def forward(self, y_pred: Variable, y_true: Variable):
        y_pred.retain_grad()
        y_true.data = y_true.data.astype(GlobalGraph.np.int64)
        y_pred = softmax(y_pred)
        to_sum_dim = reduce(lambda x, y: x * y, y_pred.data.shape[:-1])
        last_dim = y_pred.data.shape[-1]
        n = y_pred.data.shape[0]
        probs = y_pred.data.reshape(-1, last_dim)
        y_flat = y_true.data.reshape(to_sum_dim)
        loss_val = - GlobalGraph.np.sum(GlobalGraph.np.log(probs[GlobalGraph.np.arange(to_sum_dim), y_flat])) / n
        loss = Variable(data=loss_val, in_bounds=[y_pred, y_true])
        y_pred.out_bounds.append(loss)
        loss.grad_fn = self.grad_fn
        return loss


def get_objective(objective):
    if objective.__class__.__name__ == 'str':
        objective = objective.lower()
        if objective in['crossentropy', 'cross_entropy']:
            return CrossEntropy()
        elif objective in['sparsecrossentropy', 'sparse_crossentropy', 'sparse_cross_entropy']:
            return SparseCrossEntropy()
        elif objective in ['binarycrossentropy', 'binary_cross_entropy', 'binary_crossentropy']:
            return BinaryCrossEntropy()
        elif objective in ['meansquarederror', 'mean_squared_error', 'mse']:
            return MeanSquaredError()
        elif objective in ['meanabsoluteerror', 'mean_absolute_error', 'mae']:
            return MeanAbsoluteError()
    elif isinstance(objective, Objective):
        return copy.deepcopy(objective)
    else:
        raise ValueError('unknown objective type!')
