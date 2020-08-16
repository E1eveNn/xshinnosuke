from .nn.objectives import get_objective
from .nn.optimizers import get_optimizer
from .utils.toolkit import *
from .nn.core import Variable, Layer
from .nn.functional import concatenate
from .nn import global_graph as GlobalGraph
import time
import pickle
from typing import Tuple


class Base:
    def __init__(self):
        self.variables = set()

    def train(self):
        GlobalGraph.IS_TRAINING = True

    def eval(self):
        GlobalGraph.IS_TRAINING = False


class _Model(Base):
    def __init__(self):
        super().__init__()
        self.loss = None  # Variable
        self.optimizer = None  # Optimizer类型
        self.graph = None  # Layer数组

    def compile(self, optimizer, loss):
        raise NotImplemented

    def fit(self,
            x: GlobalGraph.np.ndarray,
            y: GlobalGraph.np.ndarray,
            batch_size: int = None,
            epochs: int = 1,
            verbose: int = 1,
            shuffle: bool = True,
            validation_data: Tuple[GlobalGraph.np.ndarray] = None,
            validation_split: float = 0.,
            initial_epoch: int = 0,
            ) -> dict:

        x = GlobalGraph.np.asarray(x)
        y = GlobalGraph.np.asarray(y)
        if validation_data is None and 0. < validation_split < 1.:
            split = int(x.shape[0] * validation_split)
            valid_x, valid_y = x[-split:], y[-split:]
            train_x, train_y = x[:-split], y[:-split]
            validation_data = (valid_x, valid_y)
        else:
            train_x, train_y = x, y

        history = dict()

        train_dataset = DataSet(train_x, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

        for epoch in range(initial_epoch, epochs):
            epoch_time = AverageMeter(name='epoch_time')
            train_loss = AverageMeter(name='train_loss')
            train_acc = AverageMeter(name='train_acc')
            valid_loss = AverageMeter(name='validation_loss')
            valid_acc = AverageMeter(name='validation_acc')
            history[epoch] = {
                'epoch_time': epoch_time,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'validation_loss': valid_loss,
                'validation_acc': valid_acc
            }
            start_time = time.time()

            if verbose != 0:
                print('\033[1;31m Epoch[%d/%d]\033[0m' % (epoch + 1, epochs))
            progress_bar = ProgressBar(max_iter=len(train_x), verbose=verbose)

            for idx, (xs, ys) in enumerate(train_dataloader):
                self.train()
                # reset trainable_variables grad
                self.optimizer.zero_grad()
                # forward
                pred = self.forward(xs, is_training=True)
                loss = self.loss.forward(pred, ys)
                self.backward(loss)
                self.optimizer.step()

                epoch_time.update(time.time() - start_time)
                train_loss.update(loss.data.tolist())
                train_acc.update(self.loss.acc(pred, ys))

                if validation_data is not None:
                    valid_x, valid_y = validation_data
                    v_acc, v_loss = self.evaluate(valid_x, valid_y, batch_size=batch_size)
                    valid_loss.update(v_loss)
                    valid_acc.update(v_acc)

                progress_bar.update(xs.shape[0])
                progress_bar.console(verbose, epoch_time.val, train_loss.val, train_acc.val, valid_loss.val,
                                     valid_acc.val)

            print()
        return history

    def __call__(self, x: Variable, *args, **kwargs):
        out = self.forward(x)
        return out

    def forward(self, x, *args, **kwargs):
        raise NotImplemented

    def backward(self, loss: Variable):
        loss.grad_fn(loss)
        for layer in reversed(self.graph):
            layer.backward()

    def evaluate(self, x: GlobalGraph.np.ndarray, y: GlobalGraph.np.ndarray, batch_size: int = None):
        self.eval()
        x = GlobalGraph.np.asarray(x)
        y = GlobalGraph.np.asarray(y)
        if batch_size is not None:
            assert type(batch_size) is int
            val_dataset = DataSet(x, y)
            val_dataloader = DataLoader(val_dataset, batch_size)
            acc_list = []
            loss_list = []
            for xs, ys in val_dataloader:
                y_pred = self.forward(xs)
                metric = self.loss.metric(y_pred, ys)
                acc_list.append(metric[0])
                loss_list.append(metric[1])

            acc = GlobalGraph.np.array(acc_list).mean().tolist()
            loss = GlobalGraph.np.array(loss_list).mean().tolist()
        else:
            y_pred = self.forward(Variable(x))
            acc, loss = self.loss.metric(y_pred, y)

        return acc, loss

    def predict(self, x: GlobalGraph.np.ndarray, batch_size: int = None):
        self.eval()
        if batch_size is not None:
            assert type(batch_size) is int
            test_dataset = DataSet(x)
            test_dataloader = DataLoader(test_dataset, batch_size)
            pred_list = []
            for xs in test_dataloader:
                y_pred = self.forward(xs)
                pred_list.append(y_pred)
            pred = concatenate(*pred_list, axis=0)
        else:
            pred = self.forward(Variable(x))

        return pred

    def save(self, save_path):
        if not save_path.endswith('.pkl'):
            save_path += '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([self.graph, self.optimizer, self.loss], f)

    def load(self, model_path):
        if not model_path.endswith('.pkl'):
            model_path += '.pkl'
        with open(model_path, 'rb') as f:
            graph, optimizer, loss = pickle.load(f)

        self.graph = graph
        self.optimizer = optimizer
        self.loss = loss

    def __str__(self):
        bar_nums = 75
        print('*' * bar_nums)
        print('Layer(type)'.ljust(25), 'Output Shape'.ljust(20), 'Param'.ljust(10), 'Connected to'.ljust(15))
        print('#' * bar_nums)
        if self.graph is None:
            raise ValueError('Please compile Model!')
        for layer in self.graph:
            layer_name = '%s (%s)' % (layer.name, layer.__class__.__name__)
            params = layer.params_count()
            first = True
            if layer.in_bounds:
                for prev_layer in layer.in_bounds:
                    if prev_layer.name is not None:
                        connected = prev_layer.name
                    else:
                        connected = prev_layer.__class__.__name__
                    if first:
                        print(layer_name.ljust(25), str((None,) + layer.shape).ljust(20), str(params).ljust(10),
                              connected.ljust(15))
                        first = False
                    else:
                        print(''.ljust(25), ''.ljust(20), ''.ljust(10), connected.ljust(15))
            else:
                connected = '\n'
                print(layer_name.ljust(25), str((None,) + layer.shape).ljust(20), str(params).ljust(10),
                      connected.ljust(15))
            print('-' * bar_nums)

        print('*' * bar_nums)
        total_params = 0
        trainable_params = 0
        for v in self.variables:
            total_params += v.data.size
            if v.requires_grad:
                trainable_params += v.data.size
        params_details = 'Total params: %d\n' % total_params
        params_details += 'Trainable params: %d\n' % trainable_params
        params_details += 'Non-trainable params: %d\n' % (total_params - trainable_params)
        return params_details


class Sequential(_Model):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.graph = [] if layers is None else list(layers)

    def add(self, layer):
        self.graph.append(layer)

    def compile(self, optimizer, loss, **kwargs):
        assert self.graph
        next_layer = None
        for layer in self.graph:
            layer.connect(next_layer)
            layer.initial_params()
            next_layer = layer
            for v in layer.variables:
                if v is not None:
                    self.variables.add(v)
        self.loss = get_objective(loss)
        self.optimizer = get_optimizer(optimizer, variables=self.variables, **kwargs)

    def forward(self, x, *args, **kwargs):
        for layer in self.graph:
            if isinstance(layer, Layer) and len(layer.variables) == 0:
                layer.initial_params(x.shape[1:])
            x = layer.forward(x)
        return x

    def pop(self, index=-1):
        layer = self.graph.pop(index)
        del layer
        print('success delete %s layer' % layer.__class__.__name__)


class Model(_Model):
    def __init__(self, inputs=None, outputs=None):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss, **kwargs):
        assert self.inputs is not None and self.outputs is not None
        self.graph = GlobalGraph.topological_sort(self.inputs, self.outputs)
        for g in self.graph:
            g.initial_params()
            for v in g.variables:
                if v is not None:
                    self.variables.add(v)
        self.loss = get_objective(loss)
        self.optimizer = get_optimizer(optimizer, variables=self.variables, **kwargs)

    def forward(self, x: Variable, *args, **kwargs):

        self.inputs.input_data = x
        outputs = None
        for layer in self.graph:
            outputs = layer.forward()
        return outputs


class Module(Base):
    def __init__(self, *args):
        super().__init__()

    def __call__(self, x: Variable, *args, **kwargs):
        out = self.forward(x)
        self.__collect_variables(x)
        return out

    def __collect_variables(self, x: Variable):
        # 用一个列表来模拟队列，然后bfs遍历统计所有的trainable_variables
        queue = [x]
        seen = set()  # 此处为set, python里set用的是hash table, 搜索时比数组要快。
        seen.add(x)
        while queue:
            # 队列非空，队首元素出列
            vertex = queue.pop(0)
            # 将该元素的邻接元素加入队尾
            for n in vertex.out_bounds:
                if n not in seen:
                    for v in n.get_variables():
                        if v is not None:
                            self.variables.add(v)
                    queue.append(n)
                    seen.add(n)

    def parameters(self):
        return self.variables

    def forward(self, x):
        raise NotImplemented
