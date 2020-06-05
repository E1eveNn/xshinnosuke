from .nn.objectives import get_objective
from .nn.optimizers import get_optimizer
from .utils.toolkit import *
from .nn.core import Variable, Layer
from .nn.functional import concatenate
from .nn.global_graph import np, topological_sort
import time
import pickle
from typing import Tuple


class Module:
    def __init__(self, *args):
        self.trainable_variables = []  # Variable数组
        self.loss = None  # Variable
        self.optimizer = None  # Optimizer类型
        self.graph = None  # Layer数组
        self.is_training = True  # 布尔

    def __call__(self, x: Variable, *args, **kwargs):
        out = self.forward(x)
        self.__add_trainable_variables(x)
        return out

    def __add_trainable_variables(self, x: Variable):
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
                    for v in n.in_bounds:
                        if v is not None and v.requires_grad and v.name == 'variable'and v not in self.trainable_variables:
                            self.trainable_variables.append(v)

                    queue.append(n)
                    seen.add(n)

    def parameters(self):
        return self.trainable_variables

    def compile(self, optimizer, loss):
        raise NotImplemented

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 64,
            epochs: int = 1,
            verbose: int = 1,
            shuffle: bool = True,
            validation_data: Tuple[np.ndarray] = None,
            validation_split: float = 0.,
            initial_epoch: int = 0,
            ) -> dict:

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
                xs = Variable(xs)
                ys = Variable(ys)
                self.train()
                # reset trainable_variables grad
                self.optimizer.zero_grad()
                # forward
                pred = self.forward(xs)
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

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = None):
        self.eval()
        if batch_size is not None:
            assert type(batch_size) is int
            val_dataset = DataSet(x, y)
            val_dataloader = DataLoader(val_dataset, batch_size)
            acc_list = []
            loss_list = []
            for xs, ys in val_dataloader:
                xs = Variable(xs)
                ys = Variable(ys)
                y_pred = self.forward(xs)
                metric = self.loss.metric(y_pred, ys)
                acc_list.append(metric[0])
                loss_list.append(metric[1])

            acc = np.mean(acc_list).tolist()
            loss = np.mean(loss_list).tolist()
        else:
            x = Variable(x)
            y = Variable(y)
            y_pred = self.forward(x)
            acc, loss = self.loss.metric(y_pred, y)

        return acc, loss

    def predict(self, x: np.ndarray, batch_size: int = None):
        self.eval()
        if batch_size is not None:
            assert type(batch_size) is int
            test_dataset = DataSet(x)
            test_dataloader = DataLoader(test_dataset, batch_size)
            pred_list = []
            for xs in test_dataloader:
                xs = Variable(xs)
                y_pred = self.forward(xs)
                pred_list.append(y_pred)
            pred = concatenate(*pred_list, axis=0)
        else:
            pred = self.forward(x)

        return pred

    def forward(self, x, *args):
        raise NotImplemented

    def backward(self, loss: Variable):
        loss.backward()
        for layer in reversed(self.graph):
            layer.backward()

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
        bar_nums = 120
        print('*' * bar_nums)

        print('Layer(type)'.ljust(40), 'Output Shape'.ljust(40), 'Param'.ljust(12), 'Connected to'.ljust(15))
        print('#' * bar_nums)
        total_params = 0
        for layer in self.graph:
            if layer.name is not None:
                layer_name = '%s (%s)' % (layer.name, layer.__class__.__name__)
            else:
                layer_name = str(layer.__class__.__name__)

            params = layer.params_count()
            total_params += params
            first = True
            if layer.in_bounds:
                for prev_layer in layer.in_bounds:
                    if prev_layer.name is not None:
                        connected = prev_layer.name
                    else:
                        connected = prev_layer.__class__.__name__
                    if first:
                        print(layer_name.ljust(40), str((None, ) + layer.shape).ljust(40), str(params).ljust(12),
                              connected.ljust(15))
                        first = False
                    else:
                        print(''.ljust(40), ''.ljust(40), ''.ljust(12), connected.ljust(15))
            else:
                connected = '\n'
                print(layer_name.ljust(40), str((None, ) + layer.shape).ljust(40), str(params).ljust(12),
                      connected.ljust(15))
            print('-' * bar_nums)

        print('*' * bar_nums)
        trainable_params = 0
        for v in self.trainable_variables:
            trainable_params += v.data.size
        params_details = 'Total params: %d\n' % total_params
        params_details += 'Trainable params: %d\n' % trainable_params
        params_details += 'Non-trainable params: %d\n' % (total_params - trainable_params)
        return params_details


class Sequential(Module):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.graph = [] if layers is None else layers

    def add(self, layer):
        self.graph.append(layer)

    def compile(self, optimizer, loss):
        assert self.graph
        next_layer = None
        for layer in self.graph:
            layer.connect(next_layer)
            layer.initial_params()
            next_layer = layer
            for v in layer.variables:
                if v is not None and v.requires_grad and v not in self.trainable_variables:
                    self.trainable_variables.append(v)
        self.loss = get_objective(loss)
        self.optimizer = get_optimizer(optimizer)
        self.optimizer.trainable_variables = self.trainable_variables

    def forward(self, x, *args):
        for layer in self.graph:
            if hasattr(layer, 'variables') and len(layer.variables) == 0:
                layer.initial_params(x.shape[1:])
            x = layer.forward(x, self.is_training)
        return x

    def pop(self, index=-1):
        layer = self.graph.pop(index)
        del layer
        print('success delete %s layer' % layer.__class__.__name__)

    def save(self, save_path):
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump([self.graph, self.optimizer, self.loss], f)

    def load(self, model_path):
        with open(model_path + '.pkl', 'rb') as f:
            layers, optimizer, loss = pickle.load(f)

        self.graph = layers
        self.optimizer = optimizer
        self.loss = loss


class Model(Module):
    def __init__(self, inputs=None, outputs=None):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss, **kwargs):
        assert self.inputs is not None and self.outputs is not None
        self.graph = topological_sort(self.inputs, self.outputs)
        for g in self.graph:
            g.initial_params()
            for v in g.variables:
                if v is not None and v.requires_grad and v not in self.trainable_variables:
                    self.trainable_variables.append(v)
        self.loss = get_objective(loss)
        self.optimizer = get_optimizer(optimizer, **kwargs)
        self.optimizer.trainable_variables = self.trainable_variables

    def forward(self, x: Variable, *args):
        self.inputs.input_data = x
        outputs = None
        for layer in self.graph:
            outputs = layer.forward(is_training=self.is_training)
        return outputs

