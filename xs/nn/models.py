import nn
from optim import get_optimizer
from utils.common import GLOBAL, List, Tuple
import nn.functional as F
import nn.objectives
from utils.toolkit import ProgressBar, format_time, SummaryProfile
import time
import pickle
import utils.data as data
from collections import defaultdict
import core.autograd


class _Base:
    def __init__(self):
        self._parameters = set()

    def train(self):
        GLOBAL.IS_TRAINING = True

    def eval(self):
        GLOBAL.IS_TRAINING = False

    def parameters(self, parameters: List[F.Tensor] = None):
        if parameters is None:
            return self._parameters
        else:
            self._parameters = set(parameters)

    def to(self, dst: str):
        if dst == 'cuda':
            if not GLOBAL.USE_CUDA:
                GLOBAL.USE_CUDA = True
                GLOBAL.np = __import__('cupy')
            for parameter in self._parameters:
                parameter.cuda_()
        return self


class _Model(_Base):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.optimizer = None
        self._graph = None

    def compile(self, optimizer, loss):
        if GLOBAL.USE_CUDA:
            for parameter in self._parameters:
                parameter.cuda_()

    def init_graph_out_tensor(self, x, y):
        raise NotImplemented

    def fit(self,
            x: GLOBAL.np.ndarray,
            y: GLOBAL.np.ndarray,
            batch_size: int = None,
            epochs: int = 1,
            verbose: int = 1,
            shuffle: bool = True,
            validation_data: Tuple[GLOBAL.np.ndarray] = None,
            validation_split: float = 0.,
            initial_epoch: int = 0,
            ) -> SummaryProfile:
        x = GLOBAL.np.asarray(x)
        y = GLOBAL.np.asarray(y)
        record_data_names = ['train_loss', 'train_acc']
        if validation_data is None and 0. < validation_split < 1.:
            split = int(x.shape[0] * validation_split)
            valid_x, valid_y = x[-split:], y[-split:]
            train_x, train_y = x[:-split], y[:-split]
            validation_data = (valid_x, valid_y)
            record_data_names.append('validation_loss')
            record_data_names.append('validation_acc')
        else:
            train_x, train_y = x, y

        train_dataset = data.DataSet(train_x, train_y)
        train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle)
        v_acc, v_loss = 0., 0.
        with SummaryProfile(*record_data_names) as profile:
            for epoch in range(initial_epoch, epochs):
                epoch_start_time = time.time()
                if verbose != 0:
                    print('\033[1;31m Epoch[%d/%d]\033[0m' % (epoch + 1, epochs))
                progress_bar = ProgressBar(max_iter=len(train_x), verbose=verbose)
                for idx, (batch_x, batch_y) in enumerate(train_dataloader):
                    if GLOBAL.USE_CUDA:
                        batch_x.cuda_()
                        batch_y.cuda_()
                    else:
                        batch_x.cpu_()
                        batch_y.cpu_()
                    if idx == 0 or idx == len(train_dataloader) - 1:
                        self.init_graph_out_tensor(batch_x, batch_y)
                    self.train()
                    # reset trainable_variables grad
                    self.optimizer.zero_grad()
                    # forward
                    pred = self.forward(batch_x)
                    loss = self.loss.forward(pred, batch_y)
                    self.backward(loss)
                    self.optimizer.step()
                    epoch_time = time.time() - epoch_start_time
                    train_acc = self.loss.calc_acc(pred, batch_y)
                    profile.step('train_acc', train_acc)
                    profile.step('train_loss', loss.item())

                    if validation_data is not None:
                        valid_x, valid_y = validation_data
                        v_acc, v_loss = self.evaluate(valid_x, valid_y, batch_size=batch_size)
                        profile.step('validation_loss', v_loss)
                        profile.step('validation_acc', v_acc)

                    progress_bar.update(batch_x.shape[0])
                    progress_bar.console(verbose, epoch_time, loss.item(), train_acc, v_loss,
                                         v_acc)

                if verbose != 0:
                    print()

        return profile

    def __call__(self, x: F.Tensor, *args, **kwargs):
        out = self.forward(x)
        return out

    def forward(self, x, *args, **kwargs):
        raise NotImplemented

    def backward(self, loss: F.Tensor):
        if loss.grad is None:
            loss.zero_grad()
            loss.grad.fill_(1.)
        loss.grad_fn(loss)
        for layer in reversed(self._graph):
            layer.backward()

    def evaluate(self, x: GLOBAL.np.ndarray, y: GLOBAL.np.ndarray, batch_size: int = None):
        self.eval()
        x = GLOBAL.np.asarray(x)
        y = GLOBAL.np.asarray(y)
        if batch_size is not None:
            assert type(batch_size) is int
            val_dataset = data.DataSet(x, y)
            val_dataloader = data.DataLoader(val_dataset, batch_size)
            acc_list = []
            loss_list = []
            for idx, (batch_x, batch_y) in enumerate(val_dataloader):
                if idx == 0 or idx == len(val_dataloader) - 1:
                    self.init_graph_out_tensor(batch_x, batch_y)

                y_pred = self.forward(batch_x)
                metric = self.loss.metric(y_pred, batch_y)
                acc_list.append(metric[0])
                loss_list.append(metric[1])

            acc = GLOBAL.np.array(acc_list).mean().tolist()
            loss = GLOBAL.np.array(loss_list).mean().tolist()
        else:
            y_pred = self.forward(F.Tensor(x))
            acc, loss = self.loss.metric(y_pred, F.Tensor(y))
        return acc, loss

    def predict(self, x: GLOBAL.np.ndarray, batch_size: int = None):
        self.eval()
        if batch_size is not None:
            assert type(batch_size) is int
            test_dataset = data.DataSet(x)
            test_dataloader = data.DataLoader(test_dataset, batch_size)
            pred_list = []
            for batch_x in test_dataloader:
                y_pred = self.forward(batch_x)
                pred_list.append(y_pred)
            pred = F.concat(pred_list, axis=0)
        else:
            pred = self.forward(F.Tensor(x))

        return pred

    def save(self, save_path):
        if not save_path.endswith('.pkl'):
            save_path += '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump([self._graph, self.optimizer, self.loss], f)

    def load(self, model_path):
        if not model_path.endswith('.pkl'):
            model_path += '.pkl'
        with open(model_path, 'rb') as f:
            graph, optimizer, loss = pickle.load(f)

        self._graph = graph
        self.optimizer = optimizer
        self.loss = loss

    def __str__(self):
        bar_nums = 75
        print('*' * bar_nums)
        print('Layer(type)'.ljust(25), 'Output Shape'.ljust(20), 'Param'.ljust(10), 'Connected to'.ljust(15))
        print('#' * bar_nums)
        if self._graph is None:
            raise ValueError('Please compile Model!')
        simple_name_dict = {
            'Conv2D': 'conv2d',
            'Input': 'input',
            'Dense': 'dense',
            'MaxPooling2D': 'max_pool2d',
            'AvgPooling2D': 'avg_pool2d',
            'Activation': 'activation',
            'ZeroPadding2D': 'zero_pad2d',
            'Add': 'add',
            'Flatten': 'flat'
        }
        simple_name_count_dict = defaultdict(int)
        for layer in self._graph:
            if layer.name is None:
                layer.name = simple_name_dict[layer.__class__.__name__] + str(simple_name_count_dict[simple_name_dict[layer.__class__.__name__]])
                simple_name_count_dict[simple_name_dict[layer.__class__.__name__]] += 1
            layer_name = '%s (%s)' % (layer.name, layer.__class__.__name__)
            params = layer.params_count()
            first = True
            if layer.in_bounds:
                for prev_layer in layer.in_bounds:
                    connected = prev_layer.name
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
        for v in self._parameters:
            total_params += v.eval.size
            if v.requires_grad:
                trainable_params += v.eval.size
        params_details = 'Total params: %d\n' % total_params
        params_details += 'Trainable params: %d\n' % trainable_params
        params_details += 'Non-trainable params: %d\n' % (total_params - trainable_params)
        return params_details


class Sequential(_Model):
    def __init__(self, *layers):
        super().__init__()
        self._graph = [] if len(layers) == 0 else list(layers)
        self.__initialized = False

    def add(self, layer):
        self._graph.append(layer)

    def init_graph_out_tensor(self, x, y):
        for layer in self._graph:
            layer.init_layer_out_tensor(x)
            x = layer.data
        self.loss.init_layer_out_tensor(x, y)

    def init_params(self, input_shape=None):
        from xs.layers.base import Layer
        for layer in self._graph:
            if isinstance(layer, Layer):
                if len(layer.parameters()) == 0:
                    layer.init_params(input_shape)
                for v in layer.parameters():
                    if v is not None:
                        self._parameters.add(v)
                input_shape = layer.compute_output_shape(input_shape)
            self.__initialized = True

    def compile(self, optimizer, loss, **kwargs):
        assert self._graph
        next_layer = None
        for layer in self._graph:
            layer.connect(next_layer)
            next_layer = layer
        self.init_params()
        self.loss = nn.objectives.get_objective(loss)
        self.optimizer = get_optimizer(optimizer, parameters=self._parameters, **kwargs)
        super(Sequential, self).compile(optimizer, loss)

    def forward(self, x, *args, **kwargs):
        if not self.__initialized:
            self.init_params(x.shape[1:])
        for layer in self._graph:
            x = layer.forward(x)
        return x

    def pop(self, index):
        layer = self._graph.pop(index)
        print('success delete %s layer' % layer.__class__.__name__)
        del layer


class Model(_Model):
    def __init__(self, inputs=None, outputs=None):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def init_graph_out_tensor(self, x, y):
        self.inputs._data = x
        for i in range(1, len(self._graph)):
            self._graph[i].init_layer_out_tensor()
        self.loss.init_layer_out_tensor(self._graph[-1].data, y)

    def compile(self, optimizer, loss, **kwargs):
        assert self.inputs is not None and self.outputs is not None
        self._graph = core.autograd.topological_sort(self.inputs, self.outputs)
        for g in self._graph:
            g.init_params()
            for v in g.parameters():
                if v is not None:
                    self._parameters.add(v)
        self.loss = nn.objectives.get_objective(loss)
        self.optimizer = get_optimizer(optimizer, parameters=self._parameters, **kwargs)
        super(Model, self).compile(optimizer, loss)

    def forward(self, x: F.Tensor, *args, **kwargs):
        self.inputs._in_bounds = [x]
        outputs = None
        for layer in self._graph:
            outputs = layer.forward()
        return outputs


class Module(_Base):
    def __call__(self, x: F.Tensor, *args, **kwargs):
        out = self.forward(x)
        self.__collect_variables(x)
        out.is_leaf = True
        return out

    def __collect_variables(self, x: F.Tensor):
        queue = [x]
        seen = set()
        seen.add(x)
        while queue:
            vertex = queue.pop(0)
            for n in vertex.out_bounds:
                if n not in seen:
                    for v in n.in_bounds:
                        if isinstance(v, F.Parameter):
                            self._parameters.add(v)
                    queue.append(n)
                    seen.add(n)

    def forward(self, x):
        raise NotImplemented
