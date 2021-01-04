from .common import np, List, Union, Any, no_grad


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


class AverageMeter(object):
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


def gradient_check(inputs, target, layer, criterion, epsilon: float = 1e-4):
    variables = list(layer.parameters())
    variables_mathematical_gradient_list = []
    with no_grad():
        for i in range(len(variables)):
            variable_shape = variables[i].shape
            variable_size = variables[i].numel()
            flat_variables = variables[i].eval.reshape(-1, 1)
            mathematical_gradient = np.zeros((variable_size, ))
            new_flat_add_variable = flat_variables.copy()
            new_flat_minus_variable = flat_variables.copy()
            for j in range(variable_size):
                new_flat_add_variable[j] += epsilon
                new_variable = new_flat_add_variable.reshape(variable_shape)
                variables[i].eval = new_variable
                layer.parameters(variables)
                out = layer(inputs)
                loss1 = criterion(out, target)
                # recover
                new_flat_add_variable[j] -= epsilon

                new_flat_minus_variable[j] -= epsilon
                new_variable = new_flat_minus_variable.reshape(variable_shape)
                variables[i].eval = new_variable
                layer.parameters(variables)
                out = layer(inputs)
                loss2 = criterion(out, target)
                # recover
                new_flat_minus_variable[j] += epsilon

                mathematical_gradient[j] = (loss1.eval - loss2.eval) / (2 * epsilon)
            mathematical_gradient = mathematical_gradient.reshape(variable_shape)
            variables_mathematical_gradient_list.append(mathematical_gradient)
    return variables_mathematical_gradient_list
