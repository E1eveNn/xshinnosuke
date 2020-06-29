from ..nn.global_graph import np
from typing import List, Tuple


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


class DataSet:
    def __init__(self, *datas):
        self.datas = list(datas)

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, item):
        ret_list = []
        for data in self.datas:
            ret_list.append(data[item])
        return ret_list


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int = None, shuffle: bool = False, seed: int = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = 0 if seed is None else seed
        self.mini_batches = self.make_batches(self.dataset.datas, self.batch_size, self.seed, self.shuffle)
        self.sp = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.sp += 1
        if self.sp >= len(self.mini_batches):
            self.seed += 1
            self.mini_batches = self.make_batches(self.dataset.datas, self.batch_size, self.seed, self.shuffle)
            self.sp = -1
            raise StopIteration
        return self.mini_batches[self.sp]

    def make_batches(self, datas: List[np.ndarray], batch_size: int, seed: int, shuffle: bool = False):
        if batch_size is None:
            return [datas, ]
        np.random.seed(seed)
        m = datas[0].shape[0]
        if shuffle:
            permutation = np.random.permutation(m)
            datas = list(map(lambda x: x[permutation], datas))

        mini_batches = []
        complete_batch_nums = m // batch_size  # 完整的mini_batch个数
        for i in range(complete_batch_nums):
            mini_batch = list(map(lambda x: x[batch_size * i:batch_size * (i + 1)], datas))
            mini_batches.append(mini_batch)

        if m % batch_size != 0:
            mini_batches.append(list(map(lambda x: x[batch_size * complete_batch_nums:], datas)))
        return mini_batches


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

