from utils.data import DataSet
from utils.common import np, List
from nn.initializers import Empty


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int = None, shuffle: bool = False, seed: int = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = 0 if seed is None else seed
        self.mini_batches = self.make_batches(self.dataset.datas, self.batch_size, self.seed, self.shuffle)
        self.sp = -1
        self.__tensor_pool = None
        self.__initialize_tensor_pool()

    def __iter__(self):
        return self

    def __initialize_tensor_pool(self):
        assert self.dataset.datas
        if self.batch_size > self.dataset.datas[0].shape[0]:
            self.__tensor_pool = [Empty()((self.dataset.datas[0].shape[0],) + self.dataset.datas[i].shape[1:],
                                          requires_grad=False) for i in range(len(self.dataset.datas))]
        else:
            self.__tensor_pool = [Empty()((self.batch_size,) + self.dataset.datas[i].shape[1:], requires_grad=False) for i in
                                  range(len(self.dataset.datas))]

    def __next__(self):
        self.sp += 1
        if self.sp >= len(self.mini_batches):
            self.seed += 1
            self.mini_batches = self.make_batches(self.dataset.datas, self.batch_size, self.seed, self.shuffle)
            self.sp = -1
            raise StopIteration
        for i, data in enumerate(self.mini_batches[self.sp]):
            if data.shape[0] < self.__tensor_pool[i].shape_capacity[0]:
                self.__tensor_pool[i].slices(slice(None, data.shape[0], None))
            elif self.__tensor_pool[i].shape_capacity[0] != self.__tensor_pool[i].shape[0] and data.shape[0] == self.__tensor_pool[i].shape_capacity[0]:
                self.__tensor_pool[i].slices(slice(None, None, None))
            self.__tensor_pool[i].data[:] = data
        return self.__tensor_pool[0] if len(self.__tensor_pool) == 1 else self.__tensor_pool

    def __len__(self):
        return len(self.mini_batches)

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
