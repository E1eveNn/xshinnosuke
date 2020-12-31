from util import load_mnist
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import psutil
import os


def FCNet(n_classes=10):
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(True),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Linear(256, n_classes),
    )


def ConvNet(n_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(True),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Flatten(),
        nn.Linear(288, 100),
        nn.ReLU(True),
        nn.Linear(100, n_classes),
    )


class myDataSet(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        x = self.train_x[item]
        y = self.train_y[item]
        return x, y



def go():
    import time
    #########################  EPOCH BATCH SIZE
    EPOCH = 1
    BATCH_SIZE = 256

    #########################  Read Data
    # trainset, valset, testset = load_mnist()
    train_images, train_labels = load_mnist(kind='train')
    test_images, test_labels = load_mnist(kind='t10k')
    #########################  Network
    # net = FCNet()
    # #
    # optimizer = optim.SGD(net.parameters(), 0.1)
    # criterion = nn.CrossEntropyLoss()
    #
    # train_dataset = myDataSet(train_images, train_labels)
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    # st = time.time()
    # for epoch in range(EPOCH):
    #     for x, y in train_loader:
    #         x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
    #
    #         optimizer.zero_grad()
    #         pred = net(x)
    #         loss = criterion(pred, y)
    #         loss.backward()
    #         optimizer.step()
    # et = time.time()
    # print('time cost: ', et - st)
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))




    #
    print('Start Training ConvNet---')
    #########################  Network
    net = ConvNet()

    optimizer = optim.SGD(net.parameters(), 0.1)
    criterion = nn.CrossEntropyLoss()
    train_dataset = myDataSet(train_images, train_labels)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    st = time.time()
    for epoch in range(EPOCH):
        for x, y in train_loader:
            x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
            x = x.view(-1, 1, 28, 28)
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
    et = time.time()
    print('time cost: ', et - st)
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


if __name__ == '__main__':
    go()