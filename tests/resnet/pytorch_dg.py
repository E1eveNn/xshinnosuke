import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, *args):
        out = self.conv1(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.in_channels = 64

        self.layer1 = self.make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, 2)
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, 100)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class myData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# random generate data
np.random.seed(0)
X = np.random.rand(500, 3, 56, 56).astype(np.float32)
Y = np.random.randint(0, 100, (500,)).astype(np.int64)


net = ResNet18()
EPOCH = 5
train_data = myData(X, Y)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
optimizer = SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
import time
st = time.time()
for epoch in range(EPOCH):
    for x, y in train_loader:
        # x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        acc = (torch.max(pred, 1)[1].data.cpu().numpy() == y.data.cpu().numpy()).mean()
        print('epoch %d, acc -> %f, loss -> %f' % (epoch, acc, loss.data.item()))
print('Time usage: ', time.time() - st)
