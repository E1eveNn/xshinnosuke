from xs.layers import Dense, Flatten, Conv2D, MaxPooling2D, AvgPooling2D, BatchNormalization, ReLU
from xs.utils.data import DataSet, DataLoader
import xs.nn as nn
from xs.optim import SGD
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2D(out_channels, 3, stride=stride, padding=1),
            BatchNormalization(),
            ReLU(True),
            Conv2D(out_channels, 3, stride=1, padding=1),
            BatchNormalization(),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv2D(out_channels, 1, stride=stride),
                BatchNormalization()
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = ReLU(inplace=True)

    def forward(self, x, *args):
        out = self.conv1(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv2D(out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNormalization(),
            ReLU(inplace=True)
        )
        self.pool1 = MaxPooling2D(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64

        self.layer1 = self.make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, 2)
        self.pool2 = AvgPooling2D(2)
        self.flat = Flatten()
        self.fc = Dense(100)

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
        x = self.flat(x)
        x = self.fc(x)
        return x


# random generate data
np.random.seed(0)
X = np.random.rand(500, 3, 56, 56)
Y = np.random.randint(0, 100, (500,))


net = ResNet18()
EPOCH = 5
train_data = DataSet(X, Y)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
optimizer = SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        acc = criterion.calc_acc(pred, y)
        print('epoch %d, acc -> %f, loss -> %f' % (epoch, acc, loss.item()))
