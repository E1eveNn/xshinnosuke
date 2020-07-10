from xshinnosuke.models import Module
from xshinnosuke.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization
import xshinnosuke.nn.functional as F


class FCNet(Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ConvNet(Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = Conv2D(8, 3)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(16, 3)
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D()
        self.conv3 = Conv2D(32, 3)
        self.flat = Flatten()
        self.fc1 = Dense(100, activation='relu')
        self.fc2 = Dense(n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, True)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, True)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
