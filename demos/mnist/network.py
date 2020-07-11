from xshinnosuke.models import Sequential
from xshinnosuke.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, ReLU, Reshape


def FCNet(n_classes=10):
    return Sequential(
        Dense(512, activation='relu', input_shape=(784, )),
        Dense(256, activation='relu'),
        Dense(n_classes)
    )


def ConvNet(n_classes=10):
    return Sequential(
        Reshape((1, 28, 28), input_shape=(784, )),
        Conv2D(8, 3),
        BatchNormalization(),
        ReLU(True),
        MaxPooling2D(),
        Conv2D(16, 3),
        BatchNormalization(),
        ReLU(True),
        MaxPooling2D(),
        Conv2D(32, 3),
        BatchNormalization(),
        ReLU(True),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(n_classes)
    )
