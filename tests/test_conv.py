import xs.nn as nn
from xs.layers import Dense, Reshape, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten
import numpy as np


def network():
    return nn.Sequential(
        Reshape((1, 5, 5), input_shape=(25,)),
        Conv2D(2, 3),
        BatchNormalization(),
        ReLU(True),
        Conv2D(3, 3),
        BatchNormalization(),
        ReLU(True),
        Flatten(),
        Dense(3, use_bias=False)
    )


EPOCH = 20
BATCH_SIZE = 64

#########################  Read Data
np.random.seed(0)
train_datas = np.random.randn(1000, 25)
train_labels = np.random.randint(0, 3, (1000, 1))
# print('train datas: ')
# print(train_datas)
# print('train labels: ')
# print(train_labels)
#########################  Network
net = network()
#
# #########################  Compile
net.compile(optimizer='sgd', loss='cross_entropy')
#
# #########################  Train FC

net.fit(train_datas, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2)
