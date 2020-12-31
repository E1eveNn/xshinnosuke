import xs.nn as nn
from xs.layers import Dense
import numpy as np


def network():
    return nn.Sequential(
        Dense(5, activation='relu', input_shape=(10, ), use_bias=False),
        # Dense(5, activation='relu', use_bias=False),
        Dense(3, use_bias=False)
    )


EPOCH = 3
BATCH_SIZE = 64

#########################  Read Data
np.random.seed(0)
train_datas = np.random.randn(100, 10)
train_labels = np.random.randint(0, 3, (100, 1))
#########################  Network
net = network()
#
# #########################  Compile
net.compile(optimizer='sgd', loss='cross_entropy')
#
# #########################  Train FC

net.fit(train_datas, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
