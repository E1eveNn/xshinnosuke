from xs.layers import Dense, Flatten, Conv2D, MaxPooling2D, AvgPooling2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D
from xs.nn import Model
import numpy as np


i = 0


def ide_block(X, filters, stage, block):
    global i
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters

    X_shortcut = X
    # 1
    X = Conv2D(out_channels=F1, kernel_size=(1, 1), stride=(1, 1), padding=0,
               name=i)(X)
    i += 1
    X = BatchNormalization(name=i)(X)
    i += 1
    X = Activation("relu", name=i)(X)
    i += 1
    # 2
    X = Conv2D(out_channels=F2, kernel_size=(3, 3), stride=(1, 1), padding=1,
               name=i)(X)
    i += 1
    X = BatchNormalization(name=i)(X)
    i += 1
    X = Activation("relu", name=i)(X)
    i += 1
    # 3
    X = Conv2D(F3, (1, 1), stride=(1, 1), padding=0, name=i,
               )(X)
    i += 1
    # 归一化
    X = BatchNormalization(name=i)(X)
    i += 1
    # 将主路径与shortcut相加
    X = Add(name=i)([X, X_shortcut])
    i += 1
    X = Activation('relu', name=i)(X)
    i += 1
    return X


def conv_block(X, filters, stage, block, s=2):
    global i
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters

    X_shortcut = X
    # 1
    X = Conv2D(out_channels=F1, kernel_size=(1, 1), stride=(s, s), padding=0,
               name=i)(X)
    i += 1
    X = BatchNormalization(name=i)(X)
    i += 1
    X = Activation('relu', name=i)(X)
    i += 1
    # 2
    X = Conv2D(out_channels=F2, kernel_size=(3, 3), stride=(1, 1), padding=1,
               name=i)(X)
    i += 1
    X = BatchNormalization(name=i)(X)
    i += 1
    X = Activation('relu', name=i)(X)
    i += 1
    # 3
    X = Conv2D(out_channels=F3, kernel_size=(1, 1), stride=(1, 1), padding=0,
               name=i)(X)
    i += 1
    X = BatchNormalization(name=i)(X)
    i += 1

    X_shortcut = Conv2D(out_channels=F3, kernel_size=(1, 1), stride=(s, s),
                        name=i)(X_shortcut)
    i += 1
    X_shortcut = BatchNormalization(name=i)(X_shortcut)
    i += 1
    X = Add(name=i)([X, X_shortcut])
    i += 1
    X = Activation('relu', name=i)(X)
    i += 1
    return X


def ResNet50(input_shape=(3, 64, 64), classes=100):
    global i
    X_input = Input(input_shape, name=i)
    i += 1
    # 进行填充
    X = ZeroPadding2D((3, 3), name=i)(X_input)
    i += 1
    # stage 1
    # 卷积
    X = Conv2D(64, (7, 7), stride=(2, 2), name=i)(X)
    i += 1
    # 归一化
    X = BatchNormalization(name=i)(X)
    i += 1
    # relu
    X = Activation('relu', name=i)(X)
    i += 1
    # 最大池化
    X = MaxPooling2D(3, 2, name=i)(X)
    i += 1

    # stage 2
    X = conv_block(X,filters=[64, 64, 256], stage=2, block='a', s=1)
    X = ide_block(X, filters=[64, 64, 256], stage=2, block='b')
    X = ide_block(X, filters=[64, 64, 256], stage=2, block='c')

    # stage 3
    X = conv_block(X, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = ide_block(X, filters=[128, 128, 512], stage=3, block='b')
    X = ide_block(X, filters=[128, 128, 512], stage=3, block='c')
    X = ide_block(X, filters=[128, 128, 512], stage=3, block='d')

    # stage 4
    X = conv_block(X, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = ide_block(X, filters=[256, 256, 1024], stage=4, block='b')
    X = ide_block(X, filters=[256, 256, 1024], stage=4, block='c')
    X = ide_block(X, filters=[256, 256, 1024], stage=4, block='d')
    X = ide_block(X, filters=[256, 256, 1024], stage=4, block='e')
    X = ide_block(X, filters=[256, 256, 1024], stage=4, block='f')

    # stage 5
    X = conv_block(X, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = ide_block(X, filters=[512, 512, 2048], stage=5, block='b')
    X = ide_block(X, filters=[512, 512, 2048], stage=5, block='c')

    # 均值池化
    X = AvgPooling2D(2, 1, name=i)(X)
    i += 1
    # 全连接层
    X = Flatten(name=i)(X)
    i += 1
    X = Dense(classes, name=i)(X)
    i += 1
    model = Model(inputs=X_input, outputs=X)

    return model


# random generate data
x = np.random.rand(500, 3, 64, 64)
y = np.random.randint(0, 100, (500,))
# net = ResNet18()
net = ResNet50()
net.compile(optimizer='sgd', loss='cross_entropy', lr=0.01)
# print(net)
history = net.fit(x, y, batch_size=32, epochs=5)
