from xs.layers import Dense, Flatten, Conv2D, MaxPooling2D, AvgPooling2D, BatchNormalization, Activation, Add, Input
from xs.nn.models import Model
import numpy as np


def identity_block(X, filters, stage, block, s):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2 = filters

    X_shortcut = X

    X = Conv2D(out_channels=F1, kernel_size=(3, 3), stride=(s, s), padding=1,
               name=conv_name_base + "2a")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv2D(out_channels=F2, kernel_size=(3, 3), stride=(1, 1), padding=1,
               name=conv_name_base + "2b")(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, filters, stage, block, s=2):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2 = filters

    X_shortcut = X

    X = Conv2D(out_channels=F1, kernel_size=(3, 3), stride=(s, s), padding=1,
               name=conv_name_base + "2b")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv2D(out_channels=F2, kernel_size=(3, 3), stride=(1, 1), padding=1,
               name=conv_name_base + "2c")(X)
    X = BatchNormalization()(X)

    X_shortcut = Conv2D(out_channels=F2, kernel_size=(1, 1), stride=(s, s),
                        name=conv_name_base + "1")(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def ResNet18(input_shape=(3, 56, 56), classes=100):
    X_input = Input(input_shape)

    # stage1
    X = Conv2D(out_channels=64, kernel_size=(7, 7), stride=(2, 2), name="conv1", padding=3)(X_input)
    X = BatchNormalization(name="bn1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(kernel_size=3, stride=2, padding=1)(X)

    # stage2
    X = identity_block(X, filters=[64, 64], stage=2, block="b", s=1)
    X = identity_block(X, filters=[64, 64], stage=2, block="c", s=1)

    # stage3
    X = convolutional_block(X, filters=[128, 128], stage=3, block="a", s=2)
    X = identity_block(X, filters=[128, 128], stage=3, block="b", s=1)

    # stage4
    X = convolutional_block(X, filters=[256, 256], stage=4, block="a", s=2)
    X = identity_block(X, filters=[256, 256], stage=4, block="b", s=1)

    # stage5
    X = convolutional_block(X, filters=[512, 512], stage=5, block="a", s=2)
    X = identity_block(X, filters=[512, 512], stage=5, block="b", s=1)

    X = AvgPooling2D(2)(X)

    X = Flatten()(X)
    X = Dense(classes, name="fc" + str(classes), )(X)

    model = Model(inputs=X_input, outputs=X)

    return model


np.random.seed(0)
x = np.random.rand(500, 3, 56, 56)
y = np.random.randint(0, 100, (500,))
net = ResNet18()
net.compile(optimizer='sgd', loss='cross_entropy', lr=0.1)
# print(net)
history = net.fit(x, y, batch_size=32, epochs=5)
