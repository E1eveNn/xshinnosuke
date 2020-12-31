from .utils import SGFflie
import os
import xshinnosuke
from xshinnosuke.models import Sequential
from xshinnosuke.nn import Variable
from xshinnosuke.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dense, Dropout, Reshape


class CNN:
    def __init__(self, keep_prob=0.5):
        self.sgf = SGFflie()
        # static graph
        self.net = Sequential(
            Reshape((1, 15, 15), input_shape=(225, )),
            Conv2D(out_channels=32, kernel_size=5, use_bias=True, padding=2, kernel_initializer='he_normal',
                   bias_initializer=xshinnosuke.nn.Matrix(0.1)),
            ReLU(inplace=True),
            MaxPooling2D(kernel_size=2),
            Conv2D(out_channels=64, kernel_size=5, use_bias=True, padding=2),
            ReLU(inplace=True),
            MaxPooling2D(kernel_size=2),
            Flatten(),
            Dense(out_features=1024),
            ReLU(inplace=True),
            Dropout(keep_prob),
            Dense(225)
        )
        self.net.compile(optimizer='adam', loss='sparse_cross_entropy', lr=1e-4)

    def forward(self, x):
        return self.net.forward(x)

    def training(self, data_path, save_path, epochs=1, batch_size=None):
        path = self.sgf.allFileFromDir(data_path)
        for filepath in path:
            x, y = self.sgf.createTraindataFromqipu(filepath)
            self.net.fit(x, y, batch_size=batch_size, epochs=epochs)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.restore_save(save_path, 0)

    def prediction(self, qiju):
        _qiju = self.createdataformqiju(qiju)
        pred = self.net.predict(Variable(_qiju[0]))
        point = [0, 0]
        l = pred.argmax(axis=1).data
        for i in range(15):
            if ((i + 1) * 15) > l:
                point[0] = int(i*30 + 25)
                point[1] = int((l - i * 15) * 30 + 25)
                break
        return point

    @staticmethod
    def createdataformqiju(qiju):
        data = []
        tmp = []
        for row in qiju:
            for point in row:
                if point == -1:
                    tmp.append(0.0)
                elif point == 0:
                    tmp.append(2.0)
                elif point == 1:
                    tmp.append(1.0)
        data.append(tmp)
        return data

    def restore_save(self, path, method=1):
        # read
        if method == 1:
            self.net.load(os.path.join(path, 'net'))
            print('finish reading')
        elif method == 0:
            self.net.save(os.path.join(path, 'net'))
            print('finish saving')
