from xshinnosuke.layers import Conv2D, Flatten, Dense, Embedding
from xshinnosuke.models import Module
from xshinnosuke.nn.optimizers import Adam
from xshinnosuke.nn import Variable, CrossEntropy
from xshinnosuke.utils import DataSet, DataLoader
import cupy


class CNN(Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.mean = None
        self.std = None
        self.epochs = 50
        self.embed = Embedding(output_dim=1, input_shape=(15,))
        self.conv1 = Conv2D(32, kernel_size=3, use_bias=True, padding=1, activation='relu')
        self.conv2 = Conv2D(64, kernel_size=3, use_bias=True, padding=1, activation='relu')
        self.conv3 = Conv2D(128, kernel_size=3, use_bias=True, padding=1, activation='relu')
        self.flat = Flatten()
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.fc3 = Dense(n_classes)

    def training(self, x, y, height, width):
        self.height = height
        self.width = width
        self.train()
        x, y = self.preprocess_data(x, y, height, width)
        train_dataset = DataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        optimizer = Adam(self.parameters())
        criterion = CrossEntropy()
        for epoch in range(self.epochs):
            for xs, ys in train_loader:
                xs = Variable(xs, dtype=cupy.int)
                ys = Variable(ys)
                optimizer.zero_grad()
                pred = self(xs)
                loss = criterion(pred, ys)
                loss.backward()
                optimizer.step()
                print(f'epoch: {epoch} loss: {loss.data}')
        self.save('./model')

    def prediction(self, x, height, width):
        self.height = height
        self.width = width
        self.load('./model')
        x = cupy.array(x)
        x = self.convert_x(x)
        x = Variable(x, dtype=cupy.int)
        out = self.predict(x)
        return cupy.argmax(out.data, 1).tolist()

    def preprocess_data(self, x, y, height, width):
        x = cupy.array(x)
        y = cupy.array(y)
        x = self.convert_x(x)
        y = y[:, None]
        return x, y

    def convert_x(self, x):
        for i in range(1, 15):
            target = 2 ** i
            pos = cupy.where(x == target)
            x[pos] = i
        return x

    def normalize(self, x):
        self.mean = cupy.mean(x)
        self.std = cupy.std(x)
        return (x - self.mean) / self.std

    def forward(self, x, *args):
        x = self.embed(x)
        x = x.view(-1, 1, self.height, self.width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
