from .utils import load_mnist
from .network import FCNet, ConvNet
from xshinnosuke.utils import DataSet, DataLoader
from xshinnosuke.nn import Variable, CrossEntropy
from xshinnosuke.nn.optimizers import SGD


def go():
    #########################  EPOCH BATCH SIZE
    EPOCH = 5
    BATCH_SIZE = 256

    #########################  Read Data
    trainset, valset, testset = load_mnist()

    #########################  Network
    net = FCNet()

    #########################  Compile
    net.compile(optimizer='sgd', loss='cross_entropy')

    #########################  Train FC
    net.fit(trainset[0], trainset[1], batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=valset)

    #########################  Evaluate FC
    train_acc, train_loss = net.evaluate(trainset[0], trainset[1], batch_size=500)
    val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=500)
    test_acc, test_loss = net.evaluate(testset[0], testset[1], batch_size=500)
    print('#' * 10, 'Fully Connected Network')
    print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    print(f'Val acc -> {val_acc}, Val loss -> {val_loss}')
    print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')

    print('Start Training ConvNet---')
    #########################  Network
    net = ConvNet()

    #########################  Compile
    net.compile(optimizer='sgd', loss='cross_entropy')

    #########################  Train Conv
    net.fit(trainset[0], trainset[1], batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=valset)

    train_acc, train_loss = net.evaluate(trainset[0], trainset[1], batch_size=500)
    val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=500)
    test_acc, test_loss = net.evaluate(testset[0], testset[1], batch_size=500)
    print('#' * 10, 'Convolutional Network')
    print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    print(f'Val acc -> {val_acc}, Val loss -> {val_loss}')
    print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')
