from .utils import load_mnist
from .network import FCNet, ConvNet
from xshinnosuke.utils import DataSet, DataLoader
from xshinnosuke.nn import Variable, CrossEntropy
from xshinnosuke.nn.optimizers import SGD


def go():
    #########################  EPOCH
    EPOCH = 20

    #########################  Read Data
    trainset, valset, testset = load_mnist()
    train_dataset = DataSet(*trainset)
    train_loader = DataLoader(train_dataset, batch_size=100)

    #########################  Network
    net = FCNet()

    #########################  Optimizer
    optimizer = SGD(net.parameters())

    #########################  Criterion
    criterion = CrossEntropy()

    #########################  Train FC
    for epoch in range(EPOCH):
        net.train()
        for x, y in train_loader:
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=100)
        print(f'Epoch [{epoch}]: val acc -> {val_acc}, val loss -> {val_loss}')

    train_acc, train_loss = net.evaluate(trainset[0], trainset[1], batch_size=100)
    val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=100)
    test_acc, test_loss = net.evaluate(testset[0], testset[1], batch_size=100)
    print('#' * 10, 'Fully Connected Network')
    print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    print(f'Val acc -> {val_acc}, Val loss -> {val_loss}')
    print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')


    net = ConvNet()
    optimizer = SGD(net.parameters())
    #########################  Train Conv
    for epoch in range(EPOCH):
        net.train()
        for x, y in train_loader:
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=100)
        print(f'Epoch [{epoch}]: val acc -> {val_acc}, val loss -> {val_loss}')

    train_acc, train_loss = net.evaluate(trainset[0], trainset[1], batch_size=100)
    val_acc, val_loss = net.evaluate(valset[0], valset[1], batch_size=100)
    test_acc, test_loss = net.evaluate(testset[0], testset[1], batch_size=100)
    print('#' * 10, 'Convolutional Network')
    print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    print(f'Val acc -> {val_acc}, Val loss -> {val_loss}')
    print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')
