from util import load_mnist
from network import FCNet, ConvNet
import psutil
import os


def go():
    import time
    #########################  EPOCH BATCH SIZE
    EPOCH = 1
    BATCH_SIZE = 256

    #########################  Read Data
    # trainset, valset, testset = load_mnist()
    train_images, train_labels = load_mnist(kind='train')
    test_images, test_labels = load_mnist(kind='t10k')
    # #########################  Network
    # net = FCNet()
    # #
    # # #########################  Compile
    # net.compile(optimizer='sgd', loss='cross_entropy')
    # #
    # # #########################  Train FC
    # st = time.time()
    # net.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
    # et = time.time()
    # print('time cost: ', et - st)
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #
    # #########################  Evaluate FC
    # train_acc, train_loss = net.evaluate(train_images, train_labels, batch_size=500)
    #
    # test_acc, test_loss = net.evaluate(test_images, test_labels, batch_size=500)
    # print('#' * 10, 'Fully Connected Network')
    # print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    # print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')



    print('Start Training ConvNet---')
    #########################  Network
    net = ConvNet()

    #########################  Compile
    net.compile(optimizer='sgd', loss='cross_entropy', lr=0.01)

    #########################  Train Conv
    st = time.time()
    net.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
    et = time.time()
    print('time cost: ', et - st)
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    # #########################  Evaluate FC
    train_acc, train_loss = net.evaluate(train_images, train_labels, batch_size=500)

    test_acc, test_loss = net.evaluate(test_images, test_labels, batch_size=500)
    print('#' * 10, 'Convolutional Network')
    print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')


if __name__ == '__main__':
    go()
