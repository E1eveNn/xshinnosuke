from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, BatchNormalization, MaxPooling2D, ReLU, Flatten
from keras.utils import to_categorical
from util import load_mnist
import psutil
import os


def FCNet(n_classes=10):
    net = Sequential()
    net.add(Dense(512, activation='relu', input_shape=(784,)))
    net.add(Dense(256, activation='relu'))
    net.add(Dense(n_classes, activation='softmax'))
    return net


def ConvNet(n_classes=10):
    net = Sequential()
    net.add(Reshape((28, 28, 1), input_shape=(784, )))
    net.add(Conv2D(8, 3))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Conv2D(16, 3))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(MaxPooling2D())
    net.add(Conv2D(32, 3))
    net.add(BatchNormalization())
    net.add(ReLU())
    net.add(Flatten())
    net.add(Dense(100, activation='relu'))
    net.add(Dense(n_classes))
    return net


def go():
    import time
    #########################  EPOCH BATCH SIZE
    EPOCH = 1
    BATCH_SIZE = 256

    #########################  Read Data
    # trainset, valset, testset = load_mnist()
    train_images, train_labels = load_mnist(kind='train')
    train_labels = to_categorical(train_labels)
    test_images, test_labels = load_mnist(kind='t10k')
    #########################  Network
    # net = FCNet()
    # #
    # # #########################  Compile
    # net.compile(optimizer='sgd', loss='categorical_crossentropy')
    # #
    # # #########################  Train FC
    # st = time.time()
    # net.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=0)
    # et = time.time()
    # print('time cost: ', et - st)
    # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    # #########################  Evaluate FC
    # train_acc, train_loss = net.evaluate(train_images, train_labels, batch_size=500)
    #
    # test_acc, test_loss = net.evaluate(test_images, test_labels, batch_size=500)
    # print('#' * 10, 'Fully Connected Network')
    # print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    # print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')


    #
    print('Start Training ConvNet---')
    #########################  Network
    net = ConvNet()
    #########################  Compile
    net.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    #########################  Train Conv
    st = time.time()
    net.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCH)
    et = time.time()
    print('time cost: ', et - st)
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))

    # # #########################  Evaluate Conv
    train_loss = net.evaluate(train_images, train_labels, batch_size=500)

    test_loss = net.evaluate(test_images, test_labels, batch_size=500)
    # print('#' * 10, 'Convolutional Network')
    # print(f'Train acc -> {train_acc}, Train loss -> {train_loss}')
    # print(f'Test acc -> {test_acc}, Test loss -> {test_loss}')


if __name__ == '__main__':
    go()