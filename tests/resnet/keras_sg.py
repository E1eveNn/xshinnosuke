import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import psutil
import time
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
i = 0

#输入输出维度一致的情况
def identity_block(X,f,filters,stage,block):
    '''

    :param X: 输入数据，维度为(m,n_H_prev,n_W_prev,n_C_prev)
    :param f: 中间跳过了3个隐藏层，第一个隐藏层和最后一个隐藏层卷积核的大小为1，中间为f
    :param filters:一个list，每一层卷积核的数量
    :param stage:整数，根据每层的位置来命名每一层，与block参数一起使用。
    :param block:字符串，据每层的位置来命名每一层，与stage参数一起使用。
    :return:恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)
    '''
    #定义命名前缀
    global i
    #获取每一层卷积核的个数
    F1,F2,F3=filters

    #保存输入数据，用于与最后的输出F(x)相加
    X_shortcut=X

    #主路径的第一部分
    #卷积
    X=Conv2D(F1,(1,1),strides=(1,1),padding='valid',name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    #归一化
    X=BatchNormalization(axis=3,name=str(i))(X)
    i += 1
    #relu
    X=Activation('relu', name=str(i))(X)
    i += 1

    # 主路径的第二部分
    # 卷积
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    # 归一化
    X = BatchNormalization(axis=3, name=str(i))(X)
    i += 1
    # relu
    X = Activation('relu', name=str(i))(X)
    i += 1
    # 主路径的第三部分
    # 卷积
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    # 归一化
    X = BatchNormalization(axis=3, name=str(i))(X)
    i += 1
    #将主路径与shortcut相加
    X=Add(name=str(i))([X,X_shortcut])
    i += 1
    X=Activation('relu', name=str(i))(X)
    i += 1
    return X


def conv_block(X,f,filters,stage,block,strides=2):
    '''
    :param
    X: 输入数据，维度为(m, n_H_prev, n_W_prev, n_C_prev)
    :param
    f: 中间跳过了3个隐藏层，第一个隐藏层和最后一个隐藏层卷积核的大小为1，中间为f
    :param
    filters: 一个list，每一层卷积核的数量
    :param
    stage: 整数，根据每层的位置来命名每一层，与block参数一起使用。
    :param
    block: 字符串，据每层的位置来命名每一层，与stage参数一起使用。
    :param
    strdies: 卷积步长，中间那层步长为1
    :return:恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)
    '''
    # 定义命名前缀
    global i

    # 获取每一层卷积核的个数
    F1, F2, F3 = filters

    # 保存输入数据
    X_shortcut = X

    # 主路径的第一部分
    # 卷积
    X = Conv2D(F1, (1, 1), strides=(strides, strides), padding='valid', name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    # 归一化
    X = BatchNormalization(axis=3, name=str(i))(X)
    i += 1
    # relu
    X = Activation('relu', name=str(i))(X)
    i += 1
    # 主路径的第二部分
    # 卷积
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    # 归一化
    X = BatchNormalization(axis=3, name=str(i))(X)
    i += 1
    # relu
    X = Activation('relu', name=str(i))(X)
    i += 1
    # 主路径的第三部分
    # 卷积
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=str(i),kernel_initializer=glorot_uniform(seed=0))(X)
    i += 1
    # 归一化
    X = BatchNormalization(axis=3, name=str(i))(X)
    i += 1
    #shortcut
    #卷积，主要是使捷径的维度与主路径的输出维度相同
    X_shortcut=Conv2D(F3,(1,1),strides=(strides,strides),name=str(i),kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    i += 1
    X_shortcut=BatchNormalization(axis=3,name=str(i))(X_shortcut)
    i += 1
    # 将主路径与shortcut相加
    X = Add(name=str(i))([X, X_shortcut])
    i += 1
    X = Activation('relu', name=str(i))(X)
    i += 1
    return X


# tf.reset_default_graph()
# with tf.Session() as sess:
#     np.random.seed(1)
#     A_prev=tf.placeholder('float',[3,4,4,6])
#     X=np.random.randn(3,4,4,6)
#     A_identity=identity_block(A_prev,2,[2,4,6],stage=1,block='a')
#     A_different=conv_block(A_prev,2,[2,4,6],stage=1,block='a')
#     sess.run(tf.global_variables_initializer())
#     Aout1=sess.run(A_identity,feed_dict={A_prev:X,K.learning_phase():0})
#     Aout2=sess.run(A_different,feed_dict={A_prev:X,K.learning_phase():0})
#     print("identity output:",Aout1[0,1,1])
#     print("non-identity output:",Aout2[0,1,1])


def ResNet50(input_shape,classes):
    '''
    #实现50层的ResNet
    :param input_shape:输入数据的维度大小
    :param classes:输出的分类数
    :return:定义的keras模型
    '''
    #定义输入数据
    global i
    X_input=Input(input_shape, name=str(i))
    i += 1
    #进行填充
    X=ZeroPadding2D((3,3), name=str(i))(X_input)
    i += 1
    #stage 1
    #卷积
    X=Conv2D(64,(7,7),strides=(2,2),name=str(i),kernel_initializer=glorot_uniform(0))(X)
    i += 1
    #归一化
    X=BatchNormalization(axis=3,name=str(i))(X)
    i += 1
    #relu
    X=Activation('relu', name=str(i))(X)
    i += 1
    #最大池化
    X=MaxPooling2D(pool_size=(3,3),strides=(2,2), name=str(i))(X)
    i += 1
    # stage 2
    X = conv_block(X,f=3,filters=[64,64,256],stage=2,block='a',strides=1)
    X=identity_block(X,f=3,filters=[64,64,256],stage=2,block='b')
    X=identity_block(X,f=3,filters=[64,64,256],stage=2,block='c')

    # stage 3
    X = conv_block(X,f=3,filters=[128,128,512],stage=3,block='a',strides=2)
    X=identity_block(X,f=3,filters=[128,128,512],stage=3,block='b')
    X=identity_block(X,f=3,filters=[128,128,512],stage=3,block='c')
    X=identity_block(X,f=3,filters=[128,128,512],stage=3,block='d')

    # stage 4
    X = conv_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', strides=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # stage 5
    X = conv_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', strides=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    #均值池化
    X=AveragePooling2D(pool_size=(2,2), strides=1, name=str(i))(X)
    i += 1
    #全连接层
    X=Flatten(name=str(i))(X)
    i += 1
    X=Dense(classes,activation='softmax',name=str(i),kernel_initializer=glorot_uniform(0))(X)
    i += 1
    model=Model(inputs=X_input,outputs=X,name='ResNet50')

    return model


#加载数据
np.random.seed(0)
x = np.random.rand(500, 64, 64, 3)
y = np.random.randint(0, 100, (500,))

model=ResNet50([64,64,3], 100)
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# print(model.summary())
st = time.time()
model.fit(x,y,epochs=5,batch_size=32)
print('Time usage: ', time.time() - st)
print('Memory usage: ', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
# preds=model.evaluate(X_test,Y_test)
# print("loss:",preds[0])
# print("accuracy:",preds[1])
# model.save('./ResNet50.h5')
