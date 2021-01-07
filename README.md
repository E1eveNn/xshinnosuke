# XShinnosuke : Deep Learning Framework

<div align=center>
	<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1597579280045&di=409d33924532df749524161e4c11f8b3&imgtype=0&src=http%3A%2F%2Fb-ssl.duitang.com%2Fuploads%2Fitem%2F201607%2F30%2F20160730144641_4UMvr.thumb.700_0.jpeg" width="300px" height="400px">
</div>


## Descriptions

XShinnosuke(short as **XS**) is a high-level neural network framework which supports for both **Dynamic Graph** and **Static Graph**, and has almost the same API to **Keras** and **Pytorch** with *slightly differences*. **It was written by Python only**, and dedicated to realize experimentations quickly.

Here are some features of XS:

1. Based on **Cupy**(GPU version)/**Numpy**  and **native** to Python.  
2. **Without** any other **3rd-party** deep learning library.
3. **Keras and Pytorch style API**, easy to start up.
4. Supports commonly used layers such as: **Dense, Conv2D, MaxPooling2D, LSTM, SimpleRNN, etc**, and commonly used function: **conv2d, max_pool2d, relu, etc**.
5. **Sequential** in Pytorch and Keras, **Model** in Keras and **Module** in Pytorch, **all of them are supported** by XS.
6. Training and inference supports for both **dynamic graph** and **static graph**.
7. **Autograd** is supported .

XS is compatible with: **Python 3.x (3.7 is recommended)**                  [==> C++ version](https://github.com/eLeVeNnN/xshinnosuke_cpp)

[1. API docs](https://elevennn.github.io/xshinnosuke/)       [2. Notebook](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/XShinnosuke-API.ipynb)

## Getting started

#### Compared with Pytorch and Keras

| ResNet18(5 Epochs, 32 Batch_size) | XS_static_graph(cpu) | XS_dynamic_graph(cpu) | Pytorch(cpu)      | Keras(cpu)     |
| --------------------------------- | -------------------- | --------------------- | ----------------- | -------------- |
| Speed(Ratio - seconds)            | *1x* - *65.05*       | *0.98x* - 66.33       | **2.67x** - 24.39 | *1.8x* - 35.97 |
| Memory(Ratio - GB)                | *1x* - *0.47*        | **0.47x**- 0.22       | *0.55x* - 0.26    | *0.96x* - 0.45 |

| ResNet18(5 Epochs, 32 Batch_size) | XS_static_graph(gpu) | XS_dynamic_graph(gpu) | Pytorch(gpu)     | Keras(gpu)     |
| --------------------------------- | -------------------- | --------------------- | ---------------- | -------------- |
| Speed(Ratio - seconds)            | *1x* - *9.64*        | *1.02x* - 9.45        | **3.47x** - 2.78 | *1.07x* - 9.04 |
| Memory(Ratio - GB)                | **1x** - *0.48*      | *1.02x* - 0.49        | *4.4x* - 2.11    | *4.21x* - 2.02 |

**XS holds the best memory usage!**

---

### 1. Static Graph

The core networks of XS is a model, which provide a way to combine layers. There are two model types: **Sequential** (a linear stack of layers) and **Functional** (build  a graph for layers).

For **Sequential** model:

```python
from xs.nn.models import Sequential

model = Sequential()
```

Using `.add()` to connect layers:

```python
from xs.layers import Dense

model.add(Dense(out_features=500, activation='relu', input_shape=(784, )))  # must be specify input_shape if current layer is the first layer of model
model.add(Dense(out_features=10))
```

Once you have constructed your model, you should configure it with `.compile()` before training or inference:

```python
model.compile(loss='cross_entropy', optimizer='sgd')
```

If your labels are `one-hot` encoded vectors/matrix, you shall specify loss as  *sparse_crossentropy*, otherwise use *crossentropy* instead.

Use `print(model)` to see details of model:

```python
***************************************************************************
Layer(type)               Output Shape         Param      Connected to   
###########################################################################
dense0 (Dense)            (None, 500)          392500     
              
---------------------------------------------------------------------------
dense1 (Dense)            (None, 10)           5010       dense0         
---------------------------------------------------------------------------
***************************************************************************
Total params: 397510
Trainable params: 397510
Non-trainable params: 0
```

Start training your network by `fit()`:

```python
# trainX and trainy are ndarrays
history = model.fit(trainX, trainy, batch_size=128, epochs=5)
```

Once completing training your model, you can save or load your model by `save()` / `load()`, respectively.
```python
model.save(save_path)
model.load(model_path)
```


Evaluate your model performance by `evaluate()`:

```python
# testX and testy are Cupy/Numpy ndarray
accuracy, loss = model.evaluate(testX, testy, batch_size=128)
```

Inference through `predict()`:

```python
predict = model.predict(testX)
```

---

For **Functional** model:

```python
from xs.nn.models import Model
from xs.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

X_input = Input(input_shape = (1, 28, 28))   # (channels, height, width)
X = Conv2D(8, (2, 2), activation='relu')(X_input)
X = MaxPooling2D((2, 2))(X)
X = Flatten()(X)
X = Dense(10)(X)
model = Model(inputs=X_input, outputs=X)  
model.compile(optimizer='sgd', loss='cross_entropy')
model.fit(trainX, trainy, batch_size=256, epochs=80)
```

Pass inputs and outputs layer to `Model()`, then **compile** and **fit** model as `Sequential`model.

### 2. Dynamic Graph

First design your own network, make sure your network is inherited from **Module** and *override* the `__init__()` and `forward()` function:

```python
from xs.nn.models import Module
from xs.layers import Conv2D, ReLU, Flatten, Dense
import xs.nn.functional as F

class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(out_channels=8, kernel_size=3)  # don't need to specify in_channels, which is simple than Pytorch
        self.relu = ReLU(inplace=True)
        self.flat = Flatten()
        self.fc = Dense(10)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.flat(x)
        x = self.fc(x)
        return x
```

Then manually set the training/ testing flow:

```python
from xs.nn.optimizers import SGD
from xs.utils.data import DataSet, DataLoader
import xs.nn as nn
import numpy as np

# random generate data
X = np.random.randn(100, 3, 12, 12)
Y = np.random.randint(0, 10, (100, ))
# generate training dataloader
train_dataset = DataSet(X, Y)
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
# initialize net
net = MyNet()
# specify optimizer and critetion
optimizer = SGD(net.parameters(), lr=0.1)
critetion = nn.CrossEntropyLoss()
# start training
EPOCH = 5
for epoch in range(EPOCH):
    for x, y in train_loader:
        optimizer.zero_grad()
        out = net(x)
        loss = critetion(out, y)
        loss.backward()
        optimizer.step()
        train_acc = critetion.calc_acc(out, y)
        print(f'epoch -> {epoch}, train_acc: {train_acc}, train_loss: {loss.item()}')
```

Building an image classification model, a question answering system or any other model is just as convenient and fast~

---

## Autograd

XS autograd supports for basic operators such as: `+, -, *, \, **, etc` and some common functions: `matmul(), mean(), sum(), log(), view(), etc `. 

```python
from xs.nn import Tensor

a = Tensor(5, requires_grad=True)
b = Tensor(10, requires_grad=True)
c = Tensor(3, requires_grad=True)
x = (a + b) * c
y = x ** 2
print('x: ', x)  # x:  Variable(45.0, requires_grad=True, grad_fn=<MultiplyBackward>)
print('y: ', y)  # y:  Variable(2025.0, requires_grad=True, grad_fn=<PowBackward>)
x.retain_grad()
y.backward()
print('x grad:', x.grad)  # x grad: 90.0
print('c grad:', c.grad)  # c grad: 1350.0
print('b grad:', b.grad)  # b grad: 270.0
print('a grad:', a.grad)  # a grad: 270.0
```

Here are examples of [autograd](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/autograd.ipynb). 

## Installation

Before installing XS, please install the following **dependencies**:

+ Numpy

- Cupy (**Optional**)

Then you can install XS by using pip:

`$ pip install xshinnosuke`

------



## Supports

### functional

+ admm
+ mm
+ relu
+ flatten
+ conv2d
+ max_pool2d
+ avg_pool2d
+ reshape
+ sigmoid
+ tanh
+ softmax
+ dropout2d
+ batch_norm
+ groupnorm2d
+ layernorm
+ pad_2d
+ embedding

### Two basic class:

#### - Layer:

- Dense
- Flatten
- Conv2D
- MaxPooling2D
- AvgPooling2D
- ChannelMaxPooling
- ChannelAvgPooling
- Activation
- Input
- Dropout
- BatchNormalization
- LayerNormalization
- GroupNormalization
- TimeDistributed
- SimpleRNN
- LSTM
- Embedding
- ZeroPadding2D
- Add
- Multiply
- Matmul
- Log
- Negative
- Exp
- Sum
- Abs
- Mean
- Pow

#### - Tenosr:

- Parameter

### Optimizers

- SGD
- Momentum
- RMSprop
- AdaGrad
- AdaDelta
- Adam

Waiting for implemented more

### Objectives

- MSELoss
- MAELoss
- BCELoss
- SparseCrossEntropy
- CrossEntropyLoss

### Activations

- ReLU
- Sigmoid
- Tanh
- Softmax

### Initializations

- Zeros
- Ones
- Uniform
- LecunUniform
- GlorotUniform
- HeUniform
- Normal
- LecunNormal
- GlorotNormal
- HeNormal
- Orthogonal

### Regularizes

waiting for implement.

### Preprocess

- to_categorical (convert inputs to one-hot vector/matrix)
- pad_sequences (pad sequences to the same length)

## Contact

- Email: eleven_1111@outlook.com