# XShinnosuke : Deep Learning Framework

<div align=center>
	<img src="https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268%3Bg%3D0/sign=625eaa79a864034f0fcdc50097f81e0c/8c1001e93901213f91ab2a7857e736d12e2e95fd.jpg" width="">
</div>



## Descriptions

XShinnosuke is a high-level neural network framework which supports for both **Dynamic Graph** and **Static Graph**, and has almost the same API to **Keras** and **Pytorch** with *slightly differences*. **It was written by Python only**, and dedicated to realize experimentations quickly.

Here are some features of Shinnosuke:

1. Based on **Cupy**(GPU version)/**Numpy**  and **native** to Python.  
2. **Without** any other **3rd-party** deep learning library.
3. **Keras and Pytorch style API**, easy to start.
4. Supports commonly used layers such as: *Dense, Conv2D, MaxPooling2D, LSTM, SimpleRNN, etc*, and commonly used function: *conv2d, max_pool2d, relu, etc*.
5. **Sequential** model (for most  sequence network combinations ) and **Functional** model (for resnet, etc) are implemented. Meanwhile, XShinnosuke also supports for design your own network by **Module**.
6. Training and inference supports for both **dynamic graph** and **static graph**.
7. **Autograd** is supported .

XShinnosuke is compatible with: **Python 3.x (3.7 is recommended)**



[1. API docs](https://elevennn.github.io/xshinnosuke/)       [2. Notebook](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/XShinnosuke-API.ipynb)

---

## Getting started

**Here are some demos~**

1. [ResNet18 with Dynamic Graph](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/resnet18_dynamic_graph.ipynb)

2. [ResNet18 with Static Graph](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/resnet18_static_graph.ipynb)

3. [Autograd](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/examples/autograd.ipynb)

4. [Gobang](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/gobang/ui.py)

   <img src="https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/gobang/asset/demo.png" width="400px" height="400px">

5. [2048](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/_2048/ui.py)

   <img src="https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/_2048/demo.png" width="400px" height="400px">

6. [MNIST](https://github.com/eLeVeNnN/xshinnosuke/blob/master/demos/mnist/train.py)

---

### 1. Static Graph

The core networks of XShinnosuke is a model, which provide a way to combine layers. There are two model types: **Sequential** (a linear stack of layers) and **Functional** (build  a graph for layers).

For **Sequential** model:

```python
from xshinnosuke.models import Sequential

model = Sequential()
```

Using `.add()` to connect layers:

```python
from xshinnosuke.layers import Dense

model.add(Dense(out_features=500, activation='relu', input_shape=(784, )))  # must be specify input_shape if current layer is the first layer of model
model.add(Dense(out_features=10))
```

Once you have constructed your model, you should configure it with `.compile()` before training or inference:

```python
model.compile(loss='sparse_crossentropy', optimizer='sgd')
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
# trainX and trainy are Cupy ndarray
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

Combine your layers by functional API:

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

X_input = Input(input_shape = (1, 28, 28))   # (channels, height, width)
X = Conv2D(8, (2, 2), activation='relu')(X_input)
X = MaxPooling2D((2, 2))(X)
X = Flatten()(X)
X = Dense(10)(X)
model = Model(inputs=X_input, outputs=X)  
model.compile(optimizer='sgd', loss='sparse_cross_entropy')
model.fit(trainX, trainy, batch_size=256, epochs=80)
```

Pass inputs and outputs layer to `Model()`, then **compile** and **fit** model as `Sequential`model.

### 2. Dynamic Graph

First design your own network, make sure your network is inherited from **Module** and *override* the `__init__()` and `forward()` function:

```python
from xshinnosuke.models import Module
from xshinnosuke.layers import Conv2D, ReLU, Flatten, Dense
import xshinnosuke.nn.functional as F

class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(out_channels=8, kernel_size=3)
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
from xshinnosuke.nn.optimizers import SGD
from xshinnosuke.utils import DataSet, DataLoader
from xshinnosuke.nn import CrossEntropy
import cupy as np

# random generate data
X = np.random.randn(100, 3, 12, 12)
Y = np.random.randint(0, 10, (100, ))
# generate training dataloader
train_dataset = DataSet(X, Y)
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
# initialize net
net = MyNet()
# specify optimizer and critetion
optimizer = SGD(net.parameters())
critetion = CrossEntropy()
# start training
EPOCH = 5
for epoch in range(EPOCH):
    for x, y in train_loader:
        optimizer.zero_grad()
        out = net(x)
        loss = critetion(out, y)
        loss.backward()
        optimizer.step()
        train_acc, train_loss = critetion.metric(out, y)
        print(f'epoch -> {epoch}, train_acc: {train_acc}, train_loss: {train_loss}')
```

Building an image classification model, a question answering system or any other model is just as convenient and fast~

---

## Autograd

Xshinnosuke's **autograd** supports for basic operators such as: `+, -, *, \, **, etc` and some common functions: `matmul(), mean(), sum(), log(), view(), etc `. 

```python
from xshinnosuke.nn import Variable

a = Variable(5)
b = Variable(10)
c = Variable(3)
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

Before installing XShinnosuke, please install the following **dependencies**:

+ Numpy

- Cupy = 6.0.0 (**Optional**, if installed, Xshinnosuke will use this to speed up)

```markdown
notice that cupy requires **Microsoft Visual C++ 14.0**
```

Then you can install XShinnosuke by using pip:

`$ pip install xshinnosuke`

------



## Supports

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
- Batch Normalization
- Layer Normalization
- Group Normalization
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

#### - Node:

- Variable
- Constant

### Optimizers

- SGD
- Momentum
- RMSprop
- AdaGrad
- AdaDelta
- Adam

Waiting for implemented more

### Objectives

- MeanSquaredError
- MeanAbsoluteError
- BinaryCrossEntropy
- SparseCrossEntropy
- CrossEntropy 

### Activations

- Relu
- Linear
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