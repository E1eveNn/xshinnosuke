# XShinnosuke : Deep Learning Framework

<div align=center>
	<img src="https://gss1.bdstatic.com/9vo3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268%3Bg%3D0/sign=625eaa79a864034f0fcdc50097f81e0c/8c1001e93901213f91ab2a7857e736d12e2e95fd.jpg" width="">
</div>



## Descriptions

XShinnosuke is a high-level neural network framework which supports for both **dynamic graph** and **static graph**, and has almost the same API to **Keras** and **Pytorch** with *slightly differences*. It was written by Python only, and dedicated to realize experimentations quickly.

Here are some features of Shinnosuke:

1. Based on **Cupy**(Gpu version of **Numpy**)  and **native** to Python.  
2. **Without** any other **3rd-party** deep learning library.
3. **Keras and Pytorch style API**, easy to get start.
4. Support commonly used layers such as: *Dense, Conv2D, MaxPooling2D, LSTM, SimpleRNN, etc*, and commonly used function: *conv2d, max_pool2d, relu, etc*.
5. **Sequential** model (for most  sequence network combinations ) and **Functional** model (for resnet, etc) are implemented.
6. Training and inference supports for both **dynamic graph** and **static graph**.
7. **Autograd** is supported .

Shinnosuke is compatible with: **Python 3.x (3.7 is recommended)**

`###################################### \^^shinnosuke documents^^/ ######################################`


<div align=center><a href=https://github.com/eLeVeNnN/shinnosuke/blob/master/docs/imgs/Shinnosuke-API.ipynb>Jupyter Notebook</a></div>
<div align=center><a href=https://github.com/eLeVeNnN/shinnosuke/blob/master/docs/imgs/Shinnosuke-API.md>Markdown</a></div>

------



## Getting started

### 1. keras style

The core networks of XShinnosuke is a model, which provide a way to combine layers. There are two model types: `Sequential` (a linear stack of layers) and `Functional` (build  a graph for layers).

Here is a example of `Sequential` model:

```python
from xshinnosuke.models import Sequential

model = Sequential()
```

Using `.add()` to connect layers:

```python
from shinnosuke.layers import Dense

model.add(Dense(out_features=500, activation='relu', input_shape=(784, )))  # must be specify input_shape if current layer is the first layer of model
model.add(Dense(out_features=10))
```

Once you have constructed your model, you should configure it with `.compile()` before training or inference:

```python
model.compile(loss='sparse_crossentropy', optimizer='sgd')
```

If your labels are one-hot encoded vectors/matrix, you shall specify loss as  *sparse_crossentropy*, otherwise use *crossentropy* instead. (While in **Keras** *categorical_crossentropy* supports for one-hot encoded labels).

Use `print(model)` to see details of model:

```python
***************************************************************************
Layer(type)          Output Shape         Param        Connected to   
###########################################################################
dense0 (Dense)                (None, 500)          392500       
              
---------------------------------------------------------------------------
dense1 (Dense)              (None, 10)           5010         Dense          
---------------------------------------------------------------------------
***************************************************************************
Total params: 397510
Trainable params: 397510
Non-trainable params: 0
```

Having finished `compile`, you can start training your data in batches:

```python
#trainX and trainy are Numpy arrays
m.fit(trainX, trainy, batch_size=128, epochs=5)
```

Once completing training your model, you can save or load your model by `save()` / `load()`, respectively.
```python
m.save(save_path)
m.load(model_path)
```


Evaluate your model performance by `.evaluate()`:

```python
acc, loss = m.evaluate(testX, testy, batch_size=128)
```

Or obtain predictions on new data:

```python
y_hat = m.predict(x_test)
```



For `Functional` model, first instantiate an `Input` layer:

```python
from shinnosuke.layers import Input

X_input = Input(shape = (None, 1, 28, 28))   #(batch_size,channels,height,width)
```

You need to specify the input shape, notice that for Convolutional networks,data's channels must be in the `axis 1` instead of `-1`, and you should state batch_size as None which is unnecessary in Keras.

Then Combine your layers by functional API:

```python
from shinnosuke.models import Model
from shinnosuke.layers import Conv2D,MaxPooling2D
from shinnosuke.layers import Activation
from shinnosuke.layers import BatchNormalization
from shinnosuke.layers import Flatten,Dense

X = Conv2D(8, (2, 2), padding = 'VALID', initializer = 'normal', activation = 'relu')(X_input)
X = MaxPooling2D((2, 2))(X)
X = Flatten()(X)
X = Dense(10, initializer = 'normal', activation = 'softmax')(X)
model = Model(inputs = X_input, outputs = X)  
model.compile(optimizer = 'sgd', loss = 'sparse_categorical_cross_entropy')
model.fit(trainX, trainy, batch_size = 256, epochs = 80, validation_ratio = 0.)
```

Pass inputs and outputs layer to `Model()`, and then compile and fit model like `Sequential`model.



Building an image classification model, a question answering system or any other model is just as convenient and fast~

In the [Examples folder](https://github.com/eLeVeNnN/shinnosuke/Examples/) of this repository, you can find more advanced models.

------

## Both dynamic and static graph features

As you will see soon in below, Shinnosuke has two basic classes - Layer and Node. For Layer, operations between layers can be described like this (here gives an example of '+' ):

```py
from shinnosuke.layers import Input,Add
from shinnosuke.layers import Dense

X = Input(shape = (3, 5))
X_shortcut = X
X = Dense(5)(X)  #Dense will output a (3,5) tensor
X = Add()([X_shortcut, X])
```

Meanwhile Shinnosuke will construct a graph as below:

<div align=center>
	<img src="https://github.com/eLeVeNnN/shinnosuke/blob/master/docs/imgs/layer_graph.jpg" width="300px",height="200px">
</div>





 While Node Operations have both dynamic graph and static graph features:

```python
from shinnosuke.layers.Base import Variable

x = Variable(3)
y = Variable(5)
z = x + y  
print(z.get_value())
```

You suppose get value 8, at same time shinnosuke construct a graph as below:

<div align=center>
	<img src="https://github.com/eLeVeNnN/shinnosuke/blob/master/docs/imgs/node_graph.jpg" width="300px",height="200px">
</div>



## Autograd

What is autograd? In a word, It means automatically calculate the network's gradients without any prepared backward codes for users, Shinnosuke's autograd supports for several operators, such as +, -, *, /, etc... Here gives an example:

For a simple fully connected neural network, you can use `Dense()` to construct it:

```python
from shinnosuke.models import Sequential
from shinnosuke.layers import Dense
import numpy as np

#announce a Dense layer
fullyconnected = Dense(4, n_in = 5)
m = Sequential()
m.add(fullyconnected)
m.compile(optimizer = 'sgd', loss = 'mse')  #don't mean to train it, use compile to initialize parameters
#initialize inputs
np.random.seed(0)
X = np.random.rand(3, 5)
#feed X as fullyconnected's inputs
fullyconnected.feed(X, 'inputs')
#forward
fullyconnected.forward()
out1 = fullyconnected.get_value()
print(out1.get_value())
#feed gradient to fullyconnected
fullyconnected.feed(np.ones_like(out1), 'grads')
#backward
fullyconnected.backward()
W, b = fullyconnected.variables
print(W.grads)
```

We can also construct the same layer by using following codes:

```python
from shinnosuke.layers import Variable

a = Variable(X) # the same as X in previous fullyconnected
c = Variable(W.get_value())  # the same value as W in previous fullyconnected
d = Variable(b.get_value())  # the same value as b in previous fullyconnected
out2 = a @ c + d   # @ represents for matmul
print(out2.get_value())
out2.grads = np.ones_like(out2.get_value())   #specify gradients
# by using grad(),shinnosuke will automatically calculate the gradient from out2 to c
c.grad()
print(c.grads)
```

Guess what? out1 has the same value of out2, and so did W and c's grads. This is the magic autograd of shinnosuke. **By using this feature, users can implement other networks as wishes without writing any backward codes.**

See autograd example in [Here!](https://github.com/eLeVeNnN/shinnosuke-gpu/blob/master/Examples/autograd.ipynb)

## Installation

Before installing Shinnosuke, please install the following **dependencies**:

- Numpy = 1.15.0 (recommend)
- matplotlib = 3.0.3 (recommend)

Then you can install Shinnosuke by using pip:

`$ pip install shinnosuke`

**Installation from Github source will be supported in the future.**

------



## Supports

### Two basic class:

#### - Layer:

- Dense
- Flatten
- Conv2D
- MaxPooling2D
- MeanPooling2D
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
- GRU (waiting for implemented)
- ZeroPadding2D
- Add
- Minus
- Multiply
- Matmul



#### - Node:

- Variable
- Constant



### Optimizers

- StochasticGradientDescent
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
- SparseCategoricalCrossEntropy
- CategoricalCrossEntropy 

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

### Utils

- get_batches (generate mini-batch)
- to_categorical (convert inputs to one-hot vector/matrix)
- concatenate (concatenate Nodes that have the same shape in specify axis)
- pad_sequences (pad sequences to the same length)

## Contact

- Email: eleven_1111@outlook.com