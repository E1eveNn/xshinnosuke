<div align=center><img src="https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=1784511497,119911411&fm=26&gp=0.jpg"></div>



<h1>Contents</h1>
<ol>
    <li><h2><a href="#layer">Layer</a></h2>
        <ul>
            <li><h3><a href="#input">Input</a></h3></li>
            <li><h3><a href="#dense">Dense</a></h3></li>
            <li><h3><a href="#flatten">Flatten</a></h3></li>
            <li><h3><a href="#pad">ZeroPadding2D</a></h3></li>
            <li><h3><a href="#conv">Conv2D</a></h3></li>
            <li><h3><a href="#maxpool">MaxPooling2D</a></h3></li>
            <li><h3><a href="#meanpool">AvgPooling2D</a></h3></li>
            <li><h3><a href="#act">Activation</a></h3></li>
            <li><h3><a href="#reshape">Reshape</a></h3></li>
            <li><h3><a href="#dropout">Dropout</a></h3></li>
            <li><h3><a href="#bn">Batch Normalization</a></h3></li>
            <li><h3><a href="#ln">Layer Normalization</a></h3></li>
            <li><h3><a href="#gn">Group Normalization</a></h3></li>
            <li><h3><a href="#embed">Embedding</a></h3></li>
            <li><h3><a href="#rnn">SimpleRNN</a></h3></li>
            <li><h3><a href="#lstm">LSTM</a></h3></li>
            <li><h3><a href="#timedist">TimeDistributed</a></h3></li>
        </ul>
    </li>
    <li><div></div><h2><a href="#node">Node</a></h2>
        <ul>
            <li><h3><a href="#variable">Variable</a></h3></li>
            <li><h3><a href="#constant">Constant</a></h3></li>
        </ul>
    </li>
</ol>

## Layer

<div style="color: red" id="input"><i>- Input(input_shape: Tuple,  data: ndarray = None, **kwargs)</i></div>

+ input_shape: input data's shape, for example, (C, H, W) or (features, ).
+ data: this layer's input and output tensor's value.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Input

X = Input(input_shape=(10, 5, 5))
model = Model(inputs=X, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="dense"><i>- Dense(out_features: int, out_features, activation=None, use_bias=True, kernel_initializer='normal', bias_initializer='zeros', kernel_regularizer=None, **kwargs)</i></div>

+ out_features: out feature numbers.
+ activation: activation function. see details in <a href='#Activations'>Activations</a>
+ use_bias: whether use bias.
+ kernel_initializer: kernel initialize method. see details in <a href='#Initializers'>Initializers</a>
+ bias_initializer: bias initialize method. see details in <a href='#Initializers'>Initializers</a>
+ kernel_regularizer: not implemented.

```python
from xshinnosuke.models import Sequential
from xshinnosuke.layers import Dense

model = Sequential()
model.add(Dense(out_features=100, input_shape=(500, ), activation='relu'))
model.add(Dense(out_features=10))
model.compile(loss='mse', optimizer='adam')
print(model)
```



<div style="color: red" id="flatten"><i>- Flatten(start: int = 1, **kwargs)</i></div>

+ start: flatten start axis, for example, a tensor with shape (N, C, H, W), if start = 1, after flatten tensor will be (N, C * H * W); if start = 2,  after flatten tensor will be (N, C, H * W)

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Flatten, Input

X_input = Input(input_shape=(10, 5, 8))
X = Flatten(start=1)(X_input)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="pad"><i>- ZeroPadding2D(pad_size: Tuple, **kwargs)</i></div>

+ pad_size: for example, (1, 1), which means pad input(N, C, H, W) to (N, C, H+2, W+2).

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import ZeroPadding2D, Input

X_input = Input(input_shape=(10, 5, 5))
X = ZeroPadding2D(pad_size=(2, 2))(X_input)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="conv"><i>- Conv2D(out_channels: int, kernel_size: Tuple, use_bias: bool = False, stride: int = 1, padding: str = 'VALID', activation = None, kernel_initializer= 'Normal', bias_initializer = 'zeros', **kwargs)</i></div>

+ out_channels: filter's numbers.

+ kernel_size: filter's size. for example, (3, 3) or 3.

+ use_bias: whether use bias.

+ stride: convolution stride.

+ padding: 'SAME' or 'VALID', 'VALID' means no padding, 'SAME' means pad input to get the same output size as input.

+ activation: activation function. see details in <a href='#Activations'>Activations</a>

+ kernel_initializer: kernel initialize method. see details in <a href='#Initializers'>Initializers</a>

+ bias_initializer: bias initialize method. see details in <a href='#Initializers'>Initializers</a>

  ```python
  from xshinnosuke.models import Model
  from xshinnosuke.layers import Conv2D, Input
  
  X_input = Input(input_shape=(3, 24, 24))
  X = Conv2D(out_channels=16, kernel_size=(3, 3), stride=1, padding='VALID', activation='relu')(X_input)
  model = Model(inputs=X_input, outputs=X)
  model.compile(optimizer='sgd', loss='bce')
  print(model)
  ```

  

<div style="color: red" id="maxpool"><i>- MaxPooling2D(pool_size: Tuple, stride: int = None, **kwargs)</i></div>

+ pool_size: pooling kernel size, for example (2, 2) means apply max pooling in every 2 x 2 area.
+ stride: pooling stride.



<div style="color: red" id="meanpool"><i>- AvgPooling2D(pool_size: Tuple, stride: int = None, **kwargs)</i></div>

+ pool_size: pooling kernel size, for example (2, 2) means apply mean pooling in every 2 x 2 area.
+ stride: pooling stride.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import MaxPooling2D, AvgPooling2D, Input

X_input = Input(input_shape=(3, 24, 24))
X = MaxPooling2D(kernel_size=2)(X_input)
X = AvgPooling2D(kernel_size=2)(X)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='bce')
print(model)
```



<div style="color: red" id="act"><i>- Activation(act_name='relu')</i></div>

+ act_name: activation function name, support ReLU, Sigmoid, etc. see details in <a href='#Activations'>Activations</a>

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Activation, Input

X_input = Input(input_shape=(3, 24, 24))
X = Activation('relu')(X_input)
X = Activation('sigmoid')(X)
X = Activation('softmax')(X)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='cross_entropy')
print(model)
```



<div style="color: red" id="reshape"><i>- Reshape(shape: Tuple, inplace: bool = True, **kwargs)</i></div>

+ shape: shape after reshape operations.
+ inplace: apply reshape on the original data directly.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Reshape, Input

X_input = Input(input_shape=(3, 24, 24))
X = Reshape((3, 12, 12, 4))
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='cross_entropy')
print(model)
```



<div style="color: red" id="dropout"><i>- Dropout(keep_prob: float)</i></div>

+ keep_prob:  probability of keeping a unit active.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import Dropout, Input

X_input = Input(input_shape=(500, ))
X = Dropout(keep_prob=0.5)(X_input)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="bn"><i>- Batch Normalization(epsilon=1e-6, momentum=0.9, axis=1, gamma_initializer='ones', beta_initializer='zeros', moving_mean_initializer='zeros', moving_variance_initializer='ones')</i></div>
$$
u_B = \frac{1}{m} \sum \limits_{i=1}^m x_i  \quad \quad mini-batch \quad mean
\\
\sigma_B = \frac{1}{m} \sum \limits_{i=1}^m (x_i - u_B)^2  \quad \quad mini-batch \quad variance
\\
\hat x_i = \frac{x_i - u_B}{\sqrt{\sigma_B^2 + \epsilon}}   \quad \quad normalize
\\
y_i = \gamma \hat x_i + \beta  \quad \quad scale \quad and \quad shift
$$


+ epsilon:  $\epsilon$ value.
+ momentum: at training time, we use moving averages to update $u_B \rightarrow$ $moving\_u = momentum * moving\_u + (1 - momentum) * u_B$ and $\sigma_B \rightarrow  moving\_\sigma = momentum * moving\_\sigma + (1 - momentum) * \sigma_B$ 
+ axis: use normalization on which axis, for Dense Layer, it should be 1 or -1, for Convolution Layer, it should be 1.
+ gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
+ beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>
+ moving_mean_initializer: initialize $moving\_u$ method. see details in <a href='#Initializers'>Initializers</a>
+ moving_variance_initializer: initialize $moving\_\sigma$ method. see details in <a href='#Initializers'>Initializers</a>



<div style="color: red" id="ln"><i>- Layer Normalization(epsilon=1e-10, gamma_initializer='ones', beta_initializer='zeros')</i></div>
$$
u = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W x_{ijk}  \quad \quad sample \quad mean
\\
\sigma = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W (x_{ijk} - u)^2  \quad \quad sample \quad variance
\\
\hat x = \frac{x - u}{\sqrt{\sigma^2 + \epsilon}}   \quad \quad normalize
\\
y = \gamma \hat x + \beta  \quad \quad scale \quad and \quad shift
$$


+ epsilon:  $\epsilon$ value.
+ gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
+ beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>



<div style="color: red" id="gn"><i>- Group Normalization(epsilon=1e-5, G=16,gamma_initializer='ones', beta_initializer='zeros')</i></div>
split channel into G groups, for each group, applying layer normalization separately.
$$
\\
u = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W x_{ijk}  \quad \quad sample \quad mean
\\
\sigma = \frac{1}{CHW} \sum \limits_{i=1}^C \sum \limits_{j=1}^H \sum \limits_{k=1}^W (x_{ijk} - u)^2  \quad \quad sample \quad variance
\\
\hat x = \frac{x - u}{\sqrt{\sigma^2 + \epsilon}}   \quad \quad normalize
\\
y = \gamma \hat x + \beta  \quad \quad scale \quad and \quad shift
$$


+ epsilon:  $\epsilon$ value.
+ G: group numbers.
+ gamma_initializer: initialize $\gamma$ method. see details in <a href='#Initializers'>Initializers</a>
+ beta_initializer: initialize $\beta$ method. see details in <a href='#Initializers'>Initializers</a>

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import BatchNormalization, LayerNormalization, GroupNormalization, Input

X_input = Input(input_shape=(16, 5, 5))
X = BatchNormalization()(X_input)
X = LayerNormalization()(X)
X = GroupNormalization()(X)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="embed"><i>- Embedding(output_dim,embeddings_initializer='uniform', mask_zero=False, **kwargs)</i></div>

+ out_dim: after embedding dimension, for example, out_dim = E, input data (N, T) after embedding's shape is (N, T, E).
+ embeddings_initializer: embedding kernel initialize method. see details in <a href='#Initializers'>Initializers</a>
+ mask_zero: use masks.

```python
from xshinnosuke.models import Sequential
from xshinnosuke.layers import Embedding

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=200, input_length=30))
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="rnn"><i>- SimpleRNN(units, activation='tanh', initializer='glorotuniform', recurrent_initializer='orthogonal', return_sequences=False, return_state=False, stateful=False, **kwargs)</i></div>

$$
z^t = W_{aa}\cdot a^{t-1} + W_{xa}\cdot x^t +b_a
\\
a^t = activation(z^t)
$$

+ units: rnn hidden unit numbers, for example, units = a, input data (N, T, L) after rnn will output (N, T, a).
+ activation: activation method. see details in <a href='#Activations'>Activations</a>
+ initializer: $W_{xa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
+ recurrent_initializer: $W_{aa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
+ return_sequences: if True, return all timesteps a $\rightarrow$ $[a^1, a^2,..., a^t]$; if False, return the last timesteps $a^t$.
+ return_state: if True, return return_sequences' result and all timesteps a.
+ stateful: if True, use last time $a^t$ to initialize this time $a^1$; if False, use 0 to initialize this time $a^1$.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import SimpleRNN, Input

X_input = Input(input_shape=(30, 200))
X = SimpleRNN(units=50)(X_input)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="lstm"><i>- LSTM(units, activation='tanh', recurrent_activation='sigmoid', initializer='glorotuniform', recurrent_initializer='orthogonal', unit_forget_bias=True, return_sequences=False, return_state=False, stateful=False, **kwargs)</i></div>

at every timesteps

$$
i^t = recurrent\_activation(W_i[a^{t-1}, x^t] + b_i)
\\
f^t = recurrent\_activation(W_f[a^{t-1}, x^t] + b_f)
\\
\tilde c^t = activation(W_c[a^{t-1}, x^t] + b_c)
\\
c^t = f^t \cdot c^{t-1} + i^t \cdot \tilde c^t
\\
o^t = recurrent\_activation(W_o[a^{t-1}, x^t] + b_o)
\\
a^t = o^t \cdot tanh(c^t)
$$

+ units: lstm hidden unit numbers.
+ activation: activation method. see details in <a href='#Activations'>Activations</a>
+ recurrent_activation: activation method. see details in <a href='#Activations'>Activations</a>
+ initializer: $W_{xa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
+ recurrent_initializer: $W_{aa}$ initialize method. see details in <a href='#Initializers'>Initializers</a>
+ unit_forget_bias: if True, initialize $f^t$ bias $b_f$ as 1, else 0.
+ return_sequences: if True, return all timesteps a $\rightarrow$ $[a^1, a^2,..., a^t]$; if False, return the last timesteps $a^t$.
+ return_state: if True, return return_sequences' result and all timesteps a.
+ stateful: if True, use last time $a^t$ to initialize this time $a^1$; if False, use 0 to initialize this time $a^1$.

```python
from xshinnosuke.models import Model
from xshinnosuke.layers import LSTM, Input

X_input = Input(input_shape=(30, 200))
X = LSTM(units=50, return_sequences=True)(X_input)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```



<div style="color: red" id="timedist"><i>- TimeDistributed(layer, **kwargs)</i></div>

+ layer: to apply time distributed layer.

```python
from xshinnosuke.layers import Input, Dense, LSTM, TimeDistributed
from xshinnosuke.models import Model

X_input = Input(input_shape=(25, 97))
X = LSTM(units=100, return_sequences=True, stateful=True)(X_input)
X = TimeDistributed(Dense(50))(X)
model = Model(inputs=X_input, outputs=X)
model.compile(optimizer='sgd', loss='mse')
print(model)
```

---

## Node

<div style="color: red" id="variable"><i>- Variable(data, in_bounds: List = None, out_bounds: List = None, name: str = None, requires_grad: bool = True, dtype: str = 'float64')</i></div>

+ data: initialize value of this variable.
+ in_bounds: in_bound layer(s).
+ out_bounds: out_bound layer(s).
+ name: name of this variable.
+ requires_grad: whether requires gradient.
+ dtype: data type.



<div style="color: red" id="constant"><i>- Constant(data, in_bounds: List = None, out_bounds: List = None, name: str = None, requires_grad: bool = True, dtype: str = 'float64')</i></div>

+ data: initialize value of this variable.

+ in_bounds: in_bound layer(s).

+ out_bounds: out_bound layer(s).

+ name: name of this variable.

+ requires_grad: whether requires gradient.

+ dtype: data type.

   ```python
   from xshinnosuke.nn import Constant
   
   a = Constant(5)
   print('before: ', a)  
   a.data = 4
   print('after: ', a)
   # result in console
   '''
   before:  Constant(5.0, requires_grad=False)
   after:  Constant(5.0, requires_grad=False)
   UserWarning: Can not change the value of a Constant!
   '''
   ```



## Optimizers

+ SGD(lr=0.01, decay=0.0, *args, **kwargs)
  $$
  w = w - lr * dw
  \\
  b = b - lr * db 
  $$
  

  + lr: learning rate.
  + decay: learning rate decay.

  

+ Momentum(lr=0.01, decay=0.0, rho=0.9, *args, **kwargs)
  $$
  V_{dw} = rho * V_{dw} + (1 - rho) * dw
  \\
  V_{db} = rho * V_{db} + (1 - rho) * db
  \\
  w = w - lr * V_{dw}
  \\
  b = b - lr * V_{db}
  $$
  

  + lr: learning rate.
  + decay: learning rate decay.
  + rho: moving averages parameter.

  

+ RMSprop(lr=0.001, decay=0.0, rho=0.9, epsilon=1e-7, *args, **kwargs)
  $$
  S_{dw} = rho * S_{dw} + (1 - rho) * d_w^2
  \\
  S_{db} = rho * S_{db} + (1 - rho) * d_b^2
  \\
  w = w - lr * \frac{dw}{\sqrt{S_{dw} + \epsilon}}
  \\
  b = b - lr * \frac{db}{\sqrt{S_{db} + \epsilon}}
  $$
  

  + lr: learning rate.
  + decay: learning rate decay.
  + rho: moving averages parameter.
  + epsilon: $\epsilon$ value.

  

+ Adam(lr=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7, *args, **kwargs)
  $$
  V_{dw} = rho * V_{dw} + (1 - rho) * dw
  \\
  V_{db} = rho * V_{db} + (1 - rho) * db
  \\
  S_{dw} = rho * S_{dw} + (1 - rho) * d_w^2
  \\
  S_{db} = rho * S_{db} + (1 - rho) * d_b^2
  \\
  V_{dw}^{corrected} = \frac{V_{dw}}{1 - \beta_1^t}
  \\
  V_{db}^{corrected} = \frac{V_{db}}{1 - \beta_1^t}
  \\
  S_{dw}^{corrected} = \frac{S_{dw}}{1 - \beta_2^t}
  \\
  S_{db}^{corrected} = \frac{S_{db}}{1 - \beta_2^t}
  \\
  w = w - lr * \frac{V_{dw}^{corrected}}{S_{dw}^{corrected}}
  \\
  b = b - lr * \frac{V_{db}^{corrected}}{S_{db}^{corrected}}
  $$
  

  + lr: learning rate.

  + decay: learning rate decay.

  + beta1: $\beta_1$ value.

  + beta2: $\beta_2$ value.

  + epsilon: $\epsilon$ value.

    

## Objectives

+ *MeanSquaredError*
  + loss = $$\frac{1}{2}(y - \hat y)^2$$
+ *MeanAbsoluteError*
  + loss = $$|y - \hat y|$$
+ *BinaryCrossEntropy*
  + loss = $$-ylog\hat y -(1-y)log(1 - \hat y)$$
+ *SparseCrossEntropy*
  + loss = $$-\sum \limits_{c=1}^C y_clog\hat y_c$$
  + $y_c$ should be one-hot vector.
+ *CrossEntropy*
  + loss = $$-\sum \limits_{c=1}^C y_clog\hat y_c$$
  + $y_c$ can not be ont-hot vector.

<h2 id="Activations">Activations</h2>





<h2 id="Initializers">Initializers</h2>





## Utils