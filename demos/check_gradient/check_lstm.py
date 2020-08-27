import xshinnosuke as xs
from xshinnosuke.layers import LSTM
from xshinnosuke.utils import gradient_check

# set seed
xs.manual_seed_all(0)

# random generate data
x = xs.randn(1, 4, 5)
y = xs.randint(0, 2, (1, 1))

# set criterion
criterion = xs.nn.CrossEntropy()

# declare LSTM
lstm = LSTM(units=2)
out = lstm(x)
loss = criterion(out, y)
loss.backward()

# get weight, bias from fully connected layer
weight, bias = lstm.parameters()
print("1.====================> LSTM Backward")
print("weight grad: ", weight.grad)
print("bias grad: ", bias.grad)

print("2.====================> Gradient Check")
mathematical_weight_grad, mathematical_bias_grad = gradient_check(x, y, lstm, criterion)
print("weight grad: \n", mathematical_weight_grad)
print("bias grad: \n", mathematical_bias_grad)

print("3.====================> Distance between them")
weight_distance = xs.tensor(weight.grad - mathematical_weight_grad).l2norm() / weight.numel()
bias_distance = xs.tensor(bias.grad - mathematical_bias_grad).l2norm() / bias.numel()
print("weight l2 distance: ", weight_distance)
print("bias l2 distance: ", bias_distance)
