import xs
import xs.nn as nn
from xs.layers import Dense
from xs.utils.toolkit import gradient_check

# set seed
xs.manual_seed_all(0)

# random generate data
x = xs.randn(3, 10, requires_grad=True)
y = xs.randn(3, 2)

# set criterion
criterion = nn.MSELoss()

# declare fully connected layer
dnn = Dense(out_features=2)
out = dnn(x)
loss = criterion(out, y)
loss.backward()

# get weight, bias from fully connected layer
weight, bias = dnn.parameters()
print("1.====================> DNN Backward")
print("weight grad: \n", weight.grad)
print("bias grad: \n", bias.grad)

print("2.====================> Gradient Check")
mathematical_weight_grad, mathematical_bias_grad = gradient_check(x, y, dnn, criterion)
print("weight grad: \n", mathematical_weight_grad)
print("bias grad: \n", mathematical_bias_grad)

print("3.====================> Distance between them")
weight_distance = xs.tensor(weight.grad.eval - mathematical_weight_grad).l2norm() / weight.numel()
bias_distance = xs.tensor(bias.grad.eval - mathematical_bias_grad).l2norm() / bias.numel()
print("weight l2 distance: ", weight_distance)
print("bias l2 distance: ", bias_distance)
