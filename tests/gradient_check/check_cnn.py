import xs
import xs.nn as nn
from xs.layers import Conv2D, Flatten
from xs.utils.toolkit import gradient_check

# set seed
xs.manual_seed_all(0)

# random generate data
x = xs.randn(1, 3, 5, 5, requires_grad=True)
y = xs.randn(1, 9)

# set criterion
criterion = nn.MSELoss()

cnn = nn.Sequential(
    # declare convolutional layer
    Conv2D(out_channels=1, kernel_size=3, use_bias=True),
    # add flatten function
    Flatten()
)

out = cnn(x)
loss = criterion(out, y)
loss.backward()

# get weight, bias from convolutional layer
weight, bias = cnn.parameters()
print("1.====================> CNN Backward")
print("weight grad: \n", weight.grad)
print("bias grad: \n", bias.grad)

print("2.====================> Gradient Check")
mathematical_weight_grad, mathematical_bias_grad = gradient_check(x, y, cnn, criterion)
print("weight grad: \n", mathematical_weight_grad)
print("bias grad: \n", mathematical_bias_grad)

print("3.====================> Distance between them")
weight_distance = xs.tensor(weight.grad.eval - mathematical_weight_grad).l2norm() / weight.numel()
bias_distance = xs.tensor(bias.grad.eval - mathematical_bias_grad).l2norm() / bias.numel()
print("weight l2 distance: ", weight_distance)
print("bias l2 distance: ", bias_distance)
