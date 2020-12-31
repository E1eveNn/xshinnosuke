import xs
import xs.nn as nn
from xs.layers import BatchNormalization, Flatten
from xs.utils.toolkit import gradient_check

# set seed
xs.manual_seed_all(0)

# random generate data
x = xs.randn(1, 3, 5, 5, requires_grad=True)
y = xs.randn(1, 75)

# set criterion
criterion = nn.MSELoss()

bn = nn.Sequential(
    # declare BatchNormalization layer
    BatchNormalization(),
    Flatten()
)


out = bn(x)
loss = criterion(out, y)
loss.backward()

# get weight, bias from convolutional layer
gamma, beta = bn.parameters()
print("1.====================> BN Backward")
print("weight grad: \n", gamma.grad)
print("bias grad: \n", beta.grad)

print("2.====================> Gradient Check")
mathematical_weight_grad, mathematical_bias_grad = gradient_check(x, y, bn, criterion)
print("weight grad: \n", mathematical_weight_grad)
print("bias grad: \n", mathematical_bias_grad)

print("3.====================> Distance between them")
weight_distance = xs.tensor(gamma.grad.eval - mathematical_weight_grad).l2norm() / gamma.numel()
bias_distance = xs.tensor(beta.grad.eval - mathematical_bias_grad).l2norm() / beta.numel()
print("weight l2 distance: ", weight_distance)
print("bias l2 distance: ", bias_distance)
