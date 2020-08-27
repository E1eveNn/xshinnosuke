import xshinnosuke as xs
from xshinnosuke.layers import Dense
from xshinnosuke.utils import gradient_check

# set seed
xs.manual_seed_all(0)

# random generate data
x = xs.randn(3, 5)
y = xs.randn(1, 3)

# set criterion
criterion = xs.nn.MeanSquaredError()

# declare fully connected layer
dnn = Dense(out_features=3)
out = dnn(x)
loss = criterion(out, y)
loss.backward()

# get weight, bias from fully connected layer
weight, bias = dnn.parameters()
print("1.====================> DNN Backward")
print("weight grad: ", weight.grad)
print("bias grad: ", bias.grad)

print("2.====================> Gradient Check")
mathematical_weight_grad, mathematical_bias_grad = gradient_check(x, y, dnn, criterion)
print("weight grad: \n", mathematical_weight_grad)
print("bias grad: \n", mathematical_bias_grad)

print("3.====================> Distance between them")
weight_distance = xs.tensor(weight.grad - mathematical_weight_grad).l2norm() / weight.numel()
bias_distance = xs.tensor(bias.grad - mathematical_bias_grad).l2norm() / bias.numel()
print("weight l2 distance: ", weight_distance)
print("bias l2 distance: ", bias_distance)
