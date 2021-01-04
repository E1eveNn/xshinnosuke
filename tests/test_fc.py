import xs.nn
import xs.nn.functional
import xs.optim
from xs.layers import Dense, ReLU
import numpy as np
import torch.nn
import torch.nn.functional
import torch.optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def xs_network(weight1, bias1, weight2, bias2):
    weight1 = xs.nn.Parameter(weight1)
    bias1 = xs.nn.Parameter(bias1)
    weight2 = xs.nn.Parameter(weight2)
    bias2 = xs.nn.Parameter(bias2)
    l1 = Dense(5, input_shape=(10,))
    a1 = ReLU()
    l2 = Dense(2)
    l1.parameters([weight1, bias1])
    l2.parameters([weight2, bias2])
    net = xs.nn.Sequential(l1, a1, l2)
    return net


def torch_network(weight1, bias1, weight2, bias2):
    weight1 = torch.nn.Parameter(torch.tensor(weight1.T), requires_grad=True)
    bias1 = torch.nn.Parameter(torch.tensor(bias1), requires_grad=True)
    weight2 = torch.nn.Parameter(torch.tensor(weight2.T), requires_grad=True)
    bias2 = torch.nn.Parameter(torch.tensor(bias2), requires_grad=True)
    l1 = torch.nn.Linear(10, 5)
    a1 = torch.nn.ReLU()
    l2 = torch.nn.Linear(5, 2)
    l1.weight = weight1
    l1.bias = bias1
    l2.weight = weight2
    l2.bias = bias2
    net = torch.nn.Sequential(l1, a1, l2)
    return net


#########################  Hyper Parameters
EPOCH = 3
LR = 0.5
#########################  Read Data
np.random.seed(0)
train_datas = np.random.rand(2, 10)
train_labels = np.random.randint(0, 2, (2, ), dtype=np.int64)
weight1 = np.random.rand(10, 5)
bias1 = np.random.rand(1, 5)
weight2 = np.random.rand(5, 2)
bias2 = np.random.rand(1, 2)
#########################  Network
xs_net = xs_network(weight1, bias1, weight2, bias2)
torch_net = torch_network(weight1, bias1, weight2, bias2)
#########################  Optimizer
optim1 = xs.optim.SGD(xs_net.parameters(), lr=LR)
optim2 = torch.optim.SGD(torch_net.parameters(), lr=LR)


for i in range(EPOCH):
    print('#### Epoch ', i)
    xs_x, xs_y = xs.tensor(train_datas), xs.tensor(train_labels)
    torch_x, torch_y = torch.tensor(train_datas), torch.tensor(train_labels)
    optim1.zero_grad()
    optim2.zero_grad()
    pred1 = xs_net(xs_x)
    pred2 = torch_net(torch_x)
    print('Prediction -->')
    print('XS:\n', pred1)
    print('Torch:\n', pred2)
    loss1 = xs.nn.functional.cross_entropy(pred1, xs_y)
    loss2 = torch.nn.functional.cross_entropy(pred2, torch_y)
    print('Loss -->')
    print('XS:\n', loss1)
    print('Torch:\n', loss2)
    loss1.backward()
    loss2.backward()
    optim1.step()
    optim2.step()
