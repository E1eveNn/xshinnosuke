import xs.nn
import xs.nn.functional
import xs.optim
from xs.layers import Conv2D, ReLU, BatchNormalization
import numpy as np
import torch.nn
import torch.nn.functional
import torch.optim
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def xs_network(weight1, bias1, bn_weight1, bn_bias1, weight2, bias2):
    weight1 = xs.nn.Parameter(weight1)
    bias1 = xs.nn.Parameter(bias1)
    bn_weight1 = xs.nn.Parameter(bn_weight1)
    bn_bias1 = xs.nn.Parameter(bn_bias1)
    weight2 = xs.nn.Parameter(weight2)
    bias2 = xs.nn.Parameter(bias2)
    l1 = Conv2D(out_channels=3, kernel_size=3)
    bn1 = BatchNormalization(epsilon=1e-5, momentum=0.9)
    a1 = ReLU()
    l2 = Conv2D(out_channels=5, kernel_size=3)
    l1.parameters([weight1, bias1])
    l2.parameters([weight2, bias2])
    bn1.parameters([bn_weight1, bn_bias1])
    bn1.moving_mean = xs.zeros(3)
    bn1.moving_variance = xs.ones(3)
    net = xs.nn.Sequential(l1, bn1, a1, l2)
    return net


def torch_network(weight1, bias1, bn_weight1, bn_bias1, weight2, bias2):
    weight1 = torch.nn.Parameter(torch.tensor(weight1), requires_grad=True)
    bias1 = torch.nn.Parameter(torch.tensor(bias1), requires_grad=True)
    bn_weight1 = torch.nn.Parameter(torch.tensor(bn_weight1), requires_grad=True)
    bn_bias1 = torch.nn.Parameter(torch.tensor(bn_bias1), requires_grad=True)
    weight2 = torch.nn.Parameter(torch.tensor(weight2), requires_grad=True)
    bias2 = torch.nn.Parameter(torch.tensor(bias2), requires_grad=True)
    l1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
    bn1 = torch.nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)
    a1 = torch.nn.ReLU()
    l2 = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
    l1.weight = weight1
    l1.bias = bias1
    bn1.weight = bn_weight1
    bn1.bias = bn_bias1
    l2.weight = weight2
    l2.bias = bias2
    net = torch.nn.Sequential(l1, bn1, a1, l2)
    return net


#########################  Hyper Parameters
EPOCH = 50
LR = 0.1
#########################  Read Data
np.random.seed(0)
train_datas = np.random.rand(2, 1, 6, 6).astype(np.float32)
train_labels = np.random.randint(0, 2, (2, ), dtype=np.int64)
weight1 = np.random.rand(3, 1, 3, 3).astype(np.float32)
bias1 = np.random.rand(3).astype(np.float32)
bn_weight1 = np.random.rand(3).astype(np.float32)
bn_bias1 = np.random.rand(3).astype(np.float32)
weight2 = np.random.rand(5, 3, 3, 3).astype(np.float32)
bias2 = np.random.rand(5).astype(np.float32)
#########################  Network
xs_net = xs_network(weight1, bias1, bn_weight1, bn_bias1, weight2, bias2)
torch_net = torch_network(weight1, bias1, bn_weight1, bn_bias1, weight2, bias2)
#########################  Optimizer
optim1 = xs.optim.Adam(xs_net.parameters(), lr=LR)
optim2 = torch.optim.Adam(torch_net.parameters(), lr=LR)


loss1_list = []
loss2_list = []
for i in range(EPOCH):
    print('#### Epoch ', i)
    xs_x, xs_y = xs.tensor(train_datas), xs.tensor(train_labels)
    torch_x, torch_y = torch.tensor(train_datas), torch.tensor(train_labels)
    optim1.zero_grad()
    optim2.zero_grad()
    pred1 = xs_net(xs_x).view(xs_x.size(0), -1, 2).mean(axis=1)
    pred2 = torch_net(torch_x).view(torch_x.size(0), -1, 2).mean(dim=1)
    # print('Prediction -->')
    # print('XS:\n', pred1)
    # print('Torch:\n', pred2)
    loss1 = xs.nn.functional.cross_entropy(pred1, xs_y)
    loss2 = torch.nn.functional.cross_entropy(pred2, torch_y)
    loss1_list.append(loss1.item())
    loss2_list.append(loss2.item())
    print('Loss -->')
    print('XS:\n', loss1)
    print('Torch:\n', loss2)
    loss1.backward()
    loss2.backward()
    optim1.step()
    optim2.step()

plt.figure()
plt.plot(loss1_list, label='xs_loss')
plt.plot(loss2_list, label='torch_loss')
plt.legend()
plt.show()
