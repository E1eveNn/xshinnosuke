import torch
import xs
import torch.nn
import xs.layers
import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


################ test conv
print('##### Test Conv')
t = np.random.randn(32, 128, 5, 5)
z = np.random.randn(256, 128, 3, 3)

x1 = torch.tensor(t, dtype=torch.float32, requires_grad=False)
x2 = xs.tensor(t, requires_grad=False)

w1 = torch.tensor(z, dtype=torch.float32, requires_grad=False)
w2 = xs.tensor(z, requires_grad=False)

times = 100

t1 = time.time()
for i in range(times):
    out1 = torch.nn.Conv2d(128, 256, 3)(x1)
time_cost = time.time() - t1
avg1 = time_cost / times
print(f'torch conv time usage: {time_cost}, avg: {avg1}')

t1 = time.time()
for i in range(times):
    out1 = torch.nn.functional.conv2d(x1, w1)
time_cost = time.time() - t1
avg2 = time_cost / times
print(f'torch functional conv time usage: {time_cost}, avg: {avg2}')


t1 = time.time()
for i in range(times):
    out2 = xs.layers.Conv2D(256, 3)(x2)
time_cost = time.time() - t1
avg3 = time_cost / times
print(f'xs conv time usage: {time_cost}, avg: {avg3}')


t1 = time.time()
for i in range(times):
    out1 = xs.nn.functional.conv2d(x2, w2)
time_cost = time.time() - t1
avg4 = time_cost / times
print(f'xs functional conv time usage: {time_cost}, avg: {avg4}')

print(f'\nSpeed(Torch / Xs): {round(avg3 / avg1, 2)}x, {round(avg4 / avg2, 2)}x\n')

################ test fc
print('##### Test FC')
t = np.random.randn(128, 1000)
z = np.random.randn(1000, 10)

x1 = torch.tensor(t, dtype=torch.float32, requires_grad=False)
x2 = xs.tensor(t, requires_grad=False)

w1 = torch.tensor(z, dtype=torch.float32, requires_grad=False)
w2 = xs.tensor(z, requires_grad=False)

times = 100

t1 = time.time()
for i in range(times):
    out1 = torch.nn.Linear(1000, 10)(x1)
time_cost = time.time() - t1
avg1 = time_cost / times
print(f'torch fc time usage: {time_cost}, avg: {avg1}')

t1 = time.time()
for i in range(times):
    out1 = torch.mm(x1, w1)
time_cost = time.time() - t1
avg2 = time_cost / times
print(f'torch functional fc time usage: {time_cost}, avg: {avg2}')


t1 = time.time()
for i in range(times):
    out2 = xs.layers.Dense(10)(x2)
time_cost = time.time() - t1
avg3 = time_cost / times
print(f'xs fc time usage: {time_cost}, avg: {avg3}')


t1 = time.time()
for i in range(times):
    out1 = xs.nn.functional.mm(x2, w2)
time_cost = time.time() - t1
avg4 = time_cost / times
print(f'xs functional fc time usage: {time_cost}, avg: {avg4}')
print(f'\nSpeed(Torch / Xs): {round(avg3 / avg1, 2)}x, {round(avg4 / avg2, 2)}x\n')