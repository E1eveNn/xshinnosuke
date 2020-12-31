from tests.gradient_check.util import np, draw_grad_line, compute_euclidean_distance


np.random.seed(0)
n_x = np.random.randn(3, 5).astype(np.float32)
n_y1 = np.random.randn(3, 5).astype(np.float32)
n_y2 = np.random.randint(0, 2, (3, 5)).astype(np.float32)


# pytorch
import torch.nn.functional
# declare
t_x = torch.tensor(n_x, requires_grad=True)
t_x.retain_grad()
t_y1 = torch.tensor(n_y1, requires_grad=True)
t_y1.retain_grad()
t_y2 = torch.tensor(n_y2)


# function
t_out1 = torch.nn.functional.relu(t_x)
t_out1.retain_grad()
t_out2 = torch.nn.functional.sigmoid(t_x)
t_out2.retain_grad()
t_out3 = torch.nn.functional.tanh(t_x)
t_out3.retain_grad()
t_loss1 = torch.nn.functional.mse_loss(t_out1, t_y1) / 2
t_loss2 = torch.nn.functional.binary_cross_entropy(t_out2, t_y2)
t_loss3 = torch.nn.functional.mse_loss(t_out3, t_y1) / 2
t_loss3.backward()
t_loss2.backward()
t_loss1.backward()
print("1.====================> Pytorch Backward")
print('loss3: ', t_loss3.item())
print('loss2: ', t_loss2.item())
print('loss1: ', t_loss1.item())
print('out3: ', t_out3.grad)
print('out2: ', t_out2.grad)
print('out1: ', t_out1.grad)
print('x grad: ', t_x.grad)


# xs
import xs.nn.functional
# declare
x_x = xs.tensor(n_x, requires_grad=True)
x_x.retain_grad(True)
x_y1 = xs.tensor(n_y1, requires_grad=True)
x_y1.retain_grad(True)
x_y2 = xs.tensor(n_y2, requires_grad=True)
x_y2.retain_grad(True)
# function
x_out1 = xs.nn.functional.relu(x_x)
x_out1.retain_grad(True)
x_out2 = xs.nn.functional.sigmoid(x_x)
x_out2.retain_grad(True)
x_out3 = xs.nn.functional.tanh(x_x)
x_out3.retain_grad(True)
x_loss1 = xs.nn.functional.mse_loss(x_out1, x_y1, reduction='mean')
x_loss2 = xs.nn.functional.bce_loss(x_out2, x_y2, reduction='mean')
x_loss3 = xs.nn.functional.mse_loss(x_out3, x_y1, reduction='mean')
x_loss3.backward()
x_loss2.backward()
x_loss1.backward()

print("2.====================> Xs Backward")
print('loss3: ', x_loss3.item())
print('loss2: ', x_loss2.item())
print('loss1: ', x_loss1.item())
print('out3: ', x_out3.grad)
print('out2: ', x_out2.grad)
print('out1: ', x_out1.grad)
print('x grad: ', x_x.grad)


print("3.====================> Draw grad line")
draw_grad_line(t_out3.grad.data.cpu().numpy(), x_out3.grad.eval, 'out3')
draw_grad_line(t_out2.grad.data.cpu().numpy(), x_out2.grad.eval, 'out2')
draw_grad_line(t_out1.grad.data.cpu().numpy(), x_out1.grad.eval, 'out1')
draw_grad_line(t_x.grad.data.cpu().numpy(), x_x.grad.eval, 'x')


print("4.====================> Compute Euclidean distance")
dist_y = compute_euclidean_distance(t_out3.grad.data.cpu().numpy(), x_out3.grad.eval)
print('The euclidean distance between pytorch and xs tensor out3\'s grad is: ', dist_y)
dist_y = compute_euclidean_distance(t_out2.grad.data.cpu().numpy(), x_out2.grad.eval)
print('The euclidean distance between pytorch and xs tensor out2\'s grad is: ', dist_y)
dist_y = compute_euclidean_distance(t_out1.grad.data.cpu().numpy(), x_out1.grad.eval)
print('The euclidean distance between pytorch and xs tensor out1\'s grad is: ', dist_y)
dist_x = compute_euclidean_distance(t_x.grad.data.cpu().numpy(), x_x.grad.eval)
print('The euclidean distance between pytorch and xs tensor x\'s grad is: ', dist_x)
