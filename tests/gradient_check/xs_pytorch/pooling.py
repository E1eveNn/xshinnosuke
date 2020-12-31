from tests.gradient_check.util import np, draw_grad_line, compute_euclidean_distance


np.random.seed(0)
n_x = np.random.randn(1, 3, 5, 5).astype(np.float32)
n_y = np.random.randn(1).astype(np.float32)


# pytorch
import torch.nn.functional
# declare
t_x = torch.tensor(n_x, requires_grad=True)
t_x.retain_grad()
t_y = torch.tensor(n_y, requires_grad=True)
t_y.retain_grad()

# function
t_out1 = torch.nn.functional.max_pool2d(t_x, 3, 2)
t_out1.retain_grad()
t_out2 = torch.nn.functional.avg_pool2d(t_x, 3, 2)
t_out2.retain_grad()
t_out3 = t_out1 + t_out2
t_out3.retain_grad()
t_out4 = t_out3.mean().view(1)
t_loss = torch.nn.functional.mse_loss(t_out4, t_y, reduction='mean') / 2
t_loss.backward()
print("1.====================> Pytorch Backward")
print('loss: ', t_loss.item())
print('out3: ', t_out3.grad)
print('out2: ', t_out2.grad)
print('out1: ', t_out1.grad)
print('x grad: ', t_x.grad)


# xs
import xs.nn.functional
# declare
x_x = xs.tensor(n_x, requires_grad=True)
x_x.retain_grad(True)
x_y = xs.tensor(n_y, requires_grad=True)
x_y.retain_grad(True)
# function
x_out1 = xs.nn.functional.max_pool2d(x_x, 3, 2)
x_out1.retain_grad(True)
x_out2 = xs.nn.functional.avg_pool2d(x_x, 3, 2)
x_out2.retain_grad(True)
x_out3 = x_out1 + x_out2
x_out3.retain_grad(True)
x_out4 = x_out3.mean()
x_loss = xs.nn.functional.mse_loss(x_out4, x_y, reduction='mean')
x_loss.backward()
print("2.====================> Xs Backward")
print('loss: ', x_loss.item())
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
