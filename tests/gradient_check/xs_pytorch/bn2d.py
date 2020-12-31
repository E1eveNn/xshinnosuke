from tests.gradient_check.util import np, draw_grad_line, compute_euclidean_distance


np.random.seed(0)
n_x = np.random.randn(1, 3, 2, 2).astype(np.float32)
n_y = np.random.randn(1).astype(np.float32)
n_weight = np.random.randn(3).astype(np.float32)
n_bias = np.random.randn(3).astype(np.float32)


# pytorch
import torch.nn.functional
# declare
t_x = torch.tensor(n_x, requires_grad=True)
t_x.retain_grad()
t_y = torch.tensor(n_y, requires_grad=True)
t_y.retain_grad()
t_weight = torch.tensor(n_weight, requires_grad=True)
t_weight.retain_grad()
t_bias = torch.tensor(n_bias, requires_grad=True)
t_bias.retain_grad()
# function
t_out1 = torch.nn.functional.batch_norm(t_x, torch.zeros(3), torch.zeros(3), t_weight, t_bias, training=True)
t_out1.retain_grad()
t_out2 = t_out1.sum().view(1,)
t_out2.retain_grad()
t_loss = torch.nn.functional.mse_loss(t_out2, t_y, reduction='mean') / 2
t_loss.backward()
print("1.====================> Pytorch Backward")
print('loss: ', t_loss.item())
print('out1: ', t_out1.grad)
print('weight grad: ', t_weight.grad)
print('bias grad: ', t_bias.grad)
print('x grad: ', t_x.grad)


# xs
import xs.nn.functional
# declare
x_x = xs.tensor(n_x, requires_grad=True)
x_x.retain_grad(True)
x_y = xs.tensor(n_y, requires_grad=True)
x_y.retain_grad(True)
x_weight = xs.tensor(n_weight, requires_grad=True)
x_weight.retain_grad(True)
x_bias = xs.tensor(n_bias, requires_grad=True)
x_bias.retain_grad(True)
# function
x_out1 = xs.nn.functional.batch_norm(x_x, x_weight, x_bias, xs.zeros(3), xs.zeros(3))
x_out1.retain_grad(True)
x_out2 = x_out1.sum()
x_out2.retain_grad(True)
x_loss = xs.nn.functional.mse_loss(x_out2, x_y, reduction='mean')
x_loss.backward()
print("2.====================> Xs Backward")
print('loss: ', x_loss.item())
print('out1: ', x_out1.grad)
print('weight grad: ', x_weight.grad)
print('bias grad: ', x_bias.grad)
print('x grad: ', x_x.grad)


print("3.====================> Draw grad line")
draw_grad_line(t_out1.grad.data.cpu().numpy(), x_out1.grad.eval, 'out1')
draw_grad_line(t_weight.grad.data.cpu().numpy(), x_weight.grad.eval, 'weight')
draw_grad_line(t_bias.grad.data.cpu().numpy(), x_bias.grad.eval, 'bias')
draw_grad_line(t_x.grad.data.cpu().numpy(), x_x.grad.eval, 'x')


print("4.====================> Compute Euclidean distance")
dist_y = compute_euclidean_distance(t_out1.grad.data.cpu().numpy(), x_out1.grad.eval)
print('The euclidean distance between pytorch and xs tensor out1\'s grad is: ', dist_y)
dist_weight = compute_euclidean_distance(t_weight.grad.data.cpu().numpy(), x_weight.grad.eval)
print('The euclidean distance between pytorch and xs tensor weight\'s grad is: ', dist_weight)
dist_bias = compute_euclidean_distance(t_bias.grad.data.cpu().numpy(), x_bias.grad.eval)
print('The euclidean distance between pytorch and xs tensor bias\'s grad is: ', dist_bias)
dist_x = compute_euclidean_distance(t_x.grad.data.cpu().numpy(), x_x.grad.eval)
print('The euclidean distance between pytorch and xs tensor x\'s grad is: ', dist_x)
