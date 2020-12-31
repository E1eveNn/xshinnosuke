from tests.gradient_check.util import np, draw_grad_line, compute_euclidean_distance


np.random.seed(0)
n_x = np.random.randn(3, 10).astype(np.float32)
n_y = np.random.randn(3, 2).astype(np.float32)
n_weight = np.random.randn(10, 2).astype(np.float32)
n_bias = np.random.randn(1, 2).astype(np.float32)


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
t_out = torch.nn.functional.linear(t_x, t_weight.t(), t_bias)
t_loss = torch.nn.functional.mse_loss(t_out, t_y, reduction='mean')
t_loss.backward()
print("1.====================> Pytorch Backward")
print('loss: ', t_loss.item())
print('y grad: ', t_y.grad)
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
x_out = xs.nn.functional.addmm(x_bias, x_x, x_weight)
x_loss = xs.nn.functional.mse_loss(x_out, x_y, reduction='mean')
x_loss.backward()
print("2.====================> Xs Backward")
print('loss: ', x_loss.item())
print('y grad: ', x_y.grad)
print('weight grad: ', x_weight.grad)
print('bias grad: ', x_bias.grad)
print('x grad: ', x_x.grad)


print("3.====================> Draw grad line")
draw_grad_line(t_y.grad.data.cpu().numpy(), x_y.grad.eval, 'y')
draw_grad_line(t_weight.grad.data.cpu().numpy(), x_weight.grad.eval, 'weight')
draw_grad_line(t_bias.grad.data.cpu().numpy(), x_bias.grad.eval, 'bias')
draw_grad_line(t_x.grad.data.cpu().numpy(), x_x.grad.eval, 'x')


print("4.====================> Compute Euclidean distance")
dist_y = compute_euclidean_distance(t_y.grad.data.cpu().numpy(), x_y.grad.eval)
print('The euclidean distance between pytorch and xs tensor y\'s grad is: ', dist_y)
dist_weight = compute_euclidean_distance(t_weight.grad.data.cpu().numpy(), x_weight.grad.eval)
print('The euclidean distance between pytorch and xs tensor weight\'s grad is: ', dist_weight)
dist_bias = compute_euclidean_distance(t_bias.grad.data.cpu().numpy(), x_bias.grad.eval)
print('The euclidean distance between pytorch and xs tensor bias\'s grad is: ', dist_bias)
dist_x = compute_euclidean_distance(t_x.grad.data.cpu().numpy(), x_x.grad.eval)
print('The euclidean distance between pytorch and xs tensor x\'s grad is: ', dist_x)
