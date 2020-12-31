from tests.gradient_check.util import np, draw_grad_line, compute_euclidean_distance


np.random.seed(0)
n_x1 = np.random.randn(3, 5).astype(np.float32)
n_x2 = np.random.randn(3, 5).astype(np.float32)
n_x3 = np.random.randn(3, 5).astype(np.float32)
n_y1 = np.random.randn(3, 5).astype(np.float32)
n_y2 = np.random.randint(0, 2, (3, 5)).astype(np.float32)
n_y3 = np.random.randint(0, 5, (3,)).astype(np.int64)


# pytorch
import torch.nn.functional
# declare
t_x1 = torch.tensor(n_x1, requires_grad=True)
t_x1.retain_grad()
t_x2 = torch.tensor(n_x2, requires_grad=True)
t_x2.retain_grad()
t_x3 = torch.tensor(n_x3, requires_grad=True)
t_x3.retain_grad()
t_y1 = torch.tensor(n_y1, requires_grad=True)
t_y1.retain_grad()
t_y2 = torch.tensor(n_y2)
t_y3 = torch.tensor(n_y3)


# function
t_loss1 = torch.nn.functional.mse_loss(t_x1, t_y1) / 2
t_loss2 = torch.nn.functional.binary_cross_entropy(t_x2.sigmoid(), t_y2)
t_loss3 = torch.nn.functional.cross_entropy(t_x3, t_y3, reduction='mean')
t_loss3.backward()
t_loss2.backward()
t_loss1.backward()
print("1.====================> Pytorch Backward")
print('loss3: ', t_loss3.item())
print('loss2: ', t_loss2.item())
print('loss1: ', t_loss1.item())
print('x3: ', t_x3.grad)
print('x2: ', t_x2.grad)
print('x1: ', t_x1.grad)


# xs
import xs.nn.functional
# declare
x_x1 = xs.tensor(n_x1, requires_grad=True)
x_x1.retain_grad(True)
x_x2 = xs.tensor(n_x2, requires_grad=True)
x_x2.retain_grad(True)
x_x3 = xs.tensor(n_x3, requires_grad=True)
x_x3.retain_grad(True)
x_y1 = xs.tensor(n_y1, requires_grad=True)
x_y1.retain_grad(True)
x_y2 = xs.tensor(n_y2, requires_grad=True)
x_y2.retain_grad(True)
x_y3 = xs.tensor(n_y3, requires_grad=True)
x_y3.retain_grad(True)
# function
x_loss1 = xs.nn.functional.mse_loss(x_x1, x_y1, reduction='mean')
x_loss2 = xs.nn.functional.bce_loss(x_x2.sigmoid(), x_y2, reduction='mean')
x_loss3 = xs.nn.functional.cross_entropy(x_x3, x_y3, reduction='mean')
x_loss3.backward()
x_loss2.backward()
x_loss1.backward()

print("2.====================> Xs Backward")
print('loss3: ', x_loss3.item())
print('loss2: ', x_loss2.item())
print('loss1: ', x_loss1.item())
print('x3: ', x_x3.grad)
print('x2: ', x_x2.grad)
print('x1: ', x_x1.grad)


print("3.====================> Draw grad line")
draw_grad_line(t_x3.grad.data.cpu().numpy(), x_x3.grad.eval, 'x3')
draw_grad_line(t_x2.grad.data.cpu().numpy(), x_x2.grad.eval, 'x2')
draw_grad_line(t_x1.grad.data.cpu().numpy(), x_x1.grad.eval, 'x1')



print("4.====================> Compute Euclidean distance")
dist_y = compute_euclidean_distance(t_x3.grad.data.cpu().numpy(), x_x3.grad.eval)
print('The euclidean distance between pytorch and xs tensor x3\'s grad is: ', dist_y)
dist_y = compute_euclidean_distance(t_x2.grad.data.cpu().numpy(), x_x2.grad.eval)
print('The euclidean distance between pytorch and xs tensor x2\'s grad is: ', dist_y)
dist_y = compute_euclidean_distance(t_x1.grad.data.cpu().numpy(), x_x1.grad.eval)
print('The euclidean distance between pytorch and xs tensor x1\'s grad is: ', dist_y)

