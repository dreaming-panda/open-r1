import torch
a = torch.rand(2, 3, 3).requires_grad_(True)
b = torch.rand(2, 3, 3)
c = torch.bmm(a, b)
loss = c.sum()
print(loss)
loss.backward()