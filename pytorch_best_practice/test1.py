# -*- coding: utf-8 -*-
import torch

x=torch.tensor([1,2])
y=torch.tensor([3,4])
z=x.add(y)
print(z)
print(x)
x.add_(y)
print(x)


print(torch.Tensor([1,2,3,4,5,6]))
print(torch.Tensor(2,3))
t = torch.Tensor([[1,2,3],[4,5,6]])
print(t)
print(t.size())
print(t.shape)
print(torch.Tensor(t.size()))