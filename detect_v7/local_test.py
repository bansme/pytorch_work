# -*- coding: utf-8 -*-
import cv2
import torch


x = torch.Tensor(cv2.imread("inference/images/bus.jpg", -1))

print('-' * 50)
print(x)  # tensor([1., 2., 3., 4.])
print(x.size())  # torch.Size([4])
print(x.dim())  # 1
print(x.numpy())  # [1. 2. 3. 4.]

print('-' * 50)
print(torch.unsqueeze(x, 3))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, 3).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, 3).dim())  # 2
print(torch.unsqueeze(x, 3).numpy())  # [[1. 2. 3. 4.]]
