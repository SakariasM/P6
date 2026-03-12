import torch
from torch import nn
from torchvision import datasets
from data_loader import train_dataloader, test_dataloader



data = [[1, 2],[4, 5], [6,7]]
x_data = torch.tensor(data)

print(f"Ones Tensor: \n  {x_data}")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print("\n")
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")