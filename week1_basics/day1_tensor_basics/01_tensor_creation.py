import os
import numpy as np
import torch

print(torch.__version__)
print("torch cuda available: ", torch.cuda.is_available())  # 应返回 True

has_cuda = torch.cuda.is_available()
if has_cuda:
    print("torch cuda verison: ", torch.version.cuda)  # 应输出 12.6
    print(torch.cuda.get_device_capability(0))

## create tensor

tensor1 = torch.tensor([[1, -1], [-1, 1]])
print(tensor1)
print(f"ndim {tensor1.ndim}")
print(tensor1.tolist())
print("=" * 20)

tensor2 = torch.tensor(np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]]))
print(tensor2)
print(f"ndim {tensor2.ndim}")
print(tensor1.tolist())
print("=" * 20)

tensor1 = torch.tensor(3)
print(tensor1)
print(f"ndim {tensor1.ndim}")
print(tensor1.item())
print(tensor1.tolist())
print("=" * 20)

## zeros and ones

zeros1 = torch.zeros([2, 4], dtype=torch.int32)
print(zeros1)
print("=" * 20)

if has_cuda:
    cuda0 = torch.device("cuda:0")
    ones = torch.ones([2, 4], dtype=torch.float64, device=cuda0)
    print(ones)

ones = torch.ones([2, 4], dtype=torch.float64)
print(ones)
print(f"ndim: {ones.ndim}")
print("=" * 20)

## rand

random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(f"dtype = {random_tensor.dtype}")
print(f" size = {random_tensor.shape}")
print("=" * 20)

random_tensor = torch.randn(size=(2, 3))
print(random_tensor)
print(f"dtype = {random_tensor.dtype}")
print(f" size = {random_tensor.shape}")
print("=" * 20)


## range
zero_to_ten = torch.range(0, 10, dtype=torch.int32)
print(zero_to_ten)

zero_to_ten = torch.arange(start=0, end=10, step=2)
print(zero_to_ten)

## linspace
range_tensor = torch.linspace(start=0., end=10., steps=10)
print("range_tensor ", range_tensor)

## tensors like

five_zeros = torch.zeros_like(input=zero_to_ten)
print(five_zeros)

five_ones = torch.ones_like(input=zero_to_ten)
print(five_ones)


float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded

print(float_32_tensor.shape)
print(float_32_tensor.dtype)
print(float_32_tensor.device)
