import numpy as np
import torch

## tensor operation

has_cuda = torch.cuda.is_available()
if has_cuda:
    print("torch cuda verison: ", torch.version.cuda)  # 应输出 12.6
    print(torch.cuda.get_device_capability(0))


## basic

### + - * /
tensor = torch.tensor([1, 2, 3])
print(f"tensor {tensor}")
print(f"tensor + 10 = {tensor + 10}")
print(f"tensor - 10 = {tensor - 10}")
print(f"tensor * 10 = {tensor + 10}")
print(f"tensor / 10 = {tensor / 10}")

### multipy
print(f"multiply:     {torch.multiply(tensor, 10)}")

### tensor * tensor by position
print(f"tensor = {tensor}")
print(f"tensor * tensor:")
print(tensor * tensor)

## Matrix

### vector
vector1 = torch.tensor([1, 2, 3])
print(vector1.shape)

result = vector1.matmul(vector1)
print(result)
print(result.shape)
### use @
print(vector1 @ vector1)


### matrix   n x m multipy m x n
matrix1 = torch.tensor([[1, 2, 3], [1, 2, 3]])
matrix2 = torch.tensor([[1, 2], [2, 3], [3, 4]])
print(matrix1.shape)
print(matrix2.shape)
## error 2x3 and 2x3
print(f"m @ m {matrix1 @ matrix1.T}")
print(f"m1 @ m2 {matrix1 @ matrix2}")
print(f"m1 @ m2 {matrix1.matmul(matrix2)}")


##
x = torch.arange(0, 100, 10)
print(f"x: {x}")
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}")  # won't work without float datatype
print(f"Sum: {x.sum()}")

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

## broadcast

a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状 (2,3)
b = 2  # 标量（视为形状 (1,1)）
result = a + b  # b 被广播为 (2,3)
print(f"result: {result}")
print(result.shape)

a = torch.ones(2, 1, 3)  # 形状 (2,1,3)
b = torch.ones(4, 3)  # 形状 (4,3)
result = a + b  # a 扩展为 (2,4,3)，b 扩展为 (2,4,3)
print(f"result: {result}")
print(result.shape)

a = torch.tensor([[10], [20], [30]])  # 形状 (3,1)
b = torch.tensor([1, 2, 3])  # 形状 (3,)
result = a + b  # a 扩展为 (3,3)，b 扩展为 (3,3)
print(f"result: {result}")
print(result.shape)

result = torch.broadcast_to(result, (3, 3))
print(f"result: {result}")
print(result.shape)


# PyTorch 的原位操作（In-place Operation）​是指直接修改张量本身的值，
# 而非创建新张量的操作。这类操作通过函数名后缀 _标记
# （如 add_()、mul_()），其核心特点是内存共享和计算图修改。
# 以下是详细解析：

x = torch.tensor([1.0, 2.0, 3.0])
x.add_(2)  # 直接修改 x 的值，结果为 [3.0, 4.0, 5.0]
print(f"x: {x}")
