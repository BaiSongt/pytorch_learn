import torch

# 检查是否有可用的 CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建张量并移动到gpu
x = torch.tensor([1, 2, 3], device="cuda")
print(f"x: {x}")

x = x.cpu()
# x = x.to("cpu")
print(f"x: {x}, {x.device}")

# host 创建
x = torch.tensor([1, 2, 3])  # 默认在 CPU
x = x.to("cuda")  # 移动到 GPU
# 或者
x = x.cuda()  # 简写，等价于 .to('cuda')
print(f"x: {x}")


# 创建两个张量并移动到 GPU
a = torch.tensor([1, 2, 3], device='cuda')
b = torch.tensor([4, 5, 6], device='cuda')

# 加法
c = a + b
print(c)

# 乘法（逐元素相乘）
d = a * b
print(d)

# 矩阵乘法（如果是二维张量）
A = torch.randn(2, 3, device='cuda')
B = torch.randn(3, 2, device='cuda')
C = torch.matmul(A, B)  # 或者 A @ B
print(C)


##
x_gpu = torch.tensor([1, 2, 3], device='cuda')
x_cpu = x_gpu.cpu()  # 移回 CPU
x_np = x_cpu.numpy()  # 转为 NumPy（只能在 CPU 上进行）
print(x_np)


# 进行一些基本运算
c = a + b
d = a * b
e = torch.sum(c)

print("a:", a)
print("b:", b)
print("a + b =", c)
print("a * b =", d)
print("sum of (a + b) =", e)

