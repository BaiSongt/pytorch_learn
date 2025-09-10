
import torch

# --- 1. 回顾：基础梯度计算 ---
# PyTorch的autograd引擎可以自动计算标量函数相对于其输入的梯度。
# 创建一个张量并设置 requires_grad=True 来追踪其上的所有操作。
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2 + 2*y + 1
# dz / dx = 2x   x = 2  z = 4
# dz / dy = 2    y = 3  z = 2

# 当我们调用 .backward() 时，PyTorch会计算z关于所有requires_grad=True的叶子节点的梯度。
z.backward()

print("----- Basic Gradients -----")
print(f"Gradient of z with respect to x (dz/dx) at x=2: {x.grad}") # dz/dx = 2x = 4
print(f"Gradient of z with respect to y (dz/dy) at y=3: {y.grad}") # dz/dy = 2
print("-"*30)

# --- 2. 梯度累积 (Gradient Accumulation) ---
# 默认情况下，每次调用 .backward() 时，梯度会“累积”到 .grad 属性上。
# 这在训练神经网络时非常重要，因此我们必须在每个训练步骤开始时手动将梯度清零 (optimizer.zero_grad())。

# 让我们不清零梯度，再次调用 backward()
x.grad.zero_() # 先手动清零x的梯度，y不清零
y.grad.zero_() # 不清零y的梯度

z2 = x**3 + 4*y
z2.backward()  # 3x**2    4
print("--- Gradient Accumulation ---")
print(f"Gradient of x after first backward: {x.grad}") # dz2/dx = 3x^2 = 12
print(f"Gradient of y after first backward: {y.grad}") # 4


# 再次计算z2的梯度并累积
# z2.backward()
# PyTorch会报一个试图对已释放的计算图进行反向传播的错误，除非我们设置 retain_graph=True

print("--- Gradient Accumulation ---")
# 为了演示，我们重新构建计算图
x.grad.zero_()
y.grad.zero_()
z2_graph = x**3 + 4*y
z2_graph.backward(retain_graph=True) # retain_graph=True 保留计算图，以便再次反向传播
print(f"Gradient of y after first backward: {y.grad}") # dz2/dy = 4
z2_graph.backward()
print(f"Gradient of y after second backward (accumulation): {y.grad}") # 4 + 4 = 8
print("-"*30)

# --- 3. 非标量输出的梯度 (Jacobian Product) ---
# PyTorch的 `backward()` 设计为计算标量输出的梯度。
# 如果输出是一个向量（非标量），`backward()` 需要一个 `gradient` 参数，
# 这个参数的形状必须和输出张量的形状相同，代表了对输出的梯度权重。
# 这实际上是在计算“雅可比矩阵-向量积 (Jacobian-vector product)”。

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)
# z 现在是一个向量
z = x**2 + y**2

print("--- Gradients of Non-scalar Output ---")
# 如果直接调用 z.backward() 会报错，因为z不是标量
# RuntimeError: grad can be implicitly created only for scalar outputs

# 我们需要提供一个与z形状相同的权重向量v
v = torch.tensor([1.0, 1.0]) # 这个v通常被设为全1向量，相当于对z的各个分量求和后再求导
z.backward(gradient=v)

# 计算结果是 v^T * J (雅可比矩阵J的转置与向量v的乘积)
# J = [[dz1/dx1, dz1/dx2], [dz2/dx1, dz2/dx2]] = [[2x1, 0], [0, 2x2]]
# J = [[2, 0], [0, 4]]
# v^T * J = [1, 1] * [[2, 0], [0, 4]] = [2, 4]
print(f"Gradient of z with respect to x: {x.grad}") # [2*x1, 2*x2] = [2, 4]
print(f"Gradient of z with respect to y: {y.grad}") # [2*y1, 2*y2] = [6, 8]
print("-"*30)

# --- 4. 高阶导数 (Higher-Order Derivatives) ---
# PyTorch可以计算导数的导数。
# 我们需要使用 `torch.autograd.grad` 函数，它可以处理更高阶的导数计算。

x = torch.tensor(2.0, requires_grad=True)
y = x**3

print("--- Higher-Order Derivatives ---")
# `torch.autograd.grad(outputs, inputs, ...)`
# create_graph=True 非常关键，它会创建一个用于计算高阶导数的计算图。

# 计算一阶导数 (dy/dx = 3x^2 = 12)
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"First derivative (dy/dx) at x=2: {dy_dx}")

# 基于一阶导数的计算图，计算二阶导数 (d^2y/dx^2 = 6x = 12)
d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
print(f"Second derivative (d^2y/dx^2) at x=2: {d2y_dx2}")

# 计算三阶导数 (d^3y/dx^3 = 6)
d3y_dx3 = torch.autograd.grad(d2y_dx2, x)[0]
print(f"Third derivative (d^3y/dx^3) at x=2: {d3y_dx3}")
print("-"*30)

# --- 5. 停止追踪梯度 ---
# 有时我们不希望某些操作被autograd追踪，以节省内存或进行独立的计算。
# 可以使用 `torch.no_grad()` 上下文管理器或 `.detach()` 方法。

x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = y.detach() # z 是一个新的张量，它与y共享数据，但与计算图分离

print("--- Stopping Gradient Tracking ---")
print(f"x requires_grad: {x.requires_grad}")
print(f"y requires_grad: {y.requires_grad}")
print(f"z requires_grad: {z.requires_grad}") # z不再需要梯度

# 在 `torch.no_grad()` 环境中，所有操作都不会被追踪
with torch.no_grad():
    w = x * 2
    print(f"w requires_grad: {w.requires_grad}") # False
