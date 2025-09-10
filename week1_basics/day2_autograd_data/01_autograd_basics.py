
# 一、计算图概念
# 计算图是 PyTorch 自动求导的核心，由 ​节点（Node）​​ 和 ​边（Edge）​​ 组成：
# ​节点​：表示数据（张量）或操作（如加法、乘法）
# ​边​：表示数据流动方向

# 动态图特性
# PyTorch 使用 ​动态计算图，即计算图在代码执行时动态构建，支持条件分支和循环结构。
import torch

# 构建计算图
x = torch.tensor(2.0, requires_grad=True)  # 叶子节点
w = torch.tensor(1.0, requires_grad=True)  # 叶子节点
a = x + w    # 中间节点（加法操作）
b = w + 1    # 中间节点
y = a * b    # 输出节点（乘法操作）

print("计算图结构：")
print(f"x.grad_fn: {x.grad_fn}")  # None（叶子节点）
print(f"y.grad_fn: {y.grad_fn}")  # MulBackward0（乘法操作）


# 二、requires_grad设置
# 控制张量是否参与梯度计算：
# ​requires_grad=True​：记录操作以构建计算图
# ​requires_grad=False​：不记录操作，节省内存

# 创建张量并设置 requires_grad
x = torch.tensor([1.0, 2.0], requires_grad=True)  # 需要梯度
y = torch.tensor([3.0, 4.0], requires_grad=False) # 不需要梯度

z = x * y  # z.requires_grad=True（因x.requires_grad=True）
print(f"z.requires_grad: {z.requires_grad}")  # True

# 修改 requires_grad 属性
with torch.no_grad():  # 禁用梯度计算
    y.requires_grad_(True)  # 动态修改为需要梯度
print(f"修改后 y.requires_grad: {y.requires_grad}")  # True

# 三、backward()使用
# 反向传播计算梯度，核心参数：
# ​gradient​：非标量损失的梯度权重（默认标量损失不需要）
# ​retain_graph​：保留计算图以多次反向传播
# ​create_graph​：支持高阶导数计算

# 标量损失反向传播
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()  # 自动计算 dy/dx = 2x
print(f"梯度: {x.grad}")  # tensor(4.)

# 非标量损失反向传播（需指定 gradient）
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
loss = y.mean()  # 标量 shape 为 []
print(f"loss: {loss} {loss.shape}")
gradient=torch.tensor(0.5)
loss.backward(gradient)  # 梯度加权
print(f"梯度: {x.grad}")  # tensor([2., 2.])

# 保留计算图进行多次反向传播
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)  # 保留计算图
print(f"第一次梯度: {x.grad}")  # tensor(12.)
y.backward()  # 再次反向传播（梯度累加）
print(f"第二次梯度: {x.grad}")  # tensor(24.)

# 四、grad属性与梯度访问
# ​tensor.grad​：存储梯度值（仅 requires_grad=True的张量有效）
# ​梯度累积​：默认梯度会累加，需手动清零

# 梯度累积与清零
w = torch.tensor(1.0, requires_grad=True)
for _ in range(3):
    y = w ** 2
    y.backward()  # 梯度累加
    print(f"累积梯度: {w.grad}")  # tensor(2.), tensor(4.), tensor(6.)

# 手动清零梯度
w.grad.zero_()
print(f"清零后梯度: {w.grad}")  # None

# 高阶导数计算
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
grad1 = torch.autograd.grad(y, x, create_graph=True)  # 一阶导数
grad2 = torch.autograd.grad(grad1[0], x)  # 二阶导数
print(f"一阶导数: {grad1[0]}")  # tensor(4.)
print(f"二阶导数: {grad2[0]}")  # tensor(2.)


# 五、关键注意事项
# 1.​叶子节点​：用户直接创建的张量（如 x = torch.tensor(...)），其梯度保留，
#   非叶子节点梯度在反向传播后被释放。

# 2.梯度清零​：训练循环中需在 backward()前调用 optimizer.zero_grad()。

# 3.​内存管理​：使用 with torch.no_grad()禁用梯度计算以节省内存。
