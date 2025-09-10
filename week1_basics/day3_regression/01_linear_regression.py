
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 准备数据 ---
# 线性回归的目标是找到 y = wx + b 这条直线，以最好地拟合数据点。
# 我们首先手动创建一组带有一些噪声的线性数据。

# 创建x坐标
X_numpy = np.linspace(0, 10, 100) # 从0到10生成100个点
# 假设真实的权重w=2, 偏置b=1, 并添加一些随机噪声
y_numpy = 2 * X_numpy + 1 + np.random.randn(100) * 1.5

# 将numpy数组转换为PyTorch张量
# 模型输入需要是 (样本数, 特征数) 的形状，这里是 (100, 1)
X_tensor = torch.from_numpy(X_numpy.astype(np.float32)).view(-1, 1)
y_tensor = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

print(f"Input shape: {X_tensor.shape}")
print(f"Output shape: {y_tensor.shape}")

# --- 2. 定义模型 ---
# 在PyTorch中，我们可以使用 `torch.nn.Module` 来构建模型。
# 对于线性回归，我们只需要一个线性层 `nn.Linear`。

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层
        # input_dim: 输入特征的数量 (我们的例子中是1，因为只有x)
        # output_dim: 输出特征的数量 (我们的例子中是1，因为只有y)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # forward方法定义了数据如何通过模型层
        # 在这里，输入x直接通过线性层得到输出
        return self.linear(x)

# 实例化模型
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

print("\n--- Model Definition ---")
print(model)
# 我们可以查看模型初始化的随机权重和偏置
# model.parameters()返回一个迭代器，包含模型所有可学习的参数
[w, b] = model.parameters()
print(f"Initial random weight: {w.item():.4f}")
print(f"Initial random bias: {b.item():.4f}")

# --- 3. 定义损失函数和优化器 ---

# 损失函数 (Loss Function) 用于衡量模型预测值与真实值之间的差距。
# 对于回归问题，最常用的是均方误差 (Mean Squared Error, MSE)。
criterion = nn.MSELoss()

# 优化器 (Optimizer) 用于根据损失函数的反向传播梯度来更新模型的参数（权重和偏置）。
# 随机梯度下降 (Stochastic Gradient Descent, SGD) 是最基础的优化器之一。
# - model.parameters(): 告诉优化器需要更新哪些参数。
# - lr (learning_rate): 学习率，控制每次参数更新的步长。
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# --- 4. 训练模型 ---
# 训练过程就是一个迭代循环，不断地让模型在数据上进行预测，计算损失，然后更新参数。

num_epochs = 100 # 定义训练的轮次

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    # 1. 前向传播：模型根据输入X进行预测
    y_predicted = model(X_tensor)
    
    # 2. 计算损失
    loss = criterion(y_predicted, y_tensor)
    
    # 3. 反向传播和优化
    # a. 清空之前的梯度（非常重要，否则梯度会累积）
    optimizer.zero_grad()
    # b. 计算梯度
    loss.backward()
    # c. 更新权重
    optimizer.step()
    
    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        # 我们可以在`torch.no_grad()`上下文中执行代码，以防止PyTorch跟踪这些操作的梯度
        # 这对于推理或打印信息时很有用，可以节省计算资源
        with torch.no_grad():
            [w, b] = model.parameters()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learned w: {w.item():.4f}, Learned b: {b.item():.4f}')

print("--- Training Finished ---")

# --- 5. 评估和可视化 ---
# 训练完成后，我们来看看模型学习到的直线是什么样的。

# 将模型设置为评估模式 (evaluation mode)
# 这对于某些层（如Dropout, BatchNorm）很重要，在线性回归中虽非必需，但是个好习惯。
model.eval()

with torch.no_grad():
    # 用训练好的模型进行预测
    predicted = model(X_tensor).detach().numpy()

# 绘制原始数据点
plt.scatter(X_numpy, y_numpy, label='Original Data', alpha=0.6)
# 绘制拟合的直线
plt.plot(X_numpy, predicted, 'r', label='Fitted Line')
# 绘制真实的直线
plt.plot(X_numpy, 2 * X_numpy + 1, 'g--', label='True Line')

plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 总结:
# 线性回归的PyTorch实现流程：
# 1. 准备数据 (X, y)，并转换为Tensor。
# 2. 定义模型 (nn.Module)，通常是一个nn.Linear层。
# 3. 定义损失函数 (如 nn.MSELoss) 和优化器 (如 torch.optim.SGD)。
# 4. 编写训练循环：前向传播 -> 计算损失 -> 清空梯度 -> 反向传播 -> 更新参数。
# 5. 评估模型，可视化结果。
