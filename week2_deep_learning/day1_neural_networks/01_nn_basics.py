
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 前言 ---
# 神经网络是由许多简单的、相互连接的单元（称为“神经元”或“感知机”）组成的。
# 每个神经元接收一些输入，对它们进行加权求和，然后通过一个“激活函数”来产生输出。

# --- 1. 感知机 (Perceptron) 的 PyTorch 实现 ---
# 感知机是神经网络最基础的单元。它可以被看作是一个接收多个输入，并产生一个二进制输出的决策函数。
# y = f(Σ(wi * xi) + b)，其中 f 是一个阶跃函数。
# 在PyTorch中，这等价于一个输入维度为N、输出维度为1的线性层 (Linear Layer)。

print("--- 1. Perceptron as a Linear Layer ---")
# 假设我们有一个感知机，它接收3个输入特征
input_features = 3
output_features = 1

# nn.Linear(in_features, out_features) 创建一个线性变换 y = xA^T + b
perceptron_layer = nn.Linear(input_features, output_features)

# 我们可以查看随机初始化的权重(w)和偏置(b)
print(f"Weights (w): {perceptron_layer.weight}")
print(f"Bias (b): {perceptron_layer.bias}")

# 准备一个输入样本 (batch_size=1, num_features=3)
input_sample = torch.tensor([0.5, 0.2, -1.0])

# --- 2. 前向传播 (Forward Pass) ---
# 前向传播是指将输入数据通过网络层，计算得到输出的过程。
print("\n--- 2. Forward Pass ---")
linear_output = perceptron_layer(input_sample)
print(f"Output of the linear layer (weighted sum + bias): {linear_output.item():.4f}")

# --- 3. 激活函数 (Activation Function) ---
# 如果没有激活函数，多层神经网络本质上仍然是一个线性模型，无法学习复杂的非线性关系。
# 激活函数为网络引入了非线性，使其能够拟合任意复杂的函数。

# 经典的阶跃函数在数学上不连续，无法进行梯度计算，因此在实践中我们使用平滑的、可微的函数。

# **Sigmoid 激活函数**
# - 公式: σ(x) = 1 / (1 + e^(-x))
# - 输出范围: (0, 1)
# - 作用: 常用于二分类问题的输出层，将输出解释为概率。
activation_sigmoid = torch.sigmoid(linear_output)
print(f"Output after Sigmoid activation: {activation_sigmoid.item():.4f}")

# **ReLU (Rectified Linear Unit) 激活函数**
# - 公式: f(x) = max(0, x)
# - 输出范围: [0, +∞)
# - 作用: 目前在隐藏层中最常用的激活函数。它计算简单，能有效缓解梯度消失问题。
# F.relu 是一个函数式调用，等价于 `nn.ReLU()`
activation_relu = F.relu(linear_output)
print(f"Output after ReLU activation: {activation_relu.item():.4f}")

# --- 4. 构建一个简单的网络层 ---
# 通常，一个网络层由一个线性变换和紧随其后的一个非线性激活函数组成。
print("\n--- 4. Building a Simple Network Layer ---")

# 准备一批数据 (batch_size=4, num_features=3)
input_batch = torch.tensor([
    [0.5, 0.2, -1.0],
    [-0.1, 0.8, 0.3],
    [1.2, -0.5, 0.0],
    [0.0, 0.0, 1.0]
])

# 定义一个包含10个神经元的网络层
layer = nn.Linear(in_features=3, out_features=10)

# 前向传播
linear_output_batch = layer(input_batch)
print(f"Shape of linear output: {linear_output_batch.shape}") # (batch_size, out_features)

# 应用激活函数
activated_output_batch = F.relu(linear_output_batch)
print(f"Shape of activated output: {activated_output_batch.shape}")

# --- 5. 损失计算 (Loss Calculation) ---
# 神经网络通过比较其输出和真实标签（Ground Truth）的差异来学习。
# 这个差异由“损失函数”来量化。
print("\n--- 5. Loss Calculation ---")

# 假设我们的任务是回归，真实标签如下
true_labels = torch.randn(4, 10) # 假设每个样本有10个回归目标

# 使用均方误差 (MSE) 作为损失函数
loss_fn = nn.MSELoss()

# 计算损失
loss = loss_fn(activated_output_batch, true_labels)
print(f"Calculated Loss (MSE): {loss.item():.4f}")

# 这个loss值将是下一步“反向传播”的起点。

# 总结:
# 1. 神经网络的基本单元是“神经元”，在PyTorch中通常通过 `nn.Linear` 层来实现其线性部分。
# 2. **激活函数**为网络引入非线性，是网络能够学习复杂模式的关键。ReLU是隐藏层最常用的选择。
# 3. **前向传播**是将数据通过网络层计算输出的过程。
# 4. **损失函数**量化了模型预测与真实标签之间的差距，为模型优化指明了方向。

