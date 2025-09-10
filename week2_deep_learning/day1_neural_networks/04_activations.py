
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- 前言 --- 
# 激活函数为神经网络引入了非线性，使其能够学习并拟合复杂的非线性关系。
# 如果没有激活函数，一个多层的神经网络本质上仍然是一个线性模型。
# 本脚本将可视化并分析几种最常用的激活函数及其梯度。

# 创建一个输入张量用于演示
x = torch.linspace(-6, 6, 100, requires_grad=True)

# --- 1. Sigmoid --- 
# - 公式: f(x) = 1 / (1 + exp(-x))
# - 输出范围: (0, 1)
# - 优点: 输出在0到1之间，可以用作二分类问题的输出层，表示概率。
# - 缺点:
#   1. **梯度消失 (Vanishing Gradients)**: 当输入非常大或非常小时，其梯度（导数）接近于0。在深层网络中，这会导致反向传播时梯度逐层递减，最终消失，使得网络难以训练。
#   2. 输出不是零中心的 (Not zero-centered)，可能导致收敛速度变慢。
y_sigmoid = torch.sigmoid(x)

# --- 2. Tanh (双曲正切) ---
# - 公式: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# - 输出范围: (-1, 1)
# - 优点: 输出是零中心的，通常比Sigmoid收敛更快。
# - 缺点: 仍然存在梯度消失问题。
y_tanh = torch.tanh(x)

# --- 3. ReLU (Rectified Linear Unit) ---
# - 公式: f(x) = max(0, x)
# - 输出范围: [0, +∞)
# - 优点:
#   1. 在正数部分，梯度恒为1，极大地缓解了梯度消失问题。
#   2. 计算速度极快。
#   3. 使网络具有稀疏性（一些神经元输出为0）。
# - 缺点:
#   1. **Dying ReLU Problem**: 如果一个神经元的输入恒为负，那么它的输出将永远是0，梯度也永远是0，这个神经元就“死亡”了，无法再通过梯度下降进行更新。
#   2. 输出不是零中心的。
y_relu = F.relu(x)

# --- 4. Leaky ReLU ---
# - 公式: f(x) = x if x > 0 else alpha * x (alpha是一个很小的正数，如0.01)
# - 优点: 解决了Dying ReLU问题，因为负数部分的梯度不再是0，而是alpha。
# - 缺点: 性能不一定总是比ReLU好，alpha值的选择也增加了一个超参数。
y_leaky_relu = F.leaky_relu(x, negative_slope=0.1)

# --- 5. GELU (Gaussian Error Linear Unit) ---
# - 公式: f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
# - 优点: 在大型模型（如Transformer、BERT、GPT）中表现优异，被认为是ReLU的更平滑、性能更好的替代品。
# - 缺点: 计算比ReLU复杂。
y_gelu = F.gelu(x)

# --- 6. 可视化激活函数及其梯度 ---

# 计算梯度
y_sigmoid.backward(torch.ones_like(x), retain_graph=True)
grad_sigmoid = x.grad.clone(); x.grad.zero_()

y_tanh.backward(torch.ones_like(x), retain_graph=True)
grad_tanh = x.grad.clone(); x.grad.zero_()

y_relu.backward(torch.ones_like(x), retain_graph=True)
grad_relu = x.grad.clone(); x.grad.zero_()

y_leaky_relu.backward(torch.ones_like(x), retain_graph=True)
grad_leaky_relu = x.grad.clone(); x.grad.zero_()

y_gelu.backward(torch.ones_like(x), retain_graph=True)
grad_gelu = x.grad.clone(); x.grad.zero_()

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 绘制函数图像
axes[0].plot(x.detach().numpy(), y_sigmoid.detach().numpy(), label="Sigmoid")
axes[0].plot(x.detach().numpy(), y_tanh.detach().numpy(), label="Tanh")
axes[0].plot(x.detach().numpy(), y_relu.detach().numpy(), label="ReLU")
axes[0].plot(x.detach().numpy(), y_leaky_relu.detach().numpy(), label="Leaky ReLU (a=0.1)")
axes[0].plot(x.detach().numpy(), y_gelu.detach().numpy(), label="GELU")
axes[0].set_title("Activation Functions")
axes[0].set_ylabel("Output")
axes[0].grid(True)
axes[0].legend()

# 绘制梯度图像
axes[1].plot(x.detach().numpy(), grad_sigmoid, label="Sigmoid Gradient")
axes[1].plot(x.detach().numpy(), grad_tanh, label="Tanh Gradient")
axes[1].plot(x.detach().numpy(), grad_relu, label="ReLU Gradient")
axes[1].plot(x.detach().numpy(), grad_leaky_relu, label="Leaky ReLU Gradient")
axes[1].plot(x.detach().numpy(), grad_gelu, label="GELU Gradient")
axes[1].set_title("Gradients of Activation Functions")
axes[1].set_xlabel("Input")
axes[1].set_ylabel("Gradient")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# --- 7. 如何选择？---
# 1. **首选ReLU**: 它是绝大多数现代神经网络隐藏层的默认和标准选择。从ReLU开始，通常能获得不错的结果。
# 2. **尝试Leaky ReLU / PReLU / ELU**: 如果你的网络遇到了大量的“死亡”神经元，可以尝试使用ReLU的这些变体。
# 3. **GELU**: 如果你在构建Transformer或类似的最新架构，GELU是当前的最佳实践。
# 4. **慎用Sigmoid/Tanh**: 它们在隐藏层中已经很少使用，主要因为梯度消失问题。Tanh偶尔用于需要[-1, 1]范围输出的特定场景（如GAN的生成器）。Sigmoid主要用于二分类问题的输出层。
