
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- 前言 ---
# 优化器 (Optimizer) 是神经网络训练中的核心组件，它根据模型参数的梯度来更新参数值，以最小化损失函数。
# 选择合适的优化器和学习率策略对模型的收敛速度和最终性能至关重要。
# 本脚本将对比几种主流的优化器，并介绍学习率调度的使用。

# --- 1. 创建一个简单的优化问题 ---
# 我们将优化的目标函数设为一个简单的二维二次函数（一个碗状的曲面）。
# f(x, y) = x² + 2y²
# 这个函数的全局最小值在 (0, 0)。我们的目标是让优化器从一个随机点开始，找到这个最小值。

# `nn.Parameter` 是一种特殊的张量，当它被赋值给一个 `nn.Module` 的属性时，
# 它会自动被添加到模型的参数列表中（即会出现在 `model.parameters()` 的迭代器中）。
# 这里我们用它来表示我们要优化的变量。
params_to_optimize = nn.Parameter(torch.tensor([-4.0, 3.0])) # 起始点

def objective_function(params):
    x, y = params
    return x**2 + 2*y**2

# --- 2. 对比不同优化器 ---

# 辅助函数：用于执行优化和记录轨迹
def run_optimizer(optimizer_class, optimizer_params, n_steps=50):
    # 重新初始化参数
    params = nn.Parameter(torch.tensor([-4.0, 3.0]))
    optimizer = optimizer_class([params], **optimizer_params)
    path = []

    for _ in range(n_steps):
        path.append(params.data.clone().numpy())
        optimizer.zero_grad() # 清空梯度
        loss = objective_function(params)
        loss.backward() # 计算梯度
        optimizer.step() # 更新参数

    return np.array(path)

# **优化器1: SGD (随机梯度下降)**
# - 最基础的优化器，沿着梯度的反方向更新参数。
# - 缺点: 收敛速度慢，容易在峡谷状的函数中震荡。
path_sgd = run_optimizer(optim.SGD, {'lr': 0.1})

# **优化器2: SGD with Momentum (带动量的SGD)**
# - **思想**: 引入“动量”的概念，模拟物体运动的惯性。更新方向不仅取决于当前梯度，还取决于之前的更新方向。
# - **效果**: 可以加速收敛，并帮助“冲出”局部最小值或鞍点，减少震荡。
path_momentum = run_optimizer(optim.SGD, {'lr': 0.1, 'momentum': 0.9})

# **优化器3: Adam (Adaptive Moment Estimation)**
# - **思想**: 结合了 Momentum 和 RMSprop 的优点。它为每个参数计算自适应的学习率。
# - **效果**: 通常被认为是目前最常用、最稳健的优化器，在大多数情况下都能快速收敛并取得良好效果。是大多数任务的默认首选。
path_adam = run_optimizer(optim.Adam, {'lr': 0.3})

# 可视化优化轨迹
x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
_params = np.stack([x_grid, y_grid])
z_grid = _params[0]**2 + 2*_params[1]**2

plt.figure(figsize=(10, 7))
plt.contour(x_grid, y_grid, z_grid, levels=20)
plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'o-', label='SGD')
plt.plot(path_momentum[:, 0], path_momentum[:, 1], 'o-', label='SGD with Momentum')
plt.plot(path_adam[:, 0], path_adam[:, 1], 'o-', label='Adam')
plt.title('Optimizer Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --- 3. 学习率调度 (Learning Rate Scheduling) ---
# 在训练过程中动态地调整学习率是一种非常有效的策略。
# 通常，我们会在训练开始时使用较大的学习率以快速收敛，然后随着训练的进行逐渐减小学习率，以帮助模型在最优点附近进行更精细的调整。

print("\n--- Learning Rate Scheduling ---")

# 准备一个简单的模型和优化器
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# **调度器: StepLR**
# - `step_size`: 每隔多少个epoch，学习率就进行一次衰减。
# - `gamma`: 每次衰减时，将学习率乘以该因子。
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
for epoch in range(15):
    # 在每个epoch的**末尾**调用 scheduler.step()
    # optimizer.step() 应该在 scheduler.step() 之前被调用
    # (这里我们省略了训练循环，只演示调度器的行为)
    scheduler.step()
    print(f"Epoch {epoch+1}: Learning rate = {optimizer.param_groups[0]['lr']:.4f}")

# PyTorch提供了多种调度器，例如:
# - `CosineAnnealingLR`: 按余弦曲线退火学习率，被认为是一种非常有效的策略。
# - `ReduceLROnPlateau`: 当某个评估指标（如验证集损失）在一段时间内不再改善时，降低学习率。

# 总结:
# 1. **Adam** 是大多数深度学习任务的默认首选优化器，因为它通常收敛快且性能稳健。
# 2. **SGD with Momentum** 仍然是一个非常强大的优化器，在一些研究中，精调的SGD可能会比Adam获得更好的最终性能。
# 3. **学习率调度** 是一个关键的训练技巧。在训练后期降低学习率有助于模型收敛到更好的最优点。
# 4. `StepLR` 是最简单的调度器之一，`CosineAnnealingLR` 和 `ReduceLROnPlateau` 是更常用、更高级的选择。
