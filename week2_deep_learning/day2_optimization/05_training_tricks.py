
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 前言 --- 
# 除了选择好的模型架构、优化器和损失函数，一些高级训练技巧可以进一步提升模型的性能和稳定性。
# 本脚本将介绍梯度裁剪和学习率预热等常用技巧。

# --- 1. 梯度裁剪 (Gradient Clipping) ---
# - **问题**: 在深度网络或RNN中，反向传播时梯度可能会因为链式法则而累积得非常大，
#   这种情况称为“梯度爆炸 (Exploding Gradients)”。巨大的梯度会导致权重更新过猛，使训练过程非常不稳定，甚至出现NaN值。
# - **解决方案**: 梯度裁剪。在优化器更新权重之前（即在 `optimizer.step()` 之前），检查所有参数的梯度。
#   如果梯度的范数（总大小）超过了一个预设的阈值，就按比例缩小所有梯度，使其范数等于该阈值。
# - **效果**: 这就像给梯度设置了一个“上限”，可以有效地防止梯度爆炸，稳定训练过程。

print("--- 1. Gradient Clipping ---")

# 创建一个简单的模型和优化器
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 模拟一次训练步骤
x = torch.randn(8, 10)
y_true = torch.randn(8, 1)

loss = nn.MSELoss()(model(x), y_true)
loss.backward() # 计算梯度

# 在没有裁剪的情况下，我们可以查看梯度的范数
grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Gradient norm before clipping: {grad_norm_before_clip:.4f}")

# **实现梯度裁剪**: `torch.nn.utils.clip_grad_norm_`
# `max_norm` 是你设置的阈值。这是一个需要调优的超参数，通常设为1.0或5.0。
max_grad_norm = 0.5
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Gradient norm after clipping to {max_grad_norm}: {grad_norm_after_clip:.4f}")

# 在训练循环中的位置:
# loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step()

print("-"*30)

# --- 2. 学习率预热 (Learning Rate Warmup) ---
# - **问题**: 在训练开始时，模型的参数是随机初始化的，离最优点很远。如果一开始就使用一个较大的学习率，
#   可能会导致训练在初期非常不稳定，甚至“震荡”出去。
# - **解决方案**: 学习率预热。在训练的最初几个周期（或几百个步骤）中，使用一个非常小的学习率，
#   然后线性地增加到你预设的正常学习率。之后，再由常规的学习率调度器（如CosineAnnealingLR）接管。
# - **效果**: 像“热身”一样，让模型在训练初期平稳地走向正确的优化方向，是训练大型模型（如Transformer）时的标准实践。

print("--- 2. Learning Rate Warmup ---")

# 模拟一个训练过程
initial_lr = 1e-7
final_lr = 0.1
warmup_epochs = 5
num_epochs = 20

optimizer = optim.SGD(model.parameters(), lr=initial_lr)
# 余弦退火调度器，在预热结束后接管
main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)

lr_history = []

for epoch in range(num_epochs):
    # 预热阶段
    if epoch < warmup_epochs:
        # 线性增加学习率
        lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # 预热结束后，由主调度器接管
        main_scheduler.step()
    
    lr_history.append(optimizer.param_groups[0]['lr'])

# 可视化学习率变化
plt.figure(figsize=(10, 6))
plt.plot(lr_history)
plt.title('Learning Rate Schedule with Warmup')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

print("-"*30)

# --- 3. 其他技巧回顾 ---

# **交叉验证 (Cross-Validation)**
# - 在 `week1_basics/day4_classification/04_model_optimization.py` 中已详细介绍。
# - **核心思想**: 与其只做一次训练/验证集划分，不如做K次，每次用不同的部分作验证集，
#   然后对K次的结果取平均。这能提供对模型泛化能力更稳定、更可靠的评估。
# - **应用**: 在进行模型选择或超参数调优时，是黄金标准。

# **模型集成 (Model Ensembling)**
# - 在 `week1_basics/day5_ensemble/03_ensemble_methods.py` 中已详细介绍。
# - **核心思想**: 训练多个不同的模型（或用不同数据、不同参数训练的同一种模型），
#   然后将它们的预测结果结合起来（如投票或取平均值）。
# - **应用**: 在机器学习竞赛中，这是提升最终性能的“大招”。在实际应用中，如果对性能要求极高且能接受更高的计算成本，可以采用此方法。

# 总结:
# 1. **梯度裁剪**是防止梯度爆炸、稳定训练过程的必备技巧，尤其适用于RNN和大型Transformer。
# 2. **学习率预热**是现代大型模型训练的标准实践，它能帮助模型在训练初期平稳过渡。
# 3. **交叉验证**和**模型集成**是更宏观的策略，分别用于获得可靠的性能评估和最大化最终性能。
# 4. 将这些技巧与合适的优化器、正则化方法结合，才能构建出最先进的深度学习训练流程。
