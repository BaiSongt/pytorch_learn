
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- 前言 ---
# 训练Transformer模型有一些独特的挑战，例如模型巨大、容易过拟合、对学习率敏感等。
# “Attention Is All You Need”论文不仅提出了模型架构，还介绍了一套行之有效的优化策略，
# 这些策略对于成功训练Transformer至关重要。

# --- 1. 优化器 (Optimizer) ---
# 论文中使用了 Adam 优化器，但设置了特定的超参数：
# - beta1 = 0.9
# - beta2 = 0.98
# - epsilon = 10e-9
# 这组参数被证明在训练Transformer时非常稳健。

# model = MyTransformer()
# optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
# 注意：初始学习率lr设为0，因为我们将使用自定义的学习率调度器来动态控制它。

print("--- 1. Use Adam with specific betas: (0.9, 0.98) ---")
print("-"*30)

# --- 2. 自定义学习率调度 (Custom Learning Rate Schedule) ---
# 这是训练Transformer最关键的技巧之一。
# 公式: lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
#
# 它包含两个阶段：
# 1. **线性预热 (Warmup)**: 在前 `warmup_steps` 步，学习率从0线性增加。
# 2. **平方根倒数衰减 (Decay)**: 在预热之后，学习率与训练步数的平方根成反比衰减。

class CustomLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        """在每个训练步后调用，以更新学习率。"""
        self.current_step += 1
        arg1 = self.current_step ** -0.5
        arg2 = self.current_step * (self.warmup_steps ** -1.5)

        lr = (self.d_model ** -0.5) * min(arg1, arg2)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 可视化学习率曲线
d_model = 512
warmup_steps = 4000
steps = 40000

# 模拟一个优化器和调度器
dummy_optimizer = optim.Adam([nn.Parameter(torch.randn(2,2))], lr=0)
scheduler = CustomLRScheduler(dummy_optimizer, d_model, warmup_steps)
lr_history = []
for _ in range(steps):
    scheduler.step()
    lr_history.append(scheduler.optimizer.param_groups[0]['lr'])

plt.figure(figsize=(10, 6))
plt.plot(lr_history)
plt.title("Transformer Learning Rate Schedule")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()

print("--- 2. Custom LR Schedule: Linear warmup then inverse sqrt decay. ---")
print("-"*30)

# --- 3. 标签平滑 (Label Smoothing) ---
# - **问题**: 在分类任务中，使用标准的 one-hot 编码作为目标（例如，正确类别为1，其他为0），
#   会促使模型产生“过于自信”的预测（例如，对正确类别的预测概率趋近于1.0）。
#   这可能导致模型泛化能力变差，对噪声数据更敏感。
# - **解决方案**: 标签平滑。将硬性的1和0标签，用“软”标签来替代。
#   例如，对于一个5分类问题，如果正确类别是2，one-hot标签是 [0, 0, 1, 0, 0]。
#   经过平滑后（假设平滑因子ε=0.1），目标标签变为：
#   - 正确类别2的概率: 1 - ε = 0.9
#   - 其他每个错误类别的概率: ε / (num_classes - 1) = 0.1 / 4 = 0.025
#   - 新的标签: [0.025, 0.025, 0.9, 0.025, 0.025]
# - **效果**: 这是一种正则化技术，可以防止模型变得过于自信，提高模型的准确率和校准度。

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        # 使用KL散度损失，它衡量两个概率分布之间的差异
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, prediction_logits, target_indices):
        # prediction_logits: (batch_size, num_classes)
        # target_indices: (batch_size)

        # 1. 将模型的logits转换为log概率
        log_probs = F.log_softmax(prediction_logits, dim=-1)

        # 2. 创建平滑后的目标分布
        # a. 创建一个one-hot编码的目标张量
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        # b. 将正确类别的位置填充为 1 - smoothing
        true_dist.scatter_(1, target_indices.unsqueeze(1), 1.0 - self.smoothing)

        # 3. 计算KL散度损失
        return self.criterion(log_probs, true_dist)

print("--- 3. Label Smoothing regularizes the model to be less confident. ---")

# 使用示例
num_classes = 5
loss_fn = LabelSmoothingLoss(num_classes, smoothing=0.1)

# 虚拟输入
pred = torch.randn(2, num_classes) # 2个样本
target = torch.tensor([0, 2]) # 真实类别

loss = loss_fn(pred, target)
print(f"Calculated Label Smoothing Loss: {loss.item():.4f}")

# 总结:
# 1. **优化器**: 使用带有特定beta值的Adam优化器。
# 2. **学习率**: 使用“预热+衰减”的自定义学习率调度是成功的关键。
# 3. **损失函数**: 使用标签平滑可以作为一种有效的正则化手段，提升模型性能。
# 这些技术共同构成了训练Transformer的标准“配方”。
