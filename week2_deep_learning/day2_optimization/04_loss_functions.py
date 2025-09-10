
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 前言 --- 
# 损失函数（也叫代价函数或目标函数）用于衡量模型预测的“糟糕”程度。
# 训练模型的过程，就是通过优化器调整参数，来最小化损失函数值的过程。
# PyTorch在 `torch.nn` 模块中提供了大量常用的损失函数。

# --- 1. 分类损失 (Classification Losses) ---

print("--- 1. Classification Losses ---")

# **A. 多分类交叉熵损失 (Multi-class Cross-Entropy)**
# - **用途**: 多分类问题（一个样本只属于N个类别中的一个）。
# - **输入**: 
#   - `prediction`: 模型的原始输出（logits），形状为 (N, C)，其中N是批次大小，C是类别总数。
#   - `target`: 真实的类别标签，形状为 (N)，值为 0 到 C-1 的整数。
# - **注意**: `nn.CrossEntropyLoss` 内部自动应用了 `LogSoftmax` 和 `NLLLoss`。因此，你的模型输出层**不需要**加Softmax激活函数。

loss_ce = nn.CrossEntropyLoss()

# 示例: 3个样本，4个类别
predictions_mc = torch.randn(3, 4, requires_grad=True) # 3个样本，4个类别的logits
targets_mc = torch.tensor([1, 0, 3]) # 真实标签

output_ce = loss_ce(predictions_mc, targets_mc)
print(f"Multi-class Cross Entropy Loss: {output_ce.item():.4f}")

# **B. 二分类/多标签交叉熵损失 (Binary/Multi-label Cross-Entropy)**
# - **用途**: 
#   1. 二分类问题（一个样本属于两类中的一类）。
#   2. 多标签分类问题（一个样本可以同时属于多个类别）。
# - **输入**: 
#   - `prediction`: 模型的原始输出（logits），形状为 (N, C)。
#   - `target`: 真实的标签，形状必须与prediction相同，值为0或1的浮点数。
# - **注意**: `nn.BCEWithLogitsLoss` 内部自动应用了 `Sigmoid` 激活。因此，你的模型输出层**不需要**加Sigmoid激活函数。

loss_bce = nn.BCEWithLogitsLoss()

# 示例: 3个样本，4个类别（多标签）
predictions_ml = torch.randn(3, 4, requires_grad=True) # 3个样本，4个类别的logits
targets_ml = torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32) # 真实标签

output_bce = loss_bce(predictions_ml, targets_ml)
print(f"Binary/Multi-label Cross Entropy Loss: {output_bce.item():.4f}")
print("-"*30)

# --- 2. 回归损失 (Regression Losses) ---

print("--- 2. Regression Losses ---")

# **A. 均方误差损失 (Mean Squared Error, MSE)**
# - **用途**: 回归任务。
# - **公式**: L(y, ŷ) = Σ(y - ŷ)² / n
# - **特点**: 对误差进行平方，因此对大的误差（异常值）给予更高的惩罚。这是最常用的回归损失。

loss_mse = nn.MSELoss()

predictions_reg = torch.randn(5, 1, requires_grad=True)
targets_reg = torch.randn(5, 1)

output_mse = loss_mse(predictions_reg, targets_reg)
print(f"Mean Squared Error (MSE) Loss: {output_mse.item():.4f}")

# **B. 平均绝对误差损失 (Mean Absolute Error, L1 Loss)**
# - **用途**: 回归任务。
# - **公式**: L(y, ŷ) = Σ|y - ŷ| / n
# - **特点**: 对所有误差给予相同的权重，对异常值比MSE更鲁棒（不那么敏感）。

loss_l1 = nn.L1Loss()

output_l1 = loss_l1(predictions_reg, targets_reg)
print(f"Mean Absolute Error (L1) Loss: {output_l1.item():.4f}")
print("-"*30)

# --- 3. 自定义损失函数 (Custom Loss Functions) ---
# 有时，你需要一个PyTorch没有提供的、针对特定任务的损失函数。
# 你可以像创建模型一样，通过继承 `nn.Module` 来轻松创建自己的损失函数。

print("--- 3. Custom Loss Functions ---")

class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        # 确保权重是一个张量
        self.weight = torch.tensor(weight, dtype=torch.float32)

    def forward(self, prediction, target):
        # 对正的真实值和负的真实值施加不同的权重
        # 这里我们假设，如果真实值>0，我们更关心它的误差
        # (这只是一个例子，你可以实现任何你想要的逻辑)
        se = (prediction - target)**2
        # F.relu(target) 会让所有负值变为0，正值不变
        weighted_se = se * F.relu(target) * self.weight + se * (1 - F.relu(target))
        return torch.mean(weighted_se)

# 使用自定义损失
custom_loss_fn = WeightedMSELoss(weight=5.0)
output_custom = custom_loss_fn(predictions_reg, targets_reg)
print(f"Custom Weighted MSE Loss: {output_custom.item():.4f}")

# 总结:
# 1. **任务决定损失函数**: 
#    - 多分类 -> `nn.CrossEntropyLoss`
#    - 二分类/多标签 -> `nn.BCEWithLogitsLoss`
#    - 回归 -> `nn.MSELoss` (首选) 或 `nn.L1Loss` (如果异常值多)
# 2. 注意PyTorch损失函数（如CrossEntropyLoss, BCEWithLogitsLoss）的**内置激活函数**，避免在模型末端重复添加Softmax或Sigmoid。
# 3. 通过继承 `nn.Module`，你可以灵活地创建任何满足你需求的复杂损失函数。
