
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import os

# --- 前言 ---
# 迁移学习是计算机视觉中一种非常强大且流行的技术。
# 我们不是从零开始训练一个网络，而是使用一个在大型数据集（如ImageNet，包含120万张图片和1000个类别）
# 上已经训练好的模型，并将其知识“迁移”到我们自己的、通常小得多的数据集上。
# 预训练模型的前几层已经学会了识别通用的特征（如边缘、纹理、形状），我们可以复用这些知识。

# --- 1. 准备数据 ---
# 迁移学习通常用于我们自己的数据集。这里，我们使用一个PyTorch提供的虚拟数据集作为示例。
# 假设我们的新任务是区分“蚂蚁”和“蜜蜂”。
# `hymenoptera_data` 是一个包含 `train` 和 `val` 两个子文件夹的小型数据集。
# 你需要先下载它: https://download.pytorch.org/tutorial/hymenoptera_data.zip

# 定义数据变换
# 对于预训练模型，其输入数据必须经过与它在ImageNet上训练时完全相同的预处理。
# 通常包括：缩放到224x224，标准化等。
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 2. 加载预训练模型 ---
# `torchvision.models` 提供了许多预训练好的模型，如 ResNet, VGG, MobileNet等。

print("--- 2. Loading Pre-trained Model ---")
# `pretrained=True` 会下载并加载在ImageNet上训练好的权重
model_ft = models.resnet18(pretrained=True)
print("ResNet-18 model loaded.")

# --- 3. 修改模型以适应新任务 (微调策略) ---

# **策略: 冻结卷积层，只训练全连接层（分类头）**
# 这是最常见的微调策略。我们假设预训练的卷积层已经是非常好的特征提取器，
# 我们不需要重新训练它们，只需要替换并训练最后的分类部分。

# **步骤1: 冻结所有参数**
# 我们首先将模型的所有参数的 `requires_grad` 属性设为 `False`，这样在反向传播时就不会计算它们的梯度。
for param in model_ft.parameters():
    param.requires_grad = False

# **步骤2: 替换分类头**
# ResNet的最后一层是一个名为 `fc` 的全连接层。
num_ftrs = model_ft.fc.in_features # 获取原始fc层的输入特征数

# 将原始的fc层替换为一个新的、为我们任务定制的nn.Linear层。
# 我们的任务只有2个类别（蚂蚁和蜜蜂）。
# 新创建的层的参数默认 `requires_grad=True`。
model_ft.fc = nn.Linear(num_ftrs, 2)

print("\n--- 3. Model Modified for Finetuning ---")
print("All layers frozen except the final 'fc' layer.")
print("New fc layer:", model_ft.fc)

# 收集需要更新的参数
# 优化器只会更新那些 `requires_grad=True` 的参数。
params_to_update = []
print("\nParameters to be finetuned:")
for name, param in model_ft.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print(f"\t{name}")
print("---" * 30)

# --- 4. 训练和评估 ---
# 这部分代码展示了一个完整的训练循环，但需要你有下载好的数据集才能运行。
# 如果你没有数据集，可以阅读代码来理解其逻辑。

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# ... (完整的训练和验证循环代码) ...
# 假设我们已经训练好了，并加载了最佳模型权重
# model_ft.load_state_dict(torch.load('best_model.pth'))

print("--- 4. Training Logic (Conceptual) ---")
print("The model would now be trained on the new dataset.")
print("Only the weights of the final 'fc' layer will be updated by the optimizer.")

# --- 5. 另一种策略：训练整个模型 ---
# 如果你的数据集很大（例如超过1万张图片），或者你的数据与ImageNet的数据差异很大，
# 你可以考虑解冻更多的层，甚至训练整个模型。
# 这种情况下，通常会给不同的层设置不同的学习率：
# - 靠前的卷积层使用一个非常小的学习率。
# - 靠后的层和新的分类头使用一个较大的学习率。
#
# optimizer = optim.SGD([
#     {'params': model.conv_layers.parameters(), 'lr': 1e-5},
#     {'params': model.fc.parameters(), 'lr': 1e-3}
# ], lr=1e-2, momentum=0.9)

# 总结:
# 1. 迁移学习是解决小数据集问题的强大武器。
# 2. 从 `torchvision.models` 加载 `pretrained=True` 的模型是第一步。
# 3. 最常见的策略是**冻结特征提取层**，只**替换和训练分类头**。
# 4. 确保你的数据预处理（特别是尺寸和标准化）与预训练模型的要求相匹配。
# 5. 根据你的数据集大小和与ImageNet的相似度，你可以选择微调更多的层。
