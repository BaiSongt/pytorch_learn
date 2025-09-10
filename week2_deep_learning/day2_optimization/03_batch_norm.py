
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 前言 ---
# 批量归一化 (BatchNorm) 是一种用于加速深度网络训练、提升模型稳定性和性能的技术。
# 它通过在网络层之间对数据进行重新标准化来工作，解决了所谓的“内部协变量偏移”问题。
# (Internal Covariate Shift: 在训练过程中，由于前一层参数的变化，导致后一层输入的分布不断变化，增加了学习难度)。

# --- 1. 准备数据 ---
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
X_val_t = torch.from_numpy(X_val.astype(np.float32))
y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1)

# --- 2. 定义模型：带BatchNorm vs. 不带BatchNorm ---

def create_model(with_batch_norm=False):
    layers = []
    input_dim = 20
    hidden_dims = [128, 64, 32]
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        if with_batch_norm:
            # BatchNorm层通常放在线性层之后，激活函数之前
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        input_dim = hidden_dim
        
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)

model_bn = create_model(with_batch_norm=True)
model_no_bn = create_model(with_batch_norm=False)

print("--- Model with BatchNorm ---")
print(model_bn)
print("\n--- Model without BatchNorm ---")
print(model_no_bn)
print("---" * 30)

# --- 3. BatchNorm的工作原理 ---
# - **训练模式 (`model.train()`)**:
#   1. 计算当前**批次(batch)**数据的均值和方差。
#   2. 使用该均值和方差来标准化这一批数据。
#   3. 学习两个可训练的参数 `gamma` 和 `beta`，对标准化后的数据进行缩放和平移 (y = γx + β)，以保持网络的表达能力。
#   4. 同时，维护一个全局的“移动平均”均值和方差，供评估时使用。
#
# - **评估模式 (`model.eval()`)**:
#   1. **不**使用当前批次的均值和方差。
#   2. 使用在整个训练过程中学习到的全局移动平均均值和方差来标准化数据。
#   这确保了在评估时，对于单个样本的预测是确定性的。

# --- 4. 实验对比 ---

def train_and_get_losses(model, optimizer, n_epochs=30):
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        # 简单的批处理
        for i in range(0, len(X_train_t), 64):
            X_batch = X_train_t[i:i+64]
            y_batch = y_train_t[i:i+64]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = nn.BCEWithLogitsLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = nn.BCEWithLogitsLoss()(val_outputs, y_val_t)
            val_losses.append(val_loss.item())
    return val_losses

# 训练带BatchNorm的模型
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.01)
val_losses_bn = train_and_get_losses(model_bn, optimizer_bn)

# 训练不带BatchNorm的模型
optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.01)
val_losses_no_bn = train_and_get_losses(model_no_bn, optimizer_no_bn)

# 绘制验证损失曲线
plt.figure(figsize=(10, 7))
plt.plot(val_losses_bn, label='With BatchNorm')
plt.plot(val_losses_no_bn, label='Without BatchNorm')
plt.title('Effect of BatchNorm on Convergence')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

print("观察：带BatchNorm的模型通常收敛得更快，且能达到更低的验证损失。 সন")
print("---" * 30)

# --- 5. 其他归一化层 ---
# - **Layer Normalization (`nn.LayerNorm`)**:
#   - **作用**: 在**每个样本**的**所有特征**上计算均值和方差进行归一化。
#   - **应用**: 在批次大小很小，或者每个样本的特征长度可变时特别有用。是**Transformer和RNN**中的标准选择。
#
# - **Instance Normalization (`nn.InstanceNorm1d/2d/3d`)**:
#   - **作用**: 在**每个样本**的**每个通道**上独立地计算均值和方差。
#   - **应用**: 主要用于风格迁移等图像生成任务，它能保留单个图像的对比度信息。
#
# - **Group Normalization (`nn.GroupNorm`)**:
#   - **作用**: 将通道分成若干组，在每个组内进行归一化。是BatchNorm和LayerNorm的折中。
#   - **应用**: 当批次大小很小时，性能通常优于BatchNorm。

# 总结:
# 1. BatchNorm是训练现代深度神经网络（尤其是CNN）的基石之一。
# 2. 它通过在训练时对每个批次进行归一化来稳定和加速训练过程。
# 3. 务必记得使用 `model.train()` 和 `model.eval()` 来切换模式，因为BatchNorm在这两种模式下的行为完全不同。
# 4. 对于序列数据（RNN/Transformer），`LayerNorm` 是更好的选择。

