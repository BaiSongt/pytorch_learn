
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# --- 前言 --- 
# 正则化是任何旨在降低模型泛化误差（而非训练误差）的技术的总称。
# 换句话说，正则化就是用来防止过拟合的。

# --- 1. 准备一个容易过拟合的场景 ---
# 我们创建一个小型数据集，但使用一个相对复杂的模型来拟合它。
# 这使得模型有足够的能力去“记住”训练数据，从而导致过拟合。

# 生成数据 y = sin(x)
np.random.seed(42)
X = np.linspace(-np.pi, np.pi, 400).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.15, X.shape)

# 我们只使用一小部分数据进行训练
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.9, random_state=42)

# 转换为Tensor
X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.float32))
X_val_t = torch.from_numpy(X_val.astype(np.float32))
y_val_t = torch.from_numpy(y_val.astype(np.float32))

# 定义一个足够复杂的MLP模型
def create_model(dropout_rate=0.0):
    return nn.Sequential(
        nn.Linear(1, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate), # Dropout层
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate), # Dropout层
        nn.Linear(128, 1)
    )

# 辅助函数：用于训练和记录损失
def train_and_evaluate(model, optimizer, n_epochs=2000):
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train() # 训练模式
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = nn.MSELoss()(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval() # 评估模式
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = nn.MSELoss()(val_outputs, y_val_t)
            val_losses.append(val_loss.item())
    return train_losses, val_losses

# --- 2. 无正则化 (基线模型) ---
model_no_reg = create_model()
optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001)
_, val_losses_no_reg = train_and_evaluate(model_no_reg, optimizer_no_reg)

# --- 3. 技术1: L2正则化 (权重衰减) ---
# - **思想**: 在损失函数中加入一个惩罚项，等于所有模型参数（权重）平方和的倍数。
#   Loss = MSE + λ * Σ(w²)，其中 λ 是正则化强度。
# - **效果**: 惩罚大的权重值，迫使模型学习更小、更分散的权重，从而使模型更简单、更平滑。
# - **实现**: 在PyTorch的优化器中，这通过 `weight_decay` 参数实现 (weight_decay 等价于 λ)。
model_l2 = create_model()
optimizer_l2 = optim.Adam(model_l2.parameters(), lr=0.001, weight_decay=1e-4) # 加入权重衰减
_, val_losses_l2 = train_and_evaluate(model_l2, optimizer_l2)

# --- 4. 技术2: Dropout ---
# - **思想**: 在训练过程的每次前向传播中，以一定的概率p随机地将一部分神经元的输出设置为0。
# - **效果**: 
#   1. 强迫网络学习冗余的表示，因为任何一个神经元都可能随时“消失”。
#   2. 类似于同时训练大量共享权重的、不同的“稀疏”网络，然后在测试时取其平均。
# - **实现**: 在网络中加入 `nn.Dropout(p)` 层。`model.train()` 和 `model.eval()` 的切换至关重要，
#   因为Dropout只在训练时生效，在评估和预测时必须被关闭。
model_dropout = create_model(dropout_rate=0.5) # 使用50%的Dropout概率
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)
_, val_losses_dropout = train_and_evaluate(model_dropout, optimizer_dropout)

# --- 5. 技术3: 早停 (Early Stopping) ---
# - **思想**: 在训练过程中，持续监控模型在验证集上的性能。当验证集性能在一定周期（patience）内不再提升时，就提前终止训练。
# - **效果**: 一种简单但非常有效的“免费”正则化方法，可以防止模型在训练集上走得太远而开始过拟合。
# - **实现**: 这通常需要手动在训练循环中实现逻辑。
#
#   ```python
#   patience = 10 # 如果验证损失在10个epoch内没有改善，就停止
#   best_val_loss = float('inf')
#   epochs_no_improve = 0
#
#   for epoch in range(num_epochs):
#       # ... train one epoch ...
#       val_loss = ... # calculate validation loss
#
#       if val_loss < best_val_loss:
#           best_val_loss = val_loss
#           epochs_no_improve = 0
#           torch.save(model.state_dict(), 'best_model.pth') # 保存最佳模型
#       else:
#           epochs_no_improve += 1
#
#       if epochs_no_improve == patience:
#           print("Early stopping!")
#           break
#   ```

# --- 6. 结果对比 ---
plt.figure(figsize=(12, 8))
plt.plot(val_losses_no_reg, label='No Regularization', alpha=0.7)
plt.plot(val_losses_l2, label='L2 (Weight Decay)', alpha=0.7)
plt.plot(val_losses_dropout, label='Dropout', alpha=0.7)
plt.ylim(0, 0.1) # 限制y轴范围以便观察
plt.title('Effect of Regularization on Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

print("观察：正则化方法（L2和Dropout）通常可以获得更低的最终验证损失，表明模型泛化能力更好。")

# 总结:
# 1. **L2正则化 (Weight Decay)** 是通过优化器实现的，它是一种全局性的、平滑的正则化方法。
# 2. **Dropout** 是通过在模型中添加层来实现的，它是一种更“激进”的正则化方法。
# 3. **早停 (Early Stopping)** 是一种不修改模型或损失函数，仅通过监控验证性能来实现的正则化策略。
# 4. 在实践中，这些技术经常被组合使用，以达到最佳的抗过拟合效果。
