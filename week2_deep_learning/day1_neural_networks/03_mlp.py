
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 前言 ---
# 多层感知机 (MLP) 是最基础的深度神经网络结构。
# 它由一个输入层、一个或多个隐藏层和一个输出层组成。
# 每个隐藏层都由线性变换和非线性激活函数构成。
# 本脚本将演示如何使用PyTorch的 `nn.Module` 来构建、训练和评估一个MLP。

# --- 1. 准备数据 ---
# 我们创建一个二分类问题的数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

print("--- 1. Data Preparation ---")
print(f"Training data shape: {X_train_tensor.shape}")
print("-"*30)

# --- 2. 定义MLP模型 ---
# 在PyTorch中，我们通过继承 `nn.Module` 来创建自定义模型。
# `nn.Sequential` 是一个非常有用的容器，可以按顺序将一系列层组合起来。

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()

        # `super().__init__()` 必须首先被调用

        # 定义网络结构
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1), # 输入层 -> 隐藏层1
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.3),                   # Dropout层，防止过拟合

            nn.Linear(hidden_size1, hidden_size2), # 隐藏层1 -> 隐藏层2
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.3),                   # Dropout层

            nn.Linear(hidden_size2, output_size) # 隐藏层2 -> 输出层
        )

    def forward(self, x):
        # forward方法定义了数据如何通过网络
        # 由于我们使用了nn.Sequential，前向传播变得非常简单
        return self.layers(x)

# 实例化模型
input_size = X_train.shape[1] # 20
hidden_size1 = 64
hidden_size2 = 32
output_size = 1 # 二分类问题，输出一个logit值

model = MLP(input_size, hidden_size1, hidden_size2, output_size)
print("--- 2. Model Architecture ---")
print(model)
print("-"*30)

# --- 3. 定义损失函数和优化器 ---

# 损失函数: 对于二分类问题，使用BCEWithLogitsLoss
# 它内部包含了Sigmoid激活，比手动加Sigmoid更数值稳定
criterion = nn.BCEWithLogitsLoss()

# 优化器: Adam是目前最常用、最稳健的优化器之一
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. 训练模型 ---
num_epochs = 50
batch_size = 32

print("--- 3. Training the MLP ---")
for epoch in range(num_epochs):
    model.train() # 将模型设置为训练模式 (启用Dropout等)

    # 简单的批处理循环
    for i in range(0, len(X_train_tensor), batch_size):
        # 获取一个批次的数据
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]

        # 1. 前向传播
        outputs = model(X_batch)

        # 2. 计算损失
        loss = criterion(outputs, y_batch)

        # 3. 反向传播和优化
        optimizer.zero_grad() # 清空梯度
        loss.backward()       # 计算梯度
        optimizer.step()        # 更新权重

    # 打印周期信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- Training Finished ---")
print("-"*30)

# --- 5. 评估模型 ---

model.eval() # 将模型设置为评估模式 (禁用Dropout等)

with torch.no_grad(): # 在评估时，我们不需要计算梯度
    # 对测试集进行预测
    y_predicted_logits = model(X_test_tensor)

    # 使用Sigmoid将logits转换为概率
    y_predicted_probs = torch.sigmoid(y_predicted_logits)

    # 以0.5为阈值，将概率转换为类别（0或1）
    y_predicted_cls = (y_predicted_probs > 0.5).float()

    # 计算准确率
    accuracy = (y_predicted_cls == y_test_tensor).sum() / float(y_test_tensor.shape[0])
    print(f"--- 4. Model Evaluation ---")
    print(f"Accuracy on test set: {accuracy.item():.4f}")

# 总结:
# 1. 继承 `nn.Module` 是构建自定义模型的标准方式。
# 2. `nn.Sequential` 可以极大地简化线性网络结构的定义。
# 3. 训练循环的核心是：前向传播 -> 计算损失 -> 清空梯度 -> 反向传播 -> 更新权重。
# 4. 训练和评估时，记得使用 `model.train()` 和 `model.eval()` 来切换模式，这会影响Dropout和BatchNorm等层的行为。
# 5. Adam优化器和BCEWithLogitsLoss/CrossEntropyLoss是分类任务中非常稳健和常用的选择。
