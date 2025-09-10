import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. 准备数据 ---
# 逻辑回归用于解决分类问题。我们使用 scikit-learn 的 `make_classification`
# 来生成一个二分类问题的虚拟数据集。

# n_samples: 样本数
# n_features: 特征数
# n_informative: 有效特征数
# n_redundant: 冗余特征数
# n_classes: 类别数
# random_state: 随机种子，确保结果可复现
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42,
)

# 数据标准化：对于逻辑回归和许多其他机器学习算法，标准化输入特征是一个好习惯。
# 它可以帮助优化算法更快地收敛。
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 将numpy数组转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

print(f"Training data shape: {X_train_tensor.shape}")
print(f"Training labels shape: {y_train_tensor.shape}")

# --- 2. 定义逻辑回归模型 ---
# 逻辑回归模型本质上也是一个线性模型，只是它的输出会通过一个激活函数（如Sigmoid）
# 来得到一个介于0和1之间的概率值。

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # 线性层的输出维度是1，代表二分类的一个原始分数（logit）
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 模型只返回线性的logit，激活函数将在损失函数中处理
        return self.linear(x)


# 实例化模型
input_dim = X_train.shape[1]  # 特征数量，这里是2
model = LogisticRegressionModel(input_dim)

print("\n--- Model Definition ---")
print(model)

# --- 3. 定义损失函数和优化器 ---

# 损失函数: 对于二分类问题，常用的是二元交叉熵 (Binary Cross Entropy, BCE)。
# `BCEWithLogitsLoss` 是一个更稳定和高效的选择，因为它将 Sigmoid 层和 BCE Loss 合并在一个类中。
# 它可以避免使用 `torch.sigmoid` 可能带来的数值不稳定性。
criterion = nn.BCEWithLogitsLoss()

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# --- 4. 训练模型 ---
num_epochs = 150

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    # 模型训练模式
    model.train()

    # 前向传播
    outputs = model(X_train_tensor)

    # 计算损失
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 15 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("--- Training Finished ---")

# --- 5. 评估模型 ---
# 训练完成后，我们在测试集上评估模型的性能。

# 模型评估模式
model.eval()

with torch.no_grad():  # 在评估时，我们不需要计算梯度
    # 对测试集进行预测
    y_predicted_logits = model(X_test_tensor)

    # 使用Sigmoid函数将logits转换为概率
    y_predicted_probs = torch.sigmoid(y_predicted_logits)

    # 以0.5为阈值，将概率转换为类别（0或1）
    y_predicted_cls = (y_predicted_probs > 0.5).float()

    # 计算准确率
    # (y_predicted_cls == y_test_tensor).sum() 计算预测正确的样本数
    accuracy = (y_predicted_cls == y_test_tensor).sum() / float(y_test_tensor.shape[0])
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy on test set: {accuracy.item():.4f}")

# --- 6. 可视化决策边界 ---
# 我们可以画出模型的决策边界，直观地看到模型是如何对平面上的点进行分类的。


def plot_decision_boundary(model, X, y):
    # 设置坐标轴范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 将网格点转换为张量，并进行预测
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    Z = model(grid_tensor)
    Z = torch.sigmoid(Z).detach().numpy().reshape(xx.shape)

    # 绘制等高线图和数据点
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors="k")
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# 绘制训练数据的决策边界
plot_decision_boundary(model, X_train, y_train)

# 总结:
# 1. 逻辑回归是解决分类问题的基础模型。
# 2. PyTorch中，它通常由一个 `nn.Linear` 层构成。
# 3. 损失函数推荐使用 `nn.BCEWithLogitsLoss`，它结合了Sigmoid和BCE，数值更稳定。
# 4. 训练和评估流程与回归模型类似，但评估指标变为准确率、精确率、召回率等。
# 5. 决策边界可视化是理解分类模型行为的好方法。
