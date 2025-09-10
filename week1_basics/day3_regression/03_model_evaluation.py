
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 前言 ---
# 训练完一个模型后，我们需要一套客观的指标来评估它的好坏。
# 对于回归问题，评估的是模型预测值与真实值之间的接近程度。
# 本脚本将介绍几种最常用的回归模型评估指标。
# 我们通常在“测试集”（模型训练过程中未见过的数据）上进行评估，以判断模型的泛化能力。

# --- 1. 准备数据和训练一个简单模型 ---
# 我们首先需要一个训练好的模型和一些测试数据。
# 这里我们快速地复用线性回归的例子。

# 生成数据 y = 3x + 5 + noise
X_numpy = np.linspace(0, 20, 200)
y_numpy = 3 * X_numpy + 5 + np.random.randn(200) * 3

# 转换为Tensor
X_tensor = torch.from_numpy(X_numpy.astype(np.float32)).view(-1, 1)
y_tensor = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 定义并训练模型 (简化版)
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(200):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("--- Model Trained ---")

# --- 2. 在测试集上进行预测 ---
model.eval() # 设置为评估模式
with torch.no_grad(): # 不计算梯度
    y_predicted_tensor = model(X_test)

# 将Tensor转换为Numpy数组，以便使用scikit-learn的评估函数
y_test_numpy = y_test.numpy()
y_predicted_numpy = y_predicted_tensor.numpy()

print(f"Test set size: {len(y_test_numpy)}")
print("-"*30)

# --- 3. 计算评估指标 ---
# scikit-learn.metrics 提供了各种方便的评估函数。

# 指标1: 均方误差 (Mean Squared Error, MSE)
# 公式: (1/n) * Σ(y_true - y_pred)^2
# 含义: 预测误差平方的平均值。对异常值（大的误差）非常敏感。
# 值越小越好。
mse = mean_squared_error(y_test_numpy, y_predicted_numpy)
print(f"--- Metric 1: Mean Squared Error (MSE) ---")
print(f"MSE: {mse:.4f}")
print("MSE gives a sense of the average squared difference between prediction and reality.")
print("It penalizes large errors fatores.")

# 指标2: 均方根误差 (Root Mean Squared Error, RMSE)
# 公式: sqrt(MSE)
# 含义: MSE的平方根。它的量纲与目标变量y相同，因此更易于解释。
# 例如，如果房价的RMSE是5000，意味着模型的预测平均偏离真实房价5000元。
# 值越小越好。
rmse = np.sqrt(mse)
print(f"\n--- Metric 2: Root Mean Squared Error (RMSE) ---")
print(f"RMSE: {rmse:.4f}")
print("RMSE is in the same unit as the target variable, making it more interpretable.")

# 指标3: 平均绝对误差 (Mean Absolute Error, MAE)
# 公式: (1/n) * Σ|y_true - y_pred|
# 含义: 预测误差绝对值的平均值。与MSE相比，它对异常值的敏感度较低。
# 值越小越好。
mae = mean_absolute_error(y_test_numpy, y_predicted_numpy)
print(f"\n--- Metric 3: Mean Absolute Error (MAE) ---")
print(f"MAE: {mae:.4f}")
print("MAE gives the average magnitude of error, regardless of direction.")

# 指标4: R²分数 (R-squared, Coefficient of Determination)
# 公式: 1 - (Σ(y_true - y_pred)^2) / (Σ(y_true - y_mean)^2)
# 含义: 表示模型解释了目标变量方差的百分比。
# - R² = 1: 模型完美预测所有数据。
# - R² = 0: 模型性能等同于一个简单的“平均值模型”（即无论输入是什么，总是预测y的平均值）。
# - R² < 0: 模型性能比平均值模型还要差。
# 值越接近1越好。
r2 = r2_score(y_test_numpy, y_predicted_numpy)
print(f"\n--- Metric 4: R-squared (R²) ---")
print(f"R² Score: {r2:.4f}")
print("R² indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).")

# --- 总结 ---
# - **MSE/RMSE**: 常用且对大误差敏感，是优化中常用的损失函数。
# - **MAE**: 对异常值不敏感，能更好地反映一般情况下的误差大小。
# - **R²**: 提供了一个相对的性能度量，表示模型的好坏程度（相对于基准模型）。
# 在一个完整的项目中，通常会同时报告多个指标，以全面地评估模型性能。
