import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 准备非线性数据 ---
# 多项式回归用于拟合非线性关系。我们创建一个二次函数关系的数据 y = ax^2 + bx + c。
# y = 0.5x^2 - 3x + 2

# 创建x坐标
X_numpy = np.linspace(-5, 5, 100)
# 计算y，并添加噪声
y_numpy = 0.5 * X_numpy**2 - 3 * X_numpy + 2 + np.random.randn(100) * 2.0

# 可视化原始数据
plt.scatter(X_numpy, y_numpy, label='Original Noisy Data', alpha=0.6)
plt.title('Original Non-linear Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --- 2. 特征工程：创建多项式特征 ---
# 核心思想：虽然数据本身是非线性的，但我们可以通过转换特征，使其可以用线性模型来拟合。
# 对于一个二次关系 y = w2*x^2 + w1*x + b，我们可以创建新的特征 x' = [x, x^2]。
# 这样，模型就变成了 y = w*x' + b，这又是一个线性关系了！

# 我们将使用2次多项式，所以特征是 [x, x^2]
degree = 2
x_poly_features = np.array([[x**i for i in range(1, degree + 1)] for x in X_numpy])

# 将numpy数组转换为PyTorch张量
X_tensor = torch.from_numpy(x_poly_features.astype(np.float32))
y_tensor = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

print(f"Original X shape: {X_numpy.shape}")
print(f"Polynomial features shape: {X_tensor.shape}") # (100, 2) -> 每个样本有两个特征 x 和 x^2
print(f"Target y shape: {y_tensor.shape}")

# --- 3. 定义模型、损失和优化器 ---
# 即使我们做的是多项式回归，模型本身仍然是一个线性模型！
# 因为我们已经将非线性关系编码到了输入特征中。

class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(PolynomialRegressionModel, self).__init__()
        # 输入维度现在是多项式的次数 (degree)
        self.linear = nn.Linear(input_dim, 1) # 输出仍然是1个值

    def forward(self, x):
        return self.linear(x)

# 实例化模型
# 输入维度等于多项式的次数
input_dim = degree
model = PolynomialRegressionModel(input_dim)

print("\n--- Model Definition ---")
print(model)

# 损失函数和优化器与线性回归完全相同
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 使用一个稍小的学习率

# --- 4. 训练模型 ---
num_epochs = 200

print("\n--- Starting Training ---")
for epoch in range(num_epochs):
    # 前向传播
    y_predicted = model(X_tensor)
    
    # 计算损失
    loss = criterion(y_predicted, y_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("--- Training Finished ---")

# --- 5. 评估和可视化 ---
model.eval()
with torch.no_grad():
    # 使用训练好的模型进行预测
    predicted = model(X_tensor).detach().numpy()

# 绘制结果
plt.scatter(X_numpy, y_numpy, label='Original Data', alpha=0.4)
plt.plot(X_numpy, predicted, 'r', label='Fitted Polynomial Curve')
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. 讨论：过拟合与欠拟合 ---
# - **欠拟合 (Underfitting)**: 如果我们使用次数为1的多项式（即线性回归）来拟合这些数据，
#   模型过于简单，无法捕捉数据的非线性趋势，会导致很大的误差。这就是欠拟合。
#
# - **过拟合 (Overfitting)**: 如果我们使用一个非常高次的-多项式（比如次数为20），
#   模型会变得异常复杂，它会试图穿过每一个数据点，包括噪声点。
#   虽然在训练集上误差会很小，但在新的、未见过的数据上表现会很差。这就是过拟合。
#   选择合适的多项式次数（模型复杂度）是至关重要的，这通常需要通过交叉验证等技术来确定。

print("\n--- Discussion ---")
print("Polynomial regression allows fitting non-linear data using a linear model by transforming input features.")
print("The degree of the polynomial is a hyperparameter. Too low -> Underfitting. Too high -> Overfitting.")
