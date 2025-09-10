
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# --- 前言：为什么需要正则化？ ---
# 在标准的线性回归中，模型的目标是最小化损失函数（如MSE）。
# 当特征数量很多，或者特征之间存在高度相关（多重共线性）时，模型可能会变得非常复杂，
# 权重系数会异常地大，导致对训练数据过拟合，在新数据上表现很差。
# 
# 正则化 (Regularization) 通过在损失函数中添加一个“惩罚项”来约束模型复杂度，
# 从而防止过拟合，提高模型的泛化能力。
# Loss = MSE + α * (Penalty Term)
# α (alpha) 是正则化强度参数，控制着惩罚的力度。

# --- 1. 准备数据 ---
# 我们创建一个有10个特征的数据集，其中一些特征是相关的。

X, y, coef = make_regression(
    n_samples=100, 
    n_features=10, 
    n_informative=5, # 只有5个特征是真正有用的
    noise=25, 
    coef=True, # 返回真实的系数
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Dataset Information ---")
print(f"Number of features: {X.shape[1]}")
print("True coefficients:")
print(np.round(coef, 2))
print("-"*30)

# --- 2. 训练不同模型 ---

# 创建一个字典来存储模型
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0), # alpha是正则化强度
    "Lasso (L1)": Lasso(alpha=1.0),
    "ElasticNet (L1+L2)": ElasticNet(alpha=1.0, l1_ratio=0.5) # l1_ratio控制L1和L2的混合比例
}

# 训练所有模型
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained.")

print("-"*30)

# --- 3. 比较模型系数 ---
# 正则化的效果最直观地体现在模型的系数上。

coef_df = {}
for name, model in models.items():
    coef_df[name] = model.coef_

# 打印系数进行比较
print("--- Comparing Model Coefficients ---")
print(f"Feature\t\tTrue Coef\tLinear\t\tRidge\t\tLasso\t\tElasticNet")
for i in range(X.shape[1]):
    print(f"Feature {i:2d}\t\t{coef[i]:8.2f}\t{models['Linear Regression'].coef_[i]:8.2f}\t{models['Ridge (L2)'].coef_[i]:8.2f}\t{models['Lasso (L1)'].coef_[i]:8.2f}\t{models['ElasticNet (L1+L2)'].coef_[i]:8.2f}")

print("\n观察:")
print("1. 线性回归的系数可能非常大，试图完美拟合数据。")
print("2. Ridge回归的系数被整体缩小了，但没有一个变为0。")
print("3. Lasso回归将许多不重要特征的系数直接压缩到了0！这实现了自动特征选择。")
print("4. ElasticNet是Ridge和Lasso的折中。")
print("-"*30)

# --- 4. 可视化系数 ---

plt.figure(figsize=(14, 7))
plt.plot(coef, alpha=0.7, linestyle='--', color='black', label='True Coefficients')

for name, model in models.items():
    plt.plot(model.coef_, alpha=0.7, label=name)

plt.title("Comparison of Coefficients for Different Regression Models")
plt.xlabel("Coefficient Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.grid(True)
plt.show()

# --- 5. 比较模型性能 ---
print("--- Comparing Model Performance on Test Set ---")
for name, model in models.items():
    score = model.score(X_test, y_test) # .score() for regression returns R²
    print(f"{name:>20s}: R² Score = {score:.4f}")

print("\n在存在多重共线性或特征数量多的情况下，正则化模型通常比普通线性回归有更好的泛化能力（更高的R²分数）。")

# 总结:
# 1. **Ridge (L2正则化)**: 主要用于处理特征间的“多重共线性”问题。它通过缩小系数来降低模型复杂度，使模型更稳定。
# 2. **Lasso (L1正则化)**: 主要用于“特征选择”。它能够将不重要的特征系数降为零，从而得到一个更稀疏、更易于解释的模型。
# 3. **ElasticNet**: 结合了两者的优点。当有很多相关特征时，Lasso可能会随机选择一个而忽略其他，ElasticNet则倾向于将它们一起选中或排除。
# 4. **alpha**: 正则化强度的选择至关重要，通常需要通过交叉验证（如GridSearchCV）来寻找最佳值。
