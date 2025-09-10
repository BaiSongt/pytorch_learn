
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

# --- 前言 ---
# 特征工程是将原始数据转换为能更好地代表问题潜在模式的特征的过程，从而提高模型性能。
# “数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。”
# scikit-learn 的 `preprocessing` 模块是进行特征工程的强大工具。

# --- 1. 创建一个示例数据集 ---
# 我们创建一个包含不同尺度、缺失值和分类特征的数据集。
# 使用 pandas DataFrame 可以让这一切更清晰。

data = {
    'age': [25, 30, 35, 40, np.nan, 45], # 年龄，有缺失值
    'income': [50000, 80000, 120000, 70000, 65000, 200000], # 收入，尺度很大
    'experience': [2, 5, 10, 8, 4, 15] # 经验，尺度较小
}
df = pd.DataFrame(data)

print("--- Original DataFrame ---")
print(df)
print("---" * 30)

# --- 2. 处理缺失值 (Handling Missing Values) ---
# 模型无法处理包含 `NaN` 的数据，我们必须先填充或删除它们。
# `SimpleImputer` 是一个方便的工具。

# - `strategy`: 填充策略。
#   - 'mean': 使用该列的平均值填充。
#   - 'median': 使用中位数填充（对异常值更鲁棒）。
#   - 'most_frequent': 使用最频繁出现的值填充（适用于分类特征）。
#   - 'constant': 使用一个指定的常量填充。

imputer = SimpleImputer(strategy='mean')

# .fit_transform() 会计算平均值并转换数据
df['age'] = imputer.fit_transform(df[['age']])

print("--- DataFrame after Imputing Missing Values ---")
print(df)
print(f"The value {df.loc[4, 'age']:.2f} was imputed based on the mean of the 'age' column.")
print("---" * 30)

# --- 3. 特征缩放 (Feature Scaling) ---
# 当不同特征的数值范围（尺度）差异很大时（如age和income），
# 很多算法（如梯度下降、SVM、KNN）的性能会受到影响。
# 特征缩放将所有特征调整到相似的尺度。

# 提取数值特征用于缩放
numerical_features = df[['age', 'income', 'experience']]

# 方法1: 标准化 (Standardization)
# 公式: z = (x - μ) / σ  (μ是均值, σ是标准差)
# 结果: 数据将服从均值为0，标准差为1的标准正态分布。
# 这是最常用、最通用的缩放方法。
scaler_std = StandardScaler()
scaled_std_features = scaler_std.fit_transform(numerical_features)

print("--- Standardization (StandardScaler) ---")
print(pd.DataFrame(scaled_std_features, columns=numerical_features.columns))
print(f"Mean of scaled 'income': {scaled_std_features[:, 1].mean():.2f}")
print(f"Std dev of scaled 'income': {scaled_std_features[:, 1].std():.2f}")
print("\n")

# 方法2: 归一化 (Normalization)
# 公式: x_norm = (x - x_min) / (x_max - x_min)
# 结果: 数据被缩放到 [0, 1] 的范围内。
# 适用于数据分布有明确边界的情况，或某些神经网络（如图像处理）的输入。
scaler_minmax = MinMaxScaler()
scaled_minmax_features = scaler_minmax.fit_transform(numerical_features)

print("--- Normalization (MinMaxScaler) ---")
print(pd.DataFrame(scaled_minmax_features, columns=numerical_features.columns))
print(f"Min of scaled 'income': {scaled_minmax_features[:, 1].min():.2f}")
print(f"Max of scaled 'income': {scaled_minmax_features[:, 1].max():.2f}")
print("---" * 30)

# --- 4. 特征创建 (Feature Creation) ---
# 从现有特征中创建新的、可能更有用的特征。

# 示例1: 特征交叉 (Feature Crossing)
# 创建一个表示“收入与经验之比”的新特征
df['income_per_experience'] = df['income'] / (df['experience'] + 1) # +1避免除以0

print("--- Feature Creation: Feature Crossing ---")
print(df[['income', 'experience', 'income_per_experience']])
print("\n")

# 示例2: 多项式特征 (Polynomial Features)
# 在回归问题中，这可以帮助模型捕捉非线性关系。
poly = PolynomialFeatures(degree=2, include_bias=False)

# 我们只对 'age' 和 'experience' 创建多项式特征
poly_features = poly.fit_transform(df[['age', 'experience']])

print("--- Feature Creation: Polynomial Features (degree=2) ---")
# 新的特征包括: age, experience, age^2, age*experience, experience^2
print(f"Original feature names: ['age', 'experience']")
print(f"Polynomial feature names: {poly.get_feature_names_out()}")
print(pd.DataFrame(poly_features, columns=poly.get_feature_names_out()))

# 总结:
# 1. 特征工程是一个创造性的过程，需要对数据和业务有深入的理解。
# 2. 处理缺失值是数据预处理的第一步，`SimpleImputer` 是一个好帮手。
# 3. 特征缩放（特别是标准化 `StandardScaler`）对于大多数基于梯度的算法和距离计算的算法都至关重要。
# 4. 通过特征交叉或多项式扩展来创建新特征，可以帮助模型发现更复杂的关系。
# 5. 在一个完整的机器学习流程中，这些操作通常会被组织在一个 `sklearn.pipeline.Pipeline` 中，以确保在训练集和测试集上应用相同的变换。

