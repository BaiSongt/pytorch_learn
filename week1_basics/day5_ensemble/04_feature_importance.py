
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# --- 前言 ---
# 理解模型为什么会做出特定的预测（模型可解释性）是机器学习中一个至关重要的方面。
# 特征重要性可以告诉我们哪些特征对模型的预测贡献最大，帮助我们：
# 1. 理解数据和模型之间的关系。
# 2. 进行特征选择，简化模型，降低过拟合风险。
# 3. 向非技术人员解释模型的行为。

# --- 1. 准备数据和训练模型 ---
X, y = make_classification(
    n_samples=1000, 
    n_features=15, 
    n_informative=5, # 5个有用特征
    n_redundant=5,   # 5个是其他特征的线性组合
    n_repeated=0,    # 0个重复特征
    n_classes=2, 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 创建特征名称
feature_names = [f'feature_{i}' for i in range(X.shape[1])]

# 训练一个随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("---" + " Random Forest Model Trained " + "---")
print("---"*10)

# --- 2. 方法1：基于杂质的特征重要性 (Impurity-based Importance) ---
# 这是树模型（如决策树、随机森林、GBDT）自带的一种快速计算方法。
# - **原理**: 一个特征的重要性是根据它在分裂节点时，为模型带来了多大的“不纯度减少量”（如基尼不纯度或信息熵的减少）来计算的。
# - **优点**: 计算速度快，无需额外数据。
# - **缺点**: 
#   - 倾向于高估数值特征或具有许多类别的分类特征的重要性。
#   - 它是在训练数据上计算的，可能无法完全反映特征在未知数据上的真实重要性。

print("---" + " Method 1: Impurity-based Feature Importance " + "---")
importances_impurity = model.feature_importances_

# 将结果可视化
def plot_feature_importance(importances, names, title):
    importance_df = pd.DataFrame({'Feature': names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

plot_feature_importance(importances_impurity, feature_names, 'Impurity-based Feature Importance')
print("---"*10)

# --- 3. 方法2：置换重要性 (Permutation Importance) ---
# 这是一种更可靠、更通用的方法，适用于任何已训练好的模型。
# - **原理**: 
#   1. 首先，在原始的、未打乱的测试集上计算模型的基准性能（如准确率）。
#   2. 然后，随机打乱测试集中**某一列特征**的顺序，保持其他列和标签不变。
#   3. 在这个被打乱的数据上再次评估模型性能。
#   4. 该特征的重要性 = (基准性能 - 打乱后的性能)。如果性能下降很多，说明该特征很重要。
#   5. 对每一列特征重复此过程。
# - **优点**: 
#   - 模型无关，可用于任何模型。
#   - 在测试集上计算，能更好地反映特征在泛化中的重要性。
#   - 能更好地处理特征间的相关性问题。
# - **缺点**: 计算成本更高。

print("---" + " Method 2: Permutation Importance " + "---")
# n_repeats: 重复置换的次数，以获得更稳定的结果
perm_importance = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# `perm_importance.importances_mean` 存储了平均重要性
importances_perm = perm_importance.importances_mean

plot_feature_importance(importances_perm, feature_names, 'Permutation Importance')

print("\n观察：")
print("置换重要性的结果可能与基于杂质的重要性结果有所不同。")
print("通常认为置换重要性是更可靠的指标，因为它直接衡量了特征对模型性能的贡献。")

# 总结:
# 1. 特征重要性是模型解释性的关键工具。
# 2. **基于杂质的重要性** (如 `model.feature_importances_`) 速度快，但可能存在偏差，尤其是在有高基数特征或相关特征时。
# 3. **置换重要性** (`permutation_importance`) 是一种更可靠、模型无关的方法，它通过衡量特征被随机打乱后模型性能的下降程度来评估其重要性。
# 4. 在重要的项目中，推荐使用置换重要性来验证和补充基于杂质的重要性结果。
