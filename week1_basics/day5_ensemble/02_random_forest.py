
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # 导入用于比较

# --- 前言：什么是随机森林？ ---
# 随机森林 (Random Forest) 是一种集成学习方法，它通过构建多个决策树并在预测时取其平均值（回归）或众数（分类）来工作。
# 它主要基于 "Bagging" (Bootstrap Aggregating) 技术。
# 
# 两个核心思想：
# 1. **Bootstrap (自助采样)**: 从原始训练集中有放回地随机抽取样本，创建多个不同的训练子集。每个决策树在自己的子集上训练。
# 2. **随机特征选择**: 在每个决策树的每个节点进行分裂时，不是从所有特征中选择最佳分裂点，而是从一个随机选择的特征子集中选择。
# 
# 这两个随机性来源确保了森林中的每棵树都是不同的，从而降低了整体模型的过拟合风险，提高了泛化能力。

# --- 1. 准备数据 ---
# 同样使用鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dataset: Iris")
print(f"Training data shape: {X_train.shape}")

# --- 2. 定义和训练随机森林模型 ---
# 使用 scikit-learn 的 `RandomForestClassifier`

# - `n_estimators`: 森林中决策树的数量。这是最重要的参数之一，通常值越大性能越好，但训练和预测时间也会增加。
# - `max_depth`, `min_samples_split`, `min_samples_leaf`: 这些参数与决策树中的相同，用于控制单棵树的生长。
# - `max_features`: 在寻找最佳分裂点时考虑的特征数量。'auto' 或 'sqrt' 是常用选项。
# - `n_jobs`: 并行运行的作业数。-1 表示使用所有可用的CPU核心，可以显著加快训练速度。

print("\n--- Training Random Forest Model ---")
# 创建一个包含100棵树的随机森林
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 训练模型
rf_model.fit(X_train, y_train)

print("--- Training Finished ---")

# --- 3. 评估模型并与单个决策树比较 ---

# 评估随机森林
rf_accuracy = rf_model.score(X_test, y_test)

# 为了比较，我们训练一个未剪枝的单个决策树
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = dt_model.score(X_test, y_test)

print(f"\n--- Model Evaluation ---")
print(f"Single Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy:        {rf_accuracy:.4f}")
print("\n通常，随机森林比单个决策树更准确、更不容易过拟合。" )

# --- 4. 特征重要性 ---
# 随机森林的特征重要性是森林中所有树的特征重要性的平均值，因此通常比单棵树的结果更可靠。
print("\n--- Feature Importances from Random Forest ---")
importances = rf_model.feature_importances_
# 创建一个条形图来可视化
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances in Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), np.array(iris.feature_names)[indices], rotation=45)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# --- 5. 查看森林中的一棵树 ---
# 随机森林是一个“黑盒”模型，因为它很难像单棵树那样被完全解释。
# 但我们可以提取并可视化其中的任意一棵树，来感受一下它的结构。

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
# rf_model.estimators_ 是一个列表，包含了森林里所有的决策树对象
plot_tree(rf_model.estimators_[5], # 可视化第6棵树
          filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names, 
          rounded=True, 
          fontsize=10)
plt.title("One Tree from the Random Forest", fontsize=16)
plt.show()
# 你会发现这棵树可能看起来很“深”且“杂乱”，因为它是在一个数据子集和特征子集上训练的，
# 单个树允许过拟合，但集成的力量会纠正这一点。

# 总结:
# 1. 随机森林通过 Bagging 和随机特征选择构建了多个不同的决策树。
# 2. 它是一种强大的集成模型，通常比单个决策树具有更高的准确性和更好的泛化能力。
# 3. `n_estimators` (树的数量) 是其关键参数。
# 4. 随机森林提供的特征重要性评估比单棵树更稳定可靠。
# 5. 尽管解释性不如单个决策树，但其性能优势使其成为最受欢迎的机器学习算法之一。
