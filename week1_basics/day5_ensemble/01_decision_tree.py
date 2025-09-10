
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- 前言 ---
# 决策树是一种直观的、类似流程图的监督学习模型。
# 它通过从数据特征中学习一系列“是/否”问题来进行决策。
# 随机森林是决策树的集成版本，我们将在下一个脚本中介绍。
# 和SVM一样，决策树在 scikit-learn 中有最高效的实现。

# --- 1. 准备数据 ---
# 我们使用经典的鸢尾花 (Iris) 数据集。
# 这是一个多分类问题（3个类别），包含4个特征。
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
# stratify=y 确保在训练集和测试集中，各个类别的比例与原始数据集中相同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dataset: Iris")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Training data shape: {X_train.shape}")

# --- 2. 定义和训练决策树模型 ---
# 使用 scikit-learn 的 `DecisionTreeClassifier`

# - `criterion`: 分裂质量的衡量标准。
#   - 'gini': 基尼不纯度（默认）。
#   - 'entropy': 信息增益（使用信息熵）。
# - `max_depth`: 树的最大深度。这是防止过拟合的重要参数。如果为None，则树会一直生长直到所有叶子都是纯的。
# - `min_samples_split`: 一个内部节点要进行分裂所需要的最小样本数。
# - `min_samples_leaf`: 一个叶子节点必须拥有的最小样本数。

print("\n--- Training Decision Tree Model ---")
# 我们创建一个最大深度为3的决策树，以防止过拟合，并方便可视化
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 训练模型
dt_model.fit(X_train, y_train)

print("--- Training Finished ---")

# --- 3. 评估模型 ---
accuracy = dt_model.score(X_test, y_test)
print(f"\n--- Model Evaluation ---")
print(f"Accuracy on test set: {accuracy:.4f}")

# --- 4. 特征重要性 ---
# 决策树训练完成后，我们可以查看每个特征对于决策的贡献度，即“特征重要性”。
# 重要性越高的特征，在树的分裂过程中起到的作用越大。
print("\n--- Feature Importances ---")
for feature, importance in zip(iris.feature_names, dt_model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# --- 5. 可视化决策树 ---
# 可视化是理解决策树如何工作的最佳方式。

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model, 
    filled=True, # 用颜色填充节点以表示纯度
    feature_names=iris.feature_names, # 显示特征名称
    class_names=iris.target_names, # 显示类别名称
    rounded=True, # 使用圆角矩形
    fontsize=12
)
plt.title("Decision Tree for Iris Dataset (max_depth=3)", fontsize=16)
plt.show()

# 如何解读上图：
# - 每个节点顶部的条件是分裂规则（例如 `petal length (cm) <= 2.45`）。
# - `gini`: 该节点的基尼不纯度。值越小，节点越“纯”。
# - `samples`: 落入该节点的样本数量。
# - `value`: 该节点中，每个类别的样本数量（例如 `[35, 35, 35]` 表示每个类别各有35个样本）。
# - `class`: 该节点的主要类别。

# --- 6. 剪枝技术 (Pruning) ---
# 剪枝是防止决策树过拟合的关键技术。我们上面使用的 `max_depth` 就是一种“预剪枝”。
# - **预剪枝 (Pre-pruning)**: 在树完全生长之前，通过设置一些条件（如`max_depth`, `min_samples_split`）来提前停止树的生长。
# - **后剪枝 (Post-pruning)**: 先让树完全生长，然后从下往上，移除那些对模型泛化能力贡献不大的子树。
# scikit-learn 主要通过预剪枝参数来控制模型复杂度。

# 总结:
# 1. 决策树是一种易于理解和解释的白盒模型。
# 2. scikit-learn 的 `DecisionTreeClassifier` 是标准实现。
# 3. 通过 `plot_tree` 可以轻松地可视化决策树的结构。
# 4. 特征重要性可以帮助我们理解哪些特征对预测最重要。
# 5. 必须通过剪枝（如限制`max_depth`）来控制树的复杂度，以防止过拟合。
