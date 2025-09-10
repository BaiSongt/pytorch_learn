
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # "Support Vector Classifier"

# --- 前言：关于在PyTorch学习中使用scikit-learn ---
# 支持向量机 (SVM) 是一种经典的、非常强大的机器学习算法，但它并不是深度学习模型。
# PyTorch的核心是深度学习，提供了构建神经网络的灵活工具。
# 而SVM的训练算法（如SMO）与基于梯度下降的神经网络训练方法有很大不同。
# 因此，在实践中，我们几乎总是使用像 scikit-learn 这样优化过的库来实现SVM。
# 本脚本旨在演示SVM的概念，并展示如何在项目中集成和使用scikit-learn。

# --- 1. 准备数据 ---
# 我们使用和逻辑回归示例中相同的数据生成方式，以便于比较。
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# --- 2. 定义和训练SVM模型 ---
# 使用 scikit-learn 的 `SVC` (Support Vector Classifier)

# - `kernel`: 核函数。这是SVM最强大的特性之一。
#   - 'linear': 线性核，用于线性可分数据。
#   - 'rbf' (Radial Basis Function): 径向基函数核，也叫高斯核。默认选项，适用于非线性问题。
#   - 'poly': 多项式核。
# - `C`: 正则化参数。C值越小，正则化越强，允许更多的误分类点，决策边界更平滑；C值越大，正则化越弱，试图将所有点正确分类，可能导致过拟合。
# - `gamma`: 'rbf'、'poly' 和 'sigmoid' 核的系数。可以理解为单个训练样本对模型的影响范围。值越大，影响范围越小，决策边界可能越复杂。

print("--- Training SVM Model ---")
# 我们创建一个使用RBF核的SVM模型
svm_model = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True) # probability=True 允许我们后续获取概率值

# 训练模型，scikit-learn的API非常简洁
# .fit() 方法会执行所有复杂的训练算法
svm_model.fit(X_train, y_train)

print("--- Training Finished ---")

# --- 3. 评估模型 ---
# 使用训练好的模型在测试集上进行评估

# .predict() 直接返回预测的类别
y_predicted_cls = svm_model.predict(X_test)

# .score() 方法可以直接计算并返回准确率
accuracy = svm_model.score(X_test, y_test)

print(f"--- Model Evaluation ---")
print(f"Accuracy on test set: {accuracy:.4f}")

# --- 4. 可视化决策边界 ---
# 和逻辑回归一样，我们可以可视化SVM的决策边界。
# SVM的目标是找到一个“最大间隔”超平面，使得不同类别之间的距离最大化。

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # 使用 .predict() 来获取网格上每个点的分类
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    
    # 找到并绘制支持向量
    # 支持向量是那些离决策边界最近的数据点，它们是定义决策边界的关键
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    
    plt.title("SVM Decision Boundary with RBF Kernel")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# 绘制训练数据的决策边界
plot_decision_boundary(svm_model, X_train, y_train)

# 总结:
# 1. SVM是一种强大的分类算法，尤其擅长处理非线性问题（通过核技巧）。
# 2. 在Python中，实现SVM的最佳工具是 scikit-learn。
# 3. `SVC` 的关键超参数是 `kernel`, `C`, 和 `gamma`，需要仔细调优以获得最佳性能。
# 4. SVM的核心思想是找到最大间隔超平面，由“支持向量”决定。
# 5. 在一个项目中，混合使用PyTorch（用于深度学习）和scikit-learn（用于传统机器学习）是非常常见的做法.
