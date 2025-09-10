
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# --- 前言 ---
# 集成学习 (Ensemble Learning) 是一种通过结合多个学习器（模型）的预测来获得比单个学习器更好性能的策略。
# “三个臭皮匠，顶个诸葛亮”。关键在于学习器之间应该具有“差异性”。

# --- 1. 准备数据 ---
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. 主流集成思想 ---

# **A. Bagging (Bootstrap Aggregating)**
# - **思想**: 并行训练多个同种类的模型，每个模型在不同的数据子集（通过自助采样得到）上训练。
# - **代表**: 随机森林 (Random Forest) 就是Bagging思想的集大成者。
# - **效果**: 主要用于降低模型的“方差”，防止过拟合，提高模型的稳定性。
print("--- Method 1: Bagging ---")
# 使用BaggingClassifier，我们可以将任何基模型包装起来
# 这里我们用决策树作为基模型
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(), 
    n_estimators=100, # 创建100个决策树
    random_state=42, 
    n_jobs=-1
)
bagging_model.fit(X_train, y_train)
print(f"Bagging (100 Decision Trees) Accuracy: {bagging_model.score(X_test, y_test):.4f}")

# **B. Boosting (提升)**
# - **思想**: 串行训练多个同种类的模型（通常是弱学习器），后一个模型主要关注前一个模型预测错误的样本。
# - **代表**: AdaBoost, Gradient Boosting (GBDT), XGBoost, LightGBM。
# - **效果**: 主要用于降低模型的“偏差”，提高模型的准确率。
print("\n--- Method 2: Boosting ---")
# AdaBoost (Adaptive Boosting) 示例
adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), # 通常使用弱学习器，如决策树桩
    n_estimators=100, 
    random_state=42
)
adaboost_model.fit(X_train, y_train)
print(f"AdaBoost (100 Decision Stumps) Accuracy: {adaboost_model.score(X_test, y_test):.4f}")

# **C. Stacking (堆叠)**
# - **思想**: 分层训练模型。第一层的多个模型（可以是不同种类）在原始数据上训练，
#   然后将它们的输出作为第二层模型（称为“元学习器”）的输入特征，由元学习器做出最终预测。
# - **效果**: 结合了多个不同模型的优点，可能达到非常高的性能，但实现和训练过程最复杂。

# --- 3. 投票 (Voting) ---
# 投票是一种最简单的集成方式，它将多个**不同种类**的模型组合起来。

print("\n--- Method 3: Voting ---")
# 1. 定义几个不同的基模型
clf1 = LogisticRegression(random_state=42)
clf2 = SVC(probability=True, random_state=42) # probability=True for soft voting
clf3 = DecisionTreeClassifier(random_state=42)

# 2. 创建投票分类器
# - `estimators`: 一个包含(名称, 模型)元组的列表。
# - `voting`:
#   - 'hard': 硬投票。少数服从多数，直接根据预测类别投票。
#   - 'soft': 软投票。对所有模型的预测概率进行加权平均，然后选择概率最高的类别。通常性能优于硬投票。
voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)],
    voting='soft'
)

# 3. 训练和评估
# 先单独评估每个模型
for clf, label in zip([clf1, clf2, clf3], ['Logistic Regression', 'SVC', 'Decision Tree']):
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy of {label}: {acc:.4f}")

# 评估集成的投票分类器
voting_clf.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting_clf.predict(X_test))
print(f"Accuracy of Voting Classifier: {voting_acc:.4f}")
print("\n观察：投票分类器的性能通常会比其中最差的模型要好，甚至可能超过所有单个模型。")

# 总结:
# - **Bagging (并行)**: 通过平均来降低方差。随机森林是其最著名的应用。
# - **Boosting (串行)**: 通过关注错误来降低偏差。XGBoost等是其现代高性能实现。
# - **Stacking (分层)**: 用一个模型来学习如何组合其他模型的输出。
# - **Voting (投票)**: 最简单的集成方式，直接结合多个不同模型的预测结果。
# 
# 在实践中，Boosting方法（如XGBoost, LightGBM）通常在结构化数据（如表格数据）的竞赛和应用中表现最佳。
# Stacking和Voting是强大的技术，但需要仔细的模型选择和调优。
