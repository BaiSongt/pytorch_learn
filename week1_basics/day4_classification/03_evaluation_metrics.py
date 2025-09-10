
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc

# --- 前言 ---
# 评估分类模型比回归模型更复杂。准确率（Accuracy）在类别不平衡时具有很强的误导性。
# （例如，99%的邮件是正常的，1%是垃圾邮件。一个将所有邮件都预测为“正常”的模型，准确率高达99%，但它毫无用处。）
# 因此，我们需要一整套指标来全面评估模型。

# --- 1. 准备数据和训练一个简单模型 ---
# 我们创建一个稍微不平衡的数据集来更好地说明指标的用途
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 使用scikit-learn的逻辑回归模型进行演示，更方便快捷
model = LogisticRegression()
model.fit(X_train, y_train)

# 在测试集上获取预测结果
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # 获取属于类别1的概率

print("---" + " Model Trained and Predictions Made " + "---")
print(f"Test set size: {len(y_test)}")
print(f"Number of positive class (1) in test set: {np.sum(y_test)}")
print("---" + "-" * 30)

# --- 2. 基础指标：准确率 (Accuracy) ---
accuracy = accuracy_score(y_test, y_pred)
print("---" + " Metric 1: Accuracy " + "---")
print(f"Accuracy: {accuracy:.4f}")
print("Accuracy = (Correct Predictions) / (Total Predictions). Can be misleading in imbalanced datasets.")
print("---" + "-" * 30)

# --- 3. 核心工具：混淆矩阵 (Confusion Matrix) ---
# 混淆矩阵清晰地展示了模型在每个类别上的预测情况。
# TN (True Negative): 真实为0，预测为0
# FP (False Positive): 真实为0，预测为1 (Type I Error)
# FN (False Negative): 真实为1，预测为0 (Type II Error)
# TP (True Positive): 真实为1，预测为1
cm = confusion_matrix(y_test, y_pred)

print("---" + " Tool: Confusion Matrix " + "---")
print(f"Confusion Matrix:\n{cm}")

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
print("---" + "-" * 30)

# --- 4. 精确率, 召回率, F1分数 ---
# 这些指标都基于混淆矩阵的四个值。

# **精确率 (Precision)** = TP / (TP + FP)
# - 含义：在所有被模型预测为“正类”的样本中，有多少是真正的正类。
# - 应用：当你希望预测的结果尽可能准确时（例如，股票预测，宁愿错过机会也不想预测错误）。

# **召回率 (Recall / Sensitivity)** = TP / (TP + FN)
# - 含义：在所有真正的正类样本中，有多少被模型成功地找出来了。
# - 应用：当你希望尽可能地找出所有正类时（例如，癌症诊断，宁愿误诊也不想漏掉一个真正的病人）。

# **F1分数 (F1-Score)** = 2 * (Precision * Recall) / (Precision + Recall)
# - 含义：精确率和召回率的调和平均值。它是一个综合性指标，当P和R都高时，F1值也高。

# `classification_report` 函数可以方便地打印出所有这些指标。
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("---" + " Metrics 2, 3, 4: Precision, Recall, F1-Score " + "---")
print(report)
print("---" + "-" * 30)

# --- 5. ROC曲线 和 AUC ---
# **ROC曲线 (Receiver Operating Characteristic Curve)**
# - 描述了在不同分类阈值下，模型的“真正类率 (TPR, 即Recall)”与“假正类率 (FPR)”之间的关系。
# - FPR = FP / (FP + TN)，即所有真实负类中，被错误预测为正类的比例。
# - 一个好的分类器应该在FPR很低的情况下，有很高的TPR。

# **AUC (Area Under the Curve)**
# - ROC曲线下的面积。AUC值是[0, 1]之间的一个数字。
# - AUC = 1: 完美的分类器。
# - AUC = 0.5: 随机猜测的分类器。
# - AUC < 0.5: 比随机猜测还差的分类器。
# - AUC是一个不依赖于特定阈值的、对模型整体性能的评估。

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("---" + " Metric 5: ROC Curve and AUC " + "---")
print(f"AUC Score: {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 总结:
# - 不要只看准确率！
# - 混淆矩阵是理解模型行为的基础。
# - 精确率和召回率是在不同业务场景下的重要权衡指标。
# - F1-Score是P和R的综合评价。
# - AUC是评估模型在所有阈值下整体性能的强大工具，尤其适用于比较不同模型的好坏。

