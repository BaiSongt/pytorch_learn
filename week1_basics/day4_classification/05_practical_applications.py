
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- 前言 ---
# 本脚本探讨在实现分类任务时遇到的一些常见实际问题及其标准解决方案。
# 我们将主要关注“类别不平衡”问题，并简要介绍多标签分类等概念。

# --- 1. 类别不平衡问题 (Class Imbalance) ---
# 当数据集中各个类别的样本数量差异巨大时，就会发生类别不平衡。
# 例如，在欺诈检测中，99.9%的交易是正常的，只有0.1%是欺诈交易。
# 标准模型在这种数据上训练时，会倾向于预测数量多的类别，因为它能轻易地获得高准确率，
# 但这会导致模型对于少数类的识别能力非常差，而少数类往往是我们更关心的。

# 创建一个不平衡的数据集
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                           weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("---" + " Class Imbalance Demonstration" + "---")
print(f"Training set: {len(y_train)} samples, {np.sum(y_train)} positive (class 1) samples.")

# 训练一个标准逻辑回归模型
model_standard = LogisticRegression()
model_standard.fit(X_train, y_train)
y_pred_standard = model_standard.predict(X_test)

print("\n---" + " Performance of Standard Model on Imbalanced Data" + "---")
print(classification_report(y_test, y_pred_standard))
print("观察：模型对多数类(0)的召回率很高，但对我们关心的少数类(1)的召回率(recall)可能非常低！")

# --- 2. 解决方案1：调整类别权重 (Class Weighting) ---
# 这是处理不平衡问题最常用、最简单的方法之一。
# 我们在训练模型时，告诉模型“少数类更重要”，给少数类的样本分配更高的权重。
# 当计算损失函数时，少数类的样本产生的误差会被放大，迫使模型更加关注它们。

# 在scikit-learn中，只需设置 `class_weight='balanced'`
# 模型会自动根据样本比例计算权重：weight(c) = n_samples / (n_classes * n_samples(c))
model_weighted = LogisticRegression(class_weight='balanced')
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("\n---" + " Performance of Weighted Model" + "---")
print(classification_report(y_test, y_pred_weighted))
print("观察：少数类(1)的召回率(recall)显著提升了！代价是精确率(precision)和整体准确率可能有所下降，这是一个权衡。\n")

# 在PyTorch中，这通常通过在损失函数中传递 `weight` 参数来实现。
# weights = torch.tensor([0.1, 0.9]) # 给少数类更高的权重
# criterion = nn.CrossEntropyLoss(weight=weights)

# --- 3. 解决方案2：重采样 (Resampling) ---
# - **过采样 (Oversampling)**: 复制少数类的样本，使得数据变得平衡。最著名的算法是 SMOTE (Synthetic Minority Over-sampling Technique)，它会创建新的、合成的少数类样本。
# - **欠采样 (Undersampling)**: 随机删除多数类的样本，直到数据平衡。缺点是可能会丢失重要信息。
# 这些技术通常使用 `imbalanced-learn` 这个库来实现。

print("---" * 30)

# --- 4. 多分类 vs. 多标签 ---

# **多分类 (Multi-Class Classification)**
# - 每个样本只属于**一个**类别，但类别总数大于2。
# - 例子：手写数字识别（一个图片是0-9中的一个数字），新闻主题分类（一篇文章属于体育、财经、科技中的一类）。
# - PyTorch损失函数：`nn.CrossEntropyLoss`。它内部包含了Softmax操作，输出N个类别的概率分布。

# **多标签 (Multi-Label Classification)**
# - 每个样本可以同时属于**多个**类别。
# - 例子：电影分类（一部电影可以同时是“动作”、“科幻”、“爱情”），图片内容识别（一张图里同时有“猫”、“狗”、“沙发”）。
# - PyTorch损失函数：`nn.BCEWithLogitsLoss` 或 `nn.BCELoss`。
#   - 模型最后一层有N个输出单元（N是总类别数），每个单元都经过一个Sigmoid激活函数，独立地表示“属于该类的概率”。
#   - 损失函数会计算每个类别的二元交叉熵，然后求和或求平均。

# 多标签场景的标签示例 (3个类别)
y_multilabel = torch.tensor([
    [1, 0, 1], # 样本0属于类别0和2
    [0, 1, 1], # 样本1属于类别1和2
    [1, 1, 0]  # 样本2属于类别0和1
], dtype=torch.float32)

# --- 5. 文本与图像分类简介 ---
# - **图像分类**: 在后续的CNN（卷积神经网络）周会深入学习。核心流程是：
#   1. 使用 `torchvision.transforms` 进行数据增强和预处理（如 `05_data_augmentation.py` 所示）。
#   2. 将处理后的图像张量送入一个CNN模型（如ResNet, VGG）进行特征提取和分类。
#
# - **文本分类**: 在后续的RNN和Transformer周会深入学习。核心流程是：
#   1. **文本向量化**: 将文本（字符串）转换为数值向量。方法包括：
#      - Bag-of-Words (词袋模型)
#      - TF-IDF
#      - Word Embeddings (如Word2Vec, GloVe, or learned with `nn.Embedding`)
#   2. 将向量化的文本送入模型（如RNN, LSTM, BERT）进行分类。
