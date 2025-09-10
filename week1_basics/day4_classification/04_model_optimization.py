
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC

# --- 前言 ---
# 模型优化是寻找最佳模型（包括算法选择和超参数配置）的过程，以在未知数据上获得最佳性能。
# 超参数是在训练开始前设置的参数，例如学习率、正则化强度alpha、SVM的C和gamma等。
# 手动调参非常低效，我们需要自动化的方法。

# --- 1. 准备数据 ---
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("---" + "-" * 25 + "Dataset Information" + "-" * 25 + "---")
print(f"Training data shape: {X_train.shape}")
print("-" * 30)

# --- 2. 问题：简单的Train/Test Split的局限性 ---
# 我们之前的做法是将数据一次性地分为训练集和测试集。
# 这种方法的评估结果高度依赖于数据的分割方式。如果某次分割恰好分到“简单”的测试集，
# 模型的得分可能会虚高，反之亦然。评估结果不够稳定和可靠。

# --- 3. 解决方案：K-折交叉验证 (K-Fold Cross-Validation) ---
# 交叉验证将训练集分成K个相等的部分（称为“折”，fold）。
# 然后进行K次训练和验证：
# - 第1次：用第1折作验证集，其余K-1折作训练集。
# - 第2次：用第2折作验证集，其余K-1折作训练集。
# - ...
# - 第K次：用第K折作验证集，其余K-1折作训练集。
# 最后，将K次验证的得分取平均值，作为模型性能的最终评估。
# 这提供了一个更稳定、更可靠的性能度量。

print("---" + "-" * 25 + "K-Fold Cross-Validation" + "-" * 25 + "---")
# 我们用一个未调优的SVM模型来演示
# cv=5 表示进行5折交叉验证
model_cv = SVC(random_state=42)
scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring='accuracy')

print(f"Scores for each of the 5 folds: {np.round(scores, 4)}")
print(f"Average CV Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {scores.std():.4f}")
print("-" * 30)

# --- 4. 自动化超参数调优：网格搜索 (Grid Search) ---
# 网格搜索会穷举我们提供的一个超参数网格中的所有可能组合。
# 对于每一种组合，它都会使用交叉验证来评估其性能。
# 最后，它会告诉我们哪种组合得到了最高的平均交叉验证分数。

print("---" + "-" * 25 + "Grid Search with Cross-Validation (GridSearchCV)" + "-" * 25 + "---")
# 1. 定义要搜索的超参数网格
# 我们来调整SVM的两个关键超参数：C 和 gamma
param_grid = {
    'C': [0.1, 1, 10, 100], # 正则化参数
    'gamma': [1, 0.1, 0.01, 0.001], # RBF核的系数
    'kernel': ['rbf'] # 我们只搜索RBF核
}

# 2. 创建GridSearchCV对象
# - estimator: 要调优的模型
# - param_grid: 超参数网格
# - cv: 交叉验证的折数
# - scoring: 评估指标
# - n_jobs: 并行运行的作业数 (-1表示使用所有CPU核心)
# - verbose: verbose>0 会打印搜索过程
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1 # verbose>0 会打印搜索过程
)

# 3. 在训练数据上执行搜索
# 这可能会花费一些时间，因为它要训练 4 (C) * 4 (gamma) * 5 (cv folds) = 80 个模型！
grid_search.fit(X_train, y_train)

# 4. 查看搜索结果
print("\n---" + "-" * 25 + "GridSearchCV Results" + "-" * 25 + "---")
print(f"Best Hyperparameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# `grid_search` 对象在搜索结束后，会自动用找到的最佳参数在整个训练集上重新训练一个模型。
# 这个最佳模型存储在 `grid_search.best_estimator_` 中。

# --- 5. 用最佳模型在测试集上进行最终评估 ---
print("\n---" + "-" * 25 + "Final Evaluation on Test Set" + "-" * 25 + "---")
best_model = grid_search.best_estimator_
final_accuracy = best_model.score(X_test, y_test)
print(f"Accuracy of the best model on the unseen test set: {final_accuracy:.4f}")

# 总结:
# 1. **交叉验证**是评估模型泛化能力的黄金标准，它比单次划分更可靠。
# 2. **网格搜索**是一种系统性的超参数搜索方法，它会尝试所有可能的参数组合。
# 3. **GridSearchCV** 将网格搜索和交叉验证完美地结合在一起，是模型优化的标准工具。
# 4. 调优流程：在**训练集**上使用GridSearchCV寻找最佳参数，然后用找到的最佳模型在**测试集**上进行一次最终的、独立的评估。
# 5. 永远不要在测试集上进行调优，否则你的评估结果会过于乐观，无法反映模型在真实世界中的表现。
