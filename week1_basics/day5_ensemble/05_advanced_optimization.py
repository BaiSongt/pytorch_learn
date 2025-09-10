
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --- 前言 ---
# 本脚本将演示一个更完整的机器学习工作流程，包括对一个强大的集成模型进行高级优化，
# 并最终保存训练好的模型以备将来使用。这是将模型从实验阶段推向实际应用的关键步骤。

# --- 1. 准备数据 ---
# 我们使用一个相对大一点的数据集，以更好地模拟真实场景
X, y = make_classification(
    n_samples=2000, 
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("---" + "-" * 27 + "Dataset Information" + "-" * 27 + "---")
print(f"Training data shape: {X_train.shape}")
print("-" * 30)

# --- 2. 对集成模型进行超参数调优 ---
# 随机森林有许多可以调优的超参数，以平衡性能和计算成本。
# 我们将使用 GridSearchCV 来寻找最佳组合。

print("---" + "-" * 25 + "Hyperparameter Tuning for RandomForest" + "-" * 25 + "---")

# 定义一个更精细的参数网格
# 在实际项目中，这个网格可能会更大，搜索时间也更长
param_grid = {
    'n_estimators': [100, 200], # 树的数量
    'max_depth': [10, 20, None], # 树的最大深度 (None表示不限制)
    'min_samples_leaf': [1, 2, 4], # 叶子节点的最小样本数
    'max_features': ['sqrt', 'log2'] # 寻找最佳分裂时考虑的特征数量
}

# 创建 GridSearchCV 对象
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3, # 使用3折交叉验证以加快速度
    scoring='accuracy',
    n_jobs=-1, # 使用所有CPU核心
    verbose=2 # 打印更详细的日志
)

# 执行搜索
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGridSearchCV took {end_time - start_time:.2f} seconds.")
print(f"Best Hyperparameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
print("-" * 30)

# 获取最佳模型
best_rf_model = grid_search.best_estimator_

# --- 3. 性能与部署考量 ---
# - **训练时间 vs. 预测时间**: `n_estimators` (树的数量) 对训练和预测时间的影响是线性的。
#   更多的树通常会带来更好的性能，但也会增加计算成本和延迟。在需要快速响应的在线服务中，这是一个重要的权衡。
# - **模型大小**: 更多的树和更深的树会显著增加最终模型文件的大小。

# --- 4. 模型持久化：保存和加载模型 ---
# 当我们找到了最佳模型后，不需要每次都重新训练它。
# 我们可以将训练好的模型对象保存到磁盘，以便在其他程序中加载和使用。
# `joblib` 是 scikit-learn 推荐的用于保存包含大型numpy数组的对象的库。

print("---" + "-" * 27 + "Model Persistence" + "-" * 27 + "---")

# 定义模型保存路径
model_filename = 'best_random_forest_model.joblib'

# 1. 保存模型
print(f"Saving the best model to {model_filename}...")
joblib.dump(best_rf_model, model_filename)
print("Model saved successfully.")

# 2. 加载模型
# 假设我们在一个全新的程序中，需要使用这个模型进行预测
print(f"\nLoading the model from {model_filename}...")
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# 3. 使用加载的模型进行预测
# 我们可以用它在测试集上进行预测，并验证其性能是否与原始最佳模型相同
original_model_accuracy = best_rf_model.score(X_test, y_test)
loaded_model_accuracy = loaded_model.score(X_test, y_test)

print(f"\nOriginal best model accuracy on test set: {original_model_accuracy:.4f}")
print(f"Loaded model accuracy on test set:   {loaded_model_accuracy:.4f}")

assert np.isclose(original_model_accuracy, loaded_model_accuracy)
print("\nAccuracies match! The loaded model works correctly.")

# 总结:
# 1. 对强大的集成模型（如随机森林）进行超参数调优是提升性能的关键步骤。
# 2. `GridSearchCV` 是实现这一目标的标准工具，但需要注意其计算成本。
# 3. 在选择最终模型时，需要权衡其预测性能、预测速度和模型大小等部署要求。
# 4. 使用 `joblib.dump()` 和 `joblib.load()` 可以轻松地保存和加载训练好的scikit-learn模型，
#    这是将机器学习模型投入生产应用的基础。

