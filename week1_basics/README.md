# 第一周：PyTorch 基础与机器学习

## 学习目标
通过本周的学习，掌握 PyTorch 的基础操作和机器学习的核心概念。

## 1. PyTorch 基础 (2天)

### Day 1: 基础操作
- 张量（Tensor）基础
  - 张量创建与初始化
  - 张量运算（加减乘除、矩阵运算）
  - 张量维度操作（reshape, view, squeeze）
  - 张量索引与切片
  - CPU/GPU 张量转换

### Day 2: 高级特性
- 自动求导（Autograd）
  - 计算图理解
  - requires_grad 使用
  - backward() 反向传播
  - grad 梯度获取
- 数据加载
  - Dataset 类实现
  - DataLoader 使用
  - 数据预处理和增强
  - 自定义数据集

## 2. 机器学习基础 (3天)

### Day 3: 回归算法
- 线性回归
  - 模型构建
  - 损失函数选择
  - 优化器使用
  - 模型训练与评估
- 多项式回归
  - 特征工程
  - 过拟合与欠拟合

### Day 4: 分类算法
- 逻辑回归
  - 二分类问题
  - 多分类扩展
  - 正则化技术
- 支持向量机（SVM）
  - 核函数使用
  - 软间隔 SVM
  - 参数调优

### Day 5: 集成学习
- 决策树
  - 树的构建
  - 特征重要性
  - 剪枝技术
- 随机森林
  - Bagging 原理
  - 随机特征选择
  - 模型集成

## 实践项目
1. 波士顿房价预测（回归）
2. MNIST 手写数字分类（分类）
3. 泰坦尼克生存预测（决策树）

## 代码结构
```
week1_basics/
├── day1_tensor_basics/
├── day2_autograd_data/
├── day3_regression/
├── day4_classification/
└── day5_ensemble/
```

## 学习资源
- PyTorch 官方文档
- Scikit-learn 文档
- Kaggle 数据集
- 推荐书籍：《深度学习入门之PyTorch》
