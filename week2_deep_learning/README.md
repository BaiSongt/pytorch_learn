# 第二周：深度学习基础

## 学习目标
掌握深度学习的基本原理和常用网络结构，能够独立实现简单的深度学习模型。

## 1. 神经网络基础 (2天)

### Day 1: 神经网络基础
- 神经网络基本概念
  - 感知机结构
  - 前向传播
  - 反向传播算法
  - 梯度消失/爆炸
- 激活函数
  - ReLU及其变体
  - Sigmoid
  - Tanh
  - 激活函数选择

### Day 2: 深度网络优化
- 优化器详解
  - SGD及其变体
  - Adam优化器
  - 学习率调度
- 正则化技术
  - Dropout实现
  - BatchNorm原理
  - L1/L2正则化
  - Early Stopping

## 2. 深度学习实践 (3天)

### Day 3: CNN架构
- 卷积神经网络基础
  - 卷积层实现
  - 池化层
  - 全连接层
- 经典CNN架构
  - LeNet
  - AlexNet
  - VGG
  - ResNet

### Day 4: 序列模型
- RNN基础
  - 循环神经网络原理
  - 长短期记忆（LSTM）
  - 门控循环单元（GRU）
- 序列处理技术
  - 词嵌入
  - 双向RNN
  - 注意力机制基础

### Day 5: 模型部署
- 模型保存与加载
  - state_dict使用
  - 完整模型保存
- 模型评估
  - 验证集划分
  - 交叉验证
  - 性能指标计算
- 部署基础
  - TorchScript
  - ONNX格式
  - 模型量化

## 实践项目
1. CIFAR-10图像分类（CNN）
2. 新闻文本分类（RNN/LSTM）
3. 时间序列预测（GRU）

## 代码结构
```
week2_deep_learning/
├── day1_neural_networks/
├── day2_optimization/
├── day3_cnn/
├── day4_rnn/
└── day5_deployment/
```

## 学习资源
- Deep Learning with PyTorch
- CS231n 课程
- Papers with Code
- 经典论文阅读清单
