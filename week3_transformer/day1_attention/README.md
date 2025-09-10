# Day 1: 注意力机制基础

## 学习目标
深入理解注意力机制的原理和实现

## 详细内容

### 1. 注意力机制基础
```python
01_attention_basics.py
```
- Query、Key、Value概念
- 点积注意力
- 缩放点积注意力
- 注意力权重计算

### 2. Self-Attention实现
```python
02_self_attention.py
```
- Self-Attention数学原理
- 前向传播实现
- 反向传播理解
- 并行计算优化

### 3. Multi-Head Attention
```python
03_multihead_attention.py
```
- 多头注意力架构
- 头的并行计算
- 输出合并策略
- 维度变换处理

### 4. 位置编码
```python
04_positional_encoding.py
```
- 正弦位置编码
- 可学习位置编码
- 相对位置编码
- 位置信息注入

### 5. 注意力可视化
```python
05_attention_visualization.py
```
- 注意力权重可视化
- 注意力热力图
- 跨层注意力分析
- 交互式可视化工具

## 实践项目
1. 序列建模任务
   - 文本序列处理
   - 注意力机制应用
   - 结果可视化

2. 图像注意力实验
   - 图像特征提取
   - 注意力层应用
   - 注意力图可视化

## 作业
1. 实现基础注意力机制
2. 构建多头注意力模块
3. 完成注意力可视化项目

## 参考资源
- Attention Is All You Need论文
- PyTorch注意力实现
- 可视化工具教程
- 示例代码库
