# Day 2: 自动求导与数据加载

## 学习目标
理解 PyTorch 自动求导机制和数据加载器的使用

## 详细内容

### 1. 自动求导基础
```python
01_autograd_basics.py
```
- 计算图概念
- requires_grad 设置
- backward() 使用
- grad 属性和梯度访问

### 2. 复杂梯度计算
```python
02_complex_gradients.py
```
- 多变量函数求导
- 高阶导数
- 梯度累积
- 自定义自动求导函数

### 3. Dataset 实现
```python
03_custom_dataset.py
```
- Dataset 类继承
- __getitem__ 和 __len__ 实现
- 数据预处理
- 数据增强技术

### 4. DataLoader 使用
```python
04_dataloader_usage.py
```
- DataLoader 配置
- batch_size 设置
- shuffle 和采样
- num_workers 多进程加载

### 5. 数据预处理和增强
```python
05_data_augmentation.py
```
- torchvision.transforms
- 自定义转换
- 数据标准化
- 数据增强策略

## 练习项目
1. 函数优化问题
2. 自定义图像数据集
3. 数据加载性能优化

## 作业
1. 实现一个自定义梯度函数
2. 创建自定义数据集和加载器
3. 设计数据增强流程

## 参考资源
- PyTorch autograd 教程
- Dataset/DataLoader 官方文档
- 数据增强最佳实践
