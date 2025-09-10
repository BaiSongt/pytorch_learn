# Day 1: PyTorch 张量基础操作

## 学习目标
掌握 PyTorch 中张量的基本操作和计算

## 详细内容

### 1. 张量创建
```python
# 示例代码结构
01_tensor_creation.py
```
- 从列表/NumPy数组创建张量
- 常用初始化方法（zeros, ones, rand, randn）
- 指定数据类型和设备
- arange, linspace 使用

### 2. 张量操作
```python
02_tensor_operations.py
```
- 基本算术运算
- 矩阵运算（点积、矩阵乘法）
- 广播机制
- 原位操作（inplace operations）

### 3. 张量维度操作
```python
03_tensor_dimensions.py
```
- reshape 和 view
- squeeze 和 unsqueeze
- transpose 和 permute
- cat 和 stack

### 4. 索引与切片
```python
04_indexing_slicing.py
```
- 基本索引
- 布尔索引
- 高级索引
- gather 和 scatter

### 5. GPU 加速
```python
05_gpu_operations.py
```
- CPU/GPU 张量转换
- 设备管理
- 性能对比实验

## 练习项目
1. 图像数据张量处理
2. 矩阵运算实践
3. 张量变形案例

## 作业
1. 实现一个简单的图像滤波器
2. 编写矩阵运算性能测试脚本
3. 完成张量操作练习题

## 参考资源
- PyTorch 官方文档
- 在线练习平台
- 实践代码示例
