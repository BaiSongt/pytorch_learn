### 4. 索引与切片
# - 基本索引
# - 布尔索引
# - 高级索引
# - gather 和 scatter


import numpy as np
import torch


## 一、基本索引（Integer Indexing）
# 作用：通过整数索引访问张量的元素或子张量。
# 特点：支持单个索引、切片（`start:end`）、步长（`start:end:step`）等。

### 示例 1：单个索引
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)  # (1, 2, 3)
print(x[0, 1])  # 输出: 2
print(x[1, 2])  # 输出: 6


### 示例 2：切片操作  [a, b) 左闭右开
y = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(y.shape)  # (1, 3, 3)
print(y[0:2, 1:3])  # 输出: [[2,3], [5,6]]
print(y[::2, ::2])  # 输出: [[1,3], [7,9]]  # 每隔一行、每列取一个元素


## 二、布尔索引（Boolean Indexing）
# 作用：通过布尔数组筛选张量元素，返回满足条件的子张量。
# 特点：布尔数组的形状需与原张量的维度一致。

### 示例 1：单维张量筛选
a = torch.tensor([1, 2, 3, 4, 5])
mask = torch.tensor([True, False, True, False, True])
print(a[mask])  # 输出: tensor([1, 3, 5])


### 示例 2：多维张量筛选
b = torch.tensor([[1, 2], [3, 4], [5, 6]])
mask = torch.tensor([[True, False], [False, True], [True, False]])
print(b[mask])  # 输出: tensor([1, 4, 5])

## 三、高级索引（Advanced Indexing）
# 作用：通过索引张量（或列表）直接访问元素，支持多维索引。
# 特点：可以同时指定多个维度的索引，返回一维张量。

### 示例 1：多维索引
c = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = torch.tensor([[0, 2], [1, 2], [0, 1]])
print("c[indices] terson:\n", c[indices])
# 输出:  tensor([[[1, 2, 3],
#               [7, 8, 9]],
#              [[4, 5, 6],
#               [7, 8, 9]],
#              [[1, 2, 3],
#               [4, 5, 6]]])

### 示例 2：混合索引（切片 + 索引）
d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(d[[0, 1], [1, 2]])
# 输出: tensor([2, 6])


## 四、gather（根据索引收集元素）
# 作用：根据指定的索引张量，从原始张量中收集元素。
# 特点：支持多维索引，返回的张量维度与索引张量一致。

### 示例 1：一维张量 gather
e = torch.tensor([10, 20, 30, 40])
index = torch.tensor([2, 0, 1])
print("gather: ", torch.gather(e, 0, index))
# 输出: tensor([30, 10, 20])


### 示例 2：多维张量 gather
f = torch.tensor([[10, 20], [30, 40], [50, 60]])
index = torch.tensor([[0, 1], [1, 0], [0, 1]])
print("gather: ", torch.gather(f, 1, index))
# 输出: tensor([[10, 20], [40, 30], [50, 60]])

# ==============================================
# 基本索引
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, 1])  # 单个元素访问

# 布尔索引
mask = torch.tensor([[True, False, True], [True, False, True]])
print(x[mask])  # 筛选行

# 高级索引
indices = torch.tensor([[0, 1], [1, 1]])
print(x[indices])  # 多维索引
# tensor([[[1, 2, 3],
#          [4, 5, 6]],
#         [[4, 5, 6],
#          [4, 5, 6]]])

# gather  根据索引收集
# torch.gather(input, dim, index)
print(x.shape)
index = torch.tensor([[1, 1, 0], [1, 1, 0]])
# 把 0 dim 上 的 换成 1， 1， 0 位置上的
print(torch.gather(x, 0, index))  # 根据索引收集
# tensor([[4, 5, 3],
#         [4, 5, 3]])

index = torch.tensor([[1, 1, 0], [0, 1, 1]])
print(torch.gather(x, 0, index))  # 根据索引收集
# tensor([[4, 5, 3],
#         [1, 5, 6]])


# =========================================
# scatter
# 在 PyTorch 中，`scatter` 是一种根据 索引 将值分散到目标张量中的操作，常用
# 于根据指定的索引位置更新张量的值。与 `gather` 相对，`gather` 是从张量中收集元
# 素，而 `scatter` 是将元素分散到指定位置。
## 1 维
# 初始化目标张量
input = torch.zeros(5, dtype=torch.float32)
print("原始张量:", input)

# 索引和源数据
index = torch.tensor([0, 2, 4, 1, 3])
src = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

# 使用 scatter 更新
out = torch.scatter(input, dim=0, index=index, src=src)
print("更新后张量:", out)

## 2 维
# 初始化目标张量
input = torch.zeros(3, 3)
print("原始张量:\n", input)

# 索引和源数据
index = torch.tensor([[0, 1], [1, 2], [0, 2]])
src = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

# 按 dim=1（列）操作
out = torch.scatter(input, dim=1, index=index, src=src)
print("更新后张量:\n", out)


## 3 维
# 初始化目标张量
input = torch.zeros(2, 2, 2)
print("原始张量:\n", input)
print(input.shape)

# 索引和源数据
index = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [0, 0]]])
print(f"index shape: {index.shape}")
src = torch.tensor(
    [
        [[10.0, 20.0], [30.0, 40.0]],
        [[50.0, 60.0], [70.0, 80.0]],
    ]
)

# 按 dim=2（深度）操作
out = torch.scatter(input, dim=2, index=index, src=src)
print("更新后张量:\n", out)
