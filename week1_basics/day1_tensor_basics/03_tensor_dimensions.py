import numpy as np
import torch

## tensor operation

has_cuda = torch.cuda.is_available()
if has_cuda:
    print("torch cuda verison: ", torch.version.cuda)  # 应输出 12.6
    print(torch.cuda.get_device_capability(0))


a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状 (2,3)
b = 2  # 标量（视为形状 (1,1)）
result = a + b  # b 被广播为 (2,3)
print(f"result: {result}")
print(result.shape)

result_reshape = result.reshape(1, 6)
print(f"result reshape: {result_reshape.shape} ")
z = result_reshape.view(1, 6)
print(f"z: {z.shape} ")
print(f"z: {z} ")

### 1 reshape  可以处理非连续情况
# 示例1: 一维转二维
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape(2, 3)  # 输出: [[1,2,3],[4,5,6]]
print(y)

# 示例2: 三维转二维
z = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
w = z.reshape(9)  # 输出: [1,2,3,4,5,6,7,8,9]
print(w)

### 2. view   不能处理非连续情况

# 示例1: 二维转三维
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a.view(3, 3)  # 输出: [[1,2,3],[4,5,6],[7,8,9]]
print(b)

# 示例2: 一维转三维
c = torch.tensor([1, 2, 3, 4, 5, 6])
d = c.view(2, 3)  # 输出: [[1,2,3],[4,5,6]]
print(d)


### 3 squeeze 移除维度，但是保留数据
# 移除指定轴大小为1 的 维度
# 不指定  默认移除大小为 1 的维度

# 示例1: 移除单个维度
e = torch.tensor([[1, 2, 3], [4, 5, 6]])
f = e.squeeze()  # 输出: [1,2,3,4,5,6]
print(e.shape)
print(f.shape)

# 示例2: 移除指定维度
g = torch.tensor([[[1, 2], [3, 4]]])
h = g.squeeze(1)  # 输出: [[1,2],[3,4]]
print(h)
print(g.shape)
print(h.shape)

g = torch.tensor([[1, 2, 3], [2, 3, 4], [2, 3, 4]])
h = torch.squeeze(g, 1)
print(h)
print(g.shape)
print(h.shape)

### 4. unsqueeze 指定轴添加一个大小为1 的维度
# 示例1: 添加一个维度
i = torch.tensor([1, 2, 3])
j = i.unsqueeze(0)  # 输出: [[1,2,3]]
print(j)
print(j.shape)

# 示例2: 添加多个维度
k = torch.tensor([1])
l = k.unsqueeze(1).unsqueeze(0)  # 输出: [[[1]]]
print(l)
print(l.shape)


### 5. transpose ：交换两个指定维度（仅适用于二维张量）
# 示例1: 交换行和列
m = torch.tensor([[1, 2], [3, 4]])
print(f"m : {m}")
n = m.transpose(0, 1)  # 输出: [[1,3],[2,4]]
print(f"n : {n}")

# 示例2: 交换非连续维度（需先确保张量连续）
o = torch.tensor([[1, 2], [3, 4]])
p = o.transpose(0, 1)  # 输出: [[1,3],[2,4]]
print(p)


### 6. permute
### 重新排列任意维度的顺序（支持多维张量）。
# 示例1: 三维张量重新排列维度
q = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
r = torch.permute(q, (1, 2, 0))
# 输出: [[[1, 2],[3, 4]], [[5, 6],[7, 8]]]
print(r)

# 示例2: 交换非连续维度
s = torch.tensor([[[1, 2], [3, 4]]])
t = s.permute(2, 1, 0)
# 输出: [[[[1, 2], [3, 4]]]]
print(t)

# (0, 1, 2) -> permute (2, 1, 0)
#  0 --> 2 , 1 --> 1 , 2 --> 0


### 7. cat
## 沿指定维度拼接张量（要求其他维度完全一致）。

# 示例1: 沿 dim=0 拼接
u = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([[5, 6], [7, 8]])
w = torch.cat([u, v], dim=0)  # 输出: [[1,2],[3,4],[5,6],[7,8]]
print(w)

# 示例2: 沿 dim=1 拼接
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
z = torch.cat([x, y], dim=1)  # 输出: [[1,2,5,6],[3,4,7,8]]
print(z)


### 8. stack
# 多个张量沿新维度堆叠（生成一个新的维度）。

# 示例1: 沿 dim=0 堆叠
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = torch.stack([a, b], dim=0)  # 输出: [[1,2],[3,4]]
print(c)

# 示例2: 沿 dim=1 堆叠
d = torch.tensor([[1, 2],
                   [3, 4]])
e = torch.tensor([[5, 6],
                   [7, 8]])
f = torch.stack([d, e], dim=1)
# 输出: [[[1,2],[5,6]],
#        [[3,4],[7,8]]]
print(f)
