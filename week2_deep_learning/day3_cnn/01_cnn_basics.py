
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 前言 --- 
# 卷积神经网络 (Convolutional Neural Networks, CNNs) 是深度学习在计算机视觉领域取得巨大成功的关键。
# 与全连接网络（MLP）不同，CNN利用了图像数据的空间结构特性（如像素的局部相关性），
# 通过“参数共享”和“稀疏连接”大大减少了模型参数，使其能高效地处理高维图像数据。
# 本脚本将介绍构成CNN的核心组件：卷积层和池化层。

# --- 1. 准备一个虚拟的输入图像 --- 
# 在PyTorch中，图像数据通常被表示为一个4D张量，形状为 (N, C, H, W)：
# - N: Batch size (批次大小，即一次处理的图像数量)
# - C: Channels (通道数。对于灰度图是1，对于RGB彩色图是3)
# - H: Height (图像高度)
# - W: Width (图像宽度)

# 我们创建一个批次为1，3个通道（如RGB），高和宽都为8像素的虚拟图像
batch_size = 1
channels = 3
height = 8
width = 8

# requires_grad=True 以便后续观察梯度（虽然本脚本不进行反向传播）
input_image = torch.randn(batch_size, channels, height, width, requires_grad=True)
print(f"--- Input Image Tensor ---")
print(f"Shape of the input image: {input_image.shape}")
print("-"*30)

# --- 2. 卷积层 (Convolutional Layer) ---
# `nn.Conv2d` 是PyTorch中实现2D卷积的核心。

# - `in_channels`: 输入通道数。必须与输入图像的通道数匹配（这里是3）。
# - `out_channels`: 输出通道数。这等于卷积层中“卷积核（filter/kernel）”的数量。每个卷积核负责学习一种特定的特征（如边缘、角点、纹理等）。
# - `kernel_size`: 卷积核的大小。一个3x3的卷积核会查看一个3x3的像素邻域。
# - `stride`: 卷积核在图像上滑动的步长。stride=1表示一次移动一个像素。
# - `padding`: 在图像边界周围添加的像素层数。Padding可以帮助控制输出特征图的尺寸，并更好地处理图像边缘信息。

print("--- 2. Convolutional Layer (nn.Conv2d) ---")
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 前向传播
conv_output = conv_layer(input_image)

# **输出尺寸计算公式**: W_out = (W_in - K + 2P) / S + 1
# W_in: 输入尺寸 (8)
# K: 卷积核尺寸 (3)
# P: 填充 (1)
# S: 步长 (1)
# W_out = (8 - 3 + 2*1) / 1 + 1 = 8
# 高度和宽度使用相同的公式计算。
print(f"Shape after convolution: {conv_output.shape}")
print("The output shape is (N, out_channels, H_out, W_out).")
print("Each of the 16 output channels is a 'feature map'.")
print("-"*30)

# --- 3. 激活函数 (Activation Function) ---
# 和MLP一样，卷积层之后通常会跟一个非线性激活函数，最常用的仍然是ReLU。

print("--- 3. Activation Function ---")
activated_output = F.relu(conv_output)
print("ReLU is applied element-wise, so the shape does not change.")
print(f"Shape after ReLU: {activated_output.shape}")
print("-"*30)

# --- 4. 池化层 (Pooling Layer) ---
# 池化层用于对特征图进行“降采样”，以达到以下目的：
# 1. 减小特征图的空间尺寸，从而减少后续层的参数和计算量。
# 2. 增大“感受野”，让后续的卷积层能看到更广阔的原始图像区域。
# 3. 提供一定程度的“平移不变性”，使网络对物体在图像中的微小位移不那么敏感。

# **最大池化 (Max Pooling)**: `nn.MaxPool2d`
# - 在一个窗口（如2x2）内，只取该窗口中最大的值作为输出。
# - `kernel_size`: 池化窗口的大小。
# - `stride`: 窗口滑动的步长。通常设为与kernel_size相同，以实现不重叠的池化。

print("--- 4. Pooling Layer (nn.MaxPool2d) ---")
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

pooled_output = pool_layer(activated_output)

# **输出尺寸计算公式**: W_out = (W_in - K) / S + 1 (假设padding=0)
# W_in: 输入尺寸 (8)
# K: 池化窗口尺寸 (2)
# S: 步长 (2)
# W_out = (8 - 2) / 2 + 1 = 4
print(f"Shape after Max Pooling: {pooled_output.shape}")
print("The height and width are halved, reducing the data size by 75%.")
print("-"*30)

# --- 5. 感受野 (Receptive Field) ---
# 感受野是指，在CNN输出的特征图上，一个像素点对应回原始输入图像上的区域大小。
# 经过多层卷积和池化后，一个神经元可以“看到”比其直接连接的区域更大的范围。
# 例如，在我们的例子中，经过一个3x3卷积和一个2x2池化后，
# 输出特征图上的一个像素，其感受野已经大于3x3了。
# 堆叠更多的层可以持续增大感受野，从而让网络能够学习到更全局、更抽象的特征。

# 总结:
# 一个典型的CNN层块 (block) 的结构是：
# **Convolution -> Activation (ReLU) -> Pooling**
# 通过堆叠多个这样的层块，CNN可以逐步地从低级的边缘、纹理特征，学习到高级的、可识别的物体部件和整体概念。
