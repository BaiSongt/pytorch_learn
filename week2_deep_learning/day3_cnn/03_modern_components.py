
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 前言 --- 
# 在经典架构之后，一系列创新的网络组件被提出，它们在提升模型性能和效率方面起到了关键作用。
# 本脚本将介绍其中几种最核心的现代CNN组件。

# --- 1. 残差连接 (Residual Connection) 回顾 ---
# - **核心思想**: H(x) = F(x) + x
# - **重要性**: ResNet的核心，解决了深度网络的梯度消失问题，是现代深度学习的基石。
#   几乎所有SOTA（State-of-the-art）模型都以某种形式使用了残差连接。

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x # 保存输入
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # 将输入直接加到输出上
        return F.relu(out)

print("--- 1. Residual Connection is the cornerstone of modern deep learning. ---")
print("-"*30)

# --- 2. 瓶颈层设计 (Bottleneck Design) ---
# - **动机**: 在非常深的网络中（如ResNet-50/101），标准的残差块（两个3x3卷积）计算成本仍然很高。
# - **设计**: 使用 1x1 -> 3x3 -> 1x1 的卷积序列替代两个3x3卷积。
#   1. 第一个`nn.Conv2d(256, 64, kernel_size=1)`: **降维**，将通道数从256减少到64，像瓶颈一样“压缩”特征。
#   2. 中间的`nn.Conv2d(64, 64, kernel_size=3)`: 在压缩后的低维空间中进行特征提取，计算成本大大降低。
#   3. 最后的`nn.Conv2d(64, 256, kernel_size=1)`: **升维**，将通道数恢复到256。
# - **效果**: 在获得相似感受野和性能的同时，参数量和计算量都显著减少。

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        # ... 此处省略了快捷连接的实现 ...

print("--- 2. Bottleneck Design reduces computation with 1x1 convolutions. ---")
print("-"*30)

# --- 3. 深度可分离卷积 (Depthwise Separable Convolution) ---
# - **动机**: 标准卷积的计算成本很高。一个3x3卷积核需要同时考虑空间信息和通道信息。
# - **核心思想**: MobileNet的核心组件。将标准卷积分解为两步：
#   1. **深度卷积 (Depthwise Convolution)**: 每个输入通道**独立地**被一个自己的卷积核进行滤波。这一步只处理空间信息，不混合通道信息。
#   2. **点卷积 (Pointwise Convolution)**: 使用1x1的卷积核来混合深度卷积产生的输出通道。这一步只处理通道信息。
# - **效果**: 参数量和计算量可以减少到标准卷积的1/8到1/9，同时保持了相当的性能，是移动端和嵌入式设备上CNN的首选。

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.depthwise(x)))
        out = F.relu(self.bn2(self.pointwise(out)))
        return out

print("--- 3. Depthwise Separable Convolution is key for efficient models (e.g., MobileNet). ---")
print("-"*30)

# --- 4. 注意力模块: Squeeze-and-Excitation (SE) Net ---
# - **动机**: 不同的特征通道的重要性是不同的。网络能否自动学习到这种重要性，并“关注”更有用的特征通道？
# - **核心思想**: 
#   1. **Squeeze**: 将每个通道的特征图（例如 16x16xC）通过全局平均池化（Global Average Pooling）压缩成一个单一的数值，得到一个 1x1xC 的向量。这个向量可以看作是当前所有通道的描述符。
#   2. **Excitation**: 将这个描述符向量送入一个迷你的两层全连接网络（一个瓶颈结构），来学习通道间的非线性关系。输出仍然是一个 1x1xC 的向量，但它的值（通常在0到1之间，通过Sigmoid激活）代表了每个通道的“重要性分数”或“权重”。
#   3. **Rescale**: 将学习到的权重向量逐通道地乘回到原始的特征图上，从而实现了对重要通道的增强和对不重要通道的抑制。
# - **效果**: SE模块是一个轻量级的、即插即用的组件，可以轻松地集成到现有的任何CNN架构中（如ResNet -> SE-ResNet），并带来显著的性能提升。

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.shape
        # Squeeze
        y = self.squeeze(x).view(batch_size, num_channels)
        # Excitation
        y = self.excitation(y).view(batch_size, num_channels, 1, 1)
        # Rescale
        return x * y.expand_as(x)

print("--- 4. Squeeze-and-Excitation Block enables channel-wise attention. ---")

# 总结:
# 1. **残差连接**是基础。
# 2. **瓶颈设计**和**深度可分离卷积**是构建高效网络、减少参数和计算量的关键工具。
# 3. **注意力机制 (如SE Block)** 让网络能够自适应地重新校准特征，是提升模型性能的有效插件。
# 现代的CNN架构通常是这些组件的巧妙组合。
