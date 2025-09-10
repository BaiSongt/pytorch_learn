
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 前言 --- 
# 本脚本将实现几个在CNN发展史上具有里程碑意义的经典网络架构。
# 理解它们的设计哲学和演进过程，是深入学习计算机视觉的基础。

# --- 1. LeNet-5 (1998) ---
# - **贡献**: 第一个被成功大规模应用的CNN，用于手写数字识别。
# - **结构**: Conv -> Pool -> Conv -> Pool -> FC -> FC -> Output，奠定了现代CNN的基本模板。
# - **特点**: 使用了Sigmoid/Tanh激活函数，平均池化。

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # LeNet-5的输入是 32x32 的单通道图像
        self.features = nn.Sequential(
            # C1: Conv, 6个5x5的核, 输出 28x28x6
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            # S2: AvgPool, 2x2的窗口, 输出 14x14x6
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # C3: Conv, 16个5x5的核, 输出 10x10x16
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            # S4: AvgPool, 2x2的窗口, 输出 5x5x16
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # C5: FC, 将5x5x16的特征图展平为400个节点，连接到120个节点
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            # F6: FC, 120个节点连接到84个节点
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # Output: FC, 84个节点连接到最终的类别数
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # 将特征图展平，送入分类器
        logits = self.classifier(x)
        return logits

print("--- 1. LeNet-5 Architecture ---")
lenet_model = LeNet5()
print(lenet_model)
print("-"*30)

# --- 2. AlexNet (2012) ---
# - **贡献**: 在ImageNet竞赛中取得巨大成功，引爆了深度学习的浪潮。
# - **特点**:
#   1. **首次使用ReLU**作为激活函数，极大地加速了收敛。
#   2. **首次使用Dropout**来防止过拟合。
#   3. 使用了重叠的最大池化 (Overlapping Max Pooling)。
#   4. 使用多GPU进行训练。

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # 自适应平均池化，确保输出尺寸固定
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

print("--- 2. AlexNet Architecture ---")
alexnet_model = AlexNet()
# print(alexnet_model) # The model is large, printing is omitted
print("AlexNet introduced ReLU and Dropout, setting a new standard.")
print("-"*30)

# --- 3. VGGNet (2014) ---
# - **贡献**: 证明了网络的深度是提升性能的关键因素。
# - **特点**: 结构非常简洁、统一。完全使用**3x3的小卷积核**和**2x2的池化核**。
#   通过堆叠多个3x3卷积层，可以获得与一个大的卷积层相同的感受野，但参数更少，非线性也更强。
#   例如，2个3x3卷积的感受野等同于1个5x5卷积，3个3x3卷积的感受野等同于1个7x7卷积。

# VGG-16的结构实现 (这里只展示结构，不实现完整的模型)
print("--- 3. VGGNet Architecture ---")
print("VGGNet's philosophy: depth is key. It uses very small 3x3 conv filters exclusively.")
print("Example block: Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> MaxPool2x2")
print("-"*30)

# --- 4. ResNet (Residual Network, 2015) ---
# - **贡献**: 解决了极深网络（超过百层）的训练问题，是目前影响最深远的网络架构之一。
# - **核心思想**: **残差连接 (Residual Connection)** 或称 **快捷连接 (Skip Connection)**。
#   它允许梯度在反向传播时“跳过”一些层，直接流向更早的层，极大地缓解了梯度消失问题。
#   网络不再是学习从x到H(x)的直接映射，而是学习一个“残差” F(x) = H(x) - x。

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接，用于处理输入和输出维度不匹配的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 核心：将输入x直接加到输出上
        out = F.relu(out)
        return out

print("--- 4. ResNet Architecture ---")
print("ResNet's key innovation: the residual/skip connection.")
print("This allows training of extremely deep networks (100+ layers). H(x) = F(x) + x")

# 我们可以用一个残差块来测试
res_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
# print(res_block)

# 总结:
# - **LeNet-5**: 奠定了基础模板。
# - **AlexNet**: 引入ReLU和Dropout，开启了深度学习时代。
# - **VGGNet**: 证明了深度很关键，并推广了使用小卷积核的设计模式。
# - **ResNet**: 通过残差连接解决了深度网络的训练难题，是目前大多数现代CNN架构的基础。
