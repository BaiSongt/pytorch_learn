
import torch
import torch.nn as nn
import torchvision.models as models

# --- 前言 --- 
# 基础的CNN架构非常擅长图像分类，即判断一张图片“是什么”。
# 通过对这个基础架构进行巧妙的修改和扩展，我们可以解决各种更复杂的计算机视觉任务。
# 本脚本将概念性地介绍几种主流的CNN应用。

# --- 1. 基础：图像分类 (Image Classification) ---
# - **任务**: 为整张图片分配一个类别标签。
# - **架构**: 一个标准的CNN（如ResNet），接收一张图片，输出一个类别概率向量。
#   [Image] -> [CNN Feature Extractor] -> [Classifier Head] -> [Class Probs]

print("--- 1. Image Classification: The foundation. ---")
# 一个典型的分类模型
classification_model = models.resnet18()
classification_model.fc = nn.Linear(classification_model.fc.in_features, 10) # 适应10个类别
print("-"*30)

# --- 2. 目标检测 (Object Detection) ---
# - **任务**: 在图片中找到所有感兴趣的物体，并用边界框（bounding box）标出它们的位置，同时识别出它们的类别。
# - **核心思想**: 在CNN特征提取器的基础上，增加两个并行的“头”：
#   1. **分类头 (Classification Head)**: 判断每个候选框内“是什么”物体。
#   2. **回归头 (Regression Head)**: 预测每个候选框的精确位置（通常是4个坐标值：x, y, w, h）。
# - **著名架构**: Faster R-CNN, YOLO (You Only Look Once), SSD.

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, num_boxes):
        super().__init__()
        # 1. 使用一个预训练的CNN作为特征提取的“主干网络”
        self.backbone = models.resnet50(pretrained=True).features
        
        # 2. 在主干网络之上，添加自定义的头
        # (这是一个极度简化的例子，真实模型要复杂得多)
        self.classifier = nn.Linear(2048, num_classes) # 分类头
        self.regressor = nn.Linear(2048, num_boxes * 4) # 回归头，每个box有4个坐标

    def forward(self, x):
        features = self.backbone(x)
        # ... (经过RoI Pooling等操作) ...
        # class_logits = self.classifier(features)
        # box_predictions = self.regressor(features)
        # return class_logits, box_predictions
        pass

print("--- 2. Object Detection: Adds a regression head for bounding boxes. ---")
print("[Image] -> [Backbone] -> [Classifier Head] & [Regressor Head]")
print("-"*30)

# --- 3. 语义分割 (Semantic Segmentation) ---
# - **任务**: 对图像中的**每一个像素**进行分类，从而将不同物体或区域分割开。
# - **核心思想**: 使用“编码器-解码器 (Encoder-Decoder)”架构。
#   1. **编码器 (Encoder)**: 通常是一个标准的CNN（如ResNet），它通过卷积和池化，逐步地**降采样**，将输入图像压缩成深层的、低分辨率的特征图。这个过程负责“理解”图像内容。
#   2. **解码器 (Decoder)**: 负责将编码器输出的低分辨率特征图，通过**上采样**操作（如转置卷积`nn.ConvTranspose2d`或双线性插值`upsample`），逐步地恢复到原始图像的分辨率。这个过程负责“定位”和精细化边界。
#   3. **快捷连接 (Skip Connections)**: 为了在解码过程中恢复因降采样而丢失的空间细节，通常会从编码器的对应层级向解码器引入“快捷连接”（U-Net的核心思想）。
# - **著名架构**: FCN (Fully Convolutional Network), U-Net, DeepLab.

class UNetLikeModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 编码器 (下采样)
        self.encoder_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.encoder_block2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        
        # 解码器 (上采样)
        # ConvTranspose2d 用于上采样
        self.decoder_block1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU())
        self.decoder_block2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU()) # 加上来自encoder1的skip connection后通道数为128
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码过程
        enc1 = self.encoder_block1(x) # -> 64 channels
        enc2 = self.encoder_block2(enc1) # -> 128 channels
        
        # 解码过程
        dec1 = self.decoder_block1(enc2) # -> 64 channels
        # 引入快捷连接
        skip_connection = torch.cat([dec1, enc1], dim=1) # 拼接来自编码器的特征
        dec2 = self.decoder_block2(skip_connection)
        
        return self.final_conv(dec2)

print("--- 3. Semantic Segmentation: Uses an Encoder-Decoder structure (e.g., U-Net). ---")
print("[Image] -> [Encoder (Downsample)] -> [Decoder (Upsample)] -> [Pixel-wise Mask]")
print("-"*30)

# --- 4. 风格迁移 (Style Transfer) ---
# - **任务**: 将一张“内容”图像的整体构图与另一张“风格”图像的纹理、笔触、颜色等风格相结合，生成一张新的艺术图像。
# - **核心思想**: 使用一个预训练的CNN（如VGGNet）的中间层特征。
#   1. **内容损失**: 在网络较深的某一层，计算生成图像的特征图与内容图像的特征图之间的**均方误差**。这能确保生成图像的“内容”是相似的。
#   2. **风格损失**: 在网络的多个层级，计算生成图像特征图的**格拉姆矩阵(Gram Matrix)**与风格图像特征图的格拉姆矩阵之间的均方误差。格拉姆矩阵可以被认为是特征之间相关性的度量，它捕捉了纹理和笔触等“风格”信息。
# - **过程**: 随机生成一张噪声图像，然后通过梯度下降不断修改这张图像，以同时最小化内容损失和风格损失。

print("--- 4. Style Transfer: Uses deep features from a pre-trained CNN to define content and style losses. ---")

# 总结:
# - **分类**: 是什么？
# - **检测**: 是什么，在哪里？ (分类 + 边界框回归)
# - **分割**: 每个像素是什么？ (编码器-解码器结构)
# - **风格迁移**: 利用中间层特征进行艺术创作。
# 所有的这些高级应用，其基础都源于我们已经学习的CNN特征提取思想。
