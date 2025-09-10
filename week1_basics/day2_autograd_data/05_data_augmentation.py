
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image # Python Imaging Library, a dependency of torchvision
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 什么是数据预处理和增强? ---
# - 数据预处理 (Preprocessing): 将原始数据转换为适合神经网络处理的格式。最常见的操作是：
#   1. 将数据转换为Tensor。
#   2. 对数据进行标准化 (Normalization)，使其具有零均值和单位方差，这有助于模型更快、更稳定地收敛。
#
# - 数据增强 (Augmentation): 在训练过程中，对数据（尤其是图像）进行一系列随机的变换,
#   生成新的、略有不同的训练样本。这相当于扩充了数据集，可以有效地减少过拟合，提高模型的泛化能力。
#   常见的增强操作包括：随机裁剪、翻转、旋转、颜色抖动等。

# --- 2. `torchvision.transforms` 简介 ---
# `transforms` 模块提供了大量常用的预处理和增强操作。
# `transforms.Compose` 可以将多个变换操作组合成一个序列。

# --- 3. 图像数据增强示例 ---
# 我们首先创建一个虚拟的、简单的图像，以便观察变换的效果。

def create_dummy_image():
    """创建一个 128x128 的RGB图像，左上角有一个红色的正方形"""
    img_array = np.zeros((128, 128, 3), dtype=np.uint8)
    img_array[:50, :50, 0] = 255 # 红色通道
    return Image.fromarray(img_array, 'RGB')

dummy_image = create_dummy_image()

# 定义一个图像增强和预处理的流程
# 这是一个在图像分类任务中非常典型的变换序列
train_transforms = transforms.Compose([
    # --- Augmentation transforms --- (只在训练时使用)
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 1.0)), # 随机裁剪并缩放到指定大小
    transforms.RandomHorizontalFlip(p=0.5), # 以50%的概率水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 随机调整亮度和对比度

    # --- Preprocessing transforms --- (训练和测试时都需要)
    transforms.ToTensor(), # 将PIL Image或numpy.ndarray从 (H, W, C) 转换为 (C, H, W) 的Tensor，并把像素值从 [0, 255] 缩放到 [0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 标准化：output = (input - mean) / std
                                                                    # 这里将 [0, 1] 的范围变换到 [-1, 1]
])

# 应用变换
transformed_image_tensor = train_transforms(dummy_image)

print("---" + "-" * 10 + " Image Transforms " + "-" * 10 + "---")
print(f"Original image type: {type(dummy_image)}")
print(f"Transformed image type: {type(transformed_image_tensor)}")
print(f"Transformed image shape: {transformed_image_tensor.shape}") # (C, H, W)
print("-"+"-" * 30)

# 可视化增强效果
def show_transformed_images(image, transforms, num_images=4):
    fig, axes = plt.subplots(1, num_images + 1, figsize=(15, 3))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(1, num_images + 1):
        # 每次调用都会进行一次新的随机变换
        transformed_tensor = transforms(image)
        # 为了可视化，我们需要将标准化的Tensor转换回PIL Image
        # 1. 反标准化: tensor * std + mean
        unnormalized_tensor = transformed_tensor * 0.5 + 0.5
        # 2. 将 (C, H, W) 转回 (H, W, C)
        unnormalized_tensor = unnormalized_tensor.permute(1, 2, 0)
        # 3. 转换为numpy数组
        img_display = unnormalized_tensor.numpy()
        axes[i].imshow(np.clip(img_display, 0, 1)) # clip确保值在[0,1]范围内
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')
    plt.suptitle("Data Augmentation Examples")
    plt.show()

show_transformed_images(dummy_image, train_transforms)

# --- 4. 在 Dataset 中集成 Transforms ---
# 标准做法是在 `__init__` 中接收一个 transform 对象，然后在 `__getitem__` 中应用它。

class ImageDatasetWithTransforms(Dataset):
    def __init__(self, num_samples=10, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # 在实际项目中，这里会加载文件路径列表等

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 加载原始数据（例如，从磁盘读取图片）
        image = create_dummy_image()
        label = idx % 2 # 假设一个虚拟标签

        # 2. 应用变换 (如果提供了)
        if self.transform:
            image = self.transform(image)

        return image, label

# 创建一个应用了我们定义的变换的数据集实例
transformed_dataset = ImageDatasetWithTransforms(transform=train_transforms)

# 获取一个样本来验证
first_sample, first_label = transformed_dataset[0]
print("\n---" + "-" * 10 + " Dataset with Transforms " + "-" * 10 + "---")
print(f"Sample type from dataset: {type(first_sample)}")
print(f"Sample shape from dataset: {first_sample.shape}")
print(f"Sample label: {first_label}")

# 总结:
# 1. 数据预处理和增强是构建高效、鲁棒的视觉模型的关键步骤。
# 2. `torchvision.transforms` 提供了丰富的、易于使用的工具。
# 3. `transforms.Compose` 用于将多个操作链接成一个流水线。
# 4. 数据增强（如随机裁剪、翻转）只在训练阶段使用，以增加数据多样性。
# 5. 数据预处理（如ToTensor, Normalize）在训练和测试阶段都需要使用，以确保模型接收到的数据分布一致。
# 6. 将 `transform` 作为参数传入Dataset类，并在 `__getitem__` 中应用，是标准的设计模式。
