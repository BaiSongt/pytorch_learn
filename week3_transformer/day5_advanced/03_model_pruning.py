
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# --- 前言 ---
# 模型剪枝旨在通过移除“不重要”的权重来减小模型大小和（潜在地）加速推理。
# 一个典型的剪枝流程是：训练一个模型 -> 剪枝 -> 微调模型 -> ... 循环往复。

# --- 1. 准备一个模型 ---
# 我们创建一个简单的模型作为剪枝的对象。
# 在真实场景中，这应该是一个已经训练好的模型。

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()

print("--- 1. A Simple Model to Prune ---")
# 我们可以查看第一个线性层的权重
print("Original weights of fc1:")
print(model.fc1.weight)
print("---" * 30)

# --- 2. 应用剪枝 ---
# `torch.nn.utils.prune` 提供了多种剪枝方法。
# 我们将演示最基础的一种：L1非结构化剪枝 (L1Unstructured)。
# - **L1**: 以权重的大小（绝对值）作为其重要性的衡量标准。绝对值越小的权重越不重要。
# - **非结构化 (Unstructured)**: 逐个地移除权重，而不考虑它们在权重矩阵中的位置。这会产生一个稀疏矩阵。

print("--- 2. Applying Pruning ---")

# 选择要剪枝的模块和参数
module_to_prune = model.fc1
param_name = 'weight'

# 应用剪枝
# `amount=0.6` 表示我们想要移除60%的权重（将它们设为0）
prune.l1_unstructured(module_to_prune, name=param_name, amount=0.6)

# **剪枝是如何工作的？**
# `prune`模块并**不会**直接修改权重张量。相反，它会创建一个名为 `weight_mask` 的“掩码”
# 和一个名为 `weight_orig` 的原始权重备份。在每次前向传播时，模型实际使用的是 `weight_orig * weight_mask` 的结果。
# 这使得剪枝过程是可逆的，并且可以在训练过程中动态调整。

print("Weights of fc1 after pruning:")
# 我们看到的 `model.fc1.weight` 已经是被mask后的结果了
print(model.fc1.weight)
print("\nNotice the large number of zeros.")

# 我们可以查看模块上附加的剪枝信息
print("\nPruning hooks attached to the module:")
print(list(module_to_prune.named_buffers()))
print(list(module_to_prune.named_parameters()))
print("---" * 30)

# --- 3. 固化剪枝 (Making Pruning Permanent) ---
# 在剪枝和微调完成后，如果我们想让剪枝效果永久生效，并移除附加的mask和原始权重，
# 我们可以使用 `prune.remove`。

print("--- 3. Making Pruning Permanent ---")

prune.remove(module_to_prune, name=param_name)

print("Pruning hooks removed.")
print(list(module_to_prune.named_parameters())) # 现在只剩下永久被修改过的weight和bias了
print("---" * 30)

# --- 4. 剪枝后微调的重要性 ---
# **关键思想**: 剪枝会强制性地将许多权重设为0，这几乎总会导致模型性能的下降。
# 为了恢复损失的精度，在剪枝后对模型进行几个周期的**微调**是至关重要的一步。
# 微调过程会让剩余的未被剪枝的权重进行调整，以补偿被移除的权重所造成的影响。
# 一个常见的策略是“迭代剪枝”：训练 -> 剪枝 -> 微调 -> 再剪枝 -> 再微调 ...

print("--- 4. The Importance of Finetuning After Pruning ---")
print("Pruning almost always degrades model accuracy.")
print("Finetuning the model for a few epochs after pruning is crucial to recover performance.")

# --- 5. 结构化剪枝 vs. 非结构化剪枝 ---
# - **非结构化剪枝 (Unstructured Pruning)**: 
#   - **优点**: 灵活，可以达到非常高的稀疏度。
#   - **缺点**: 产生的稀疏矩阵在通用硬件（CPU/GPU）上通常带不来实际的推理加速，因为它破坏了矩阵的规整结构。需要专门的硬件或库才能利用这种稀疏性。
# - **结构化剪枝 (Structured Pruning)**:
#   - **思想**: 移除整个的结构单元，例如，移除整个神经元（对应权重矩阵的一列）或整个卷积核通道。
#   - **优点**: 直接减小了模型的尺寸和计算量，可以在任何硬件上都带来实际的加速。
#   - **实现**: `torch.nn.utils.prune` 也提供了如 `ln_structured` 等方法来实现结构化剪枝。

# 总结:
# 1. **剪枝**是一种通过移除不重要权重来压缩模型的技术。
# 2. PyTorch的 `torch.nn.utils.prune` 模块提供了一个灵活的框架来实现剪枝。
# 3. **剪枝后微调**是恢复模型精度的关键步骤。
# 4. **结构化剪枝**比非结构化剪枝通常更能带来实际的推理速度提升。
# 5. 剪枝、量化和知识蒸馏是模型压缩的三大主要技术，它们可以被组合使用以达到最佳效果。
