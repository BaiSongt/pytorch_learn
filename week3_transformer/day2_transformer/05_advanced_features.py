
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# --- 前言 --- 
# 随着模型规模的增大和应用场景的复杂化，标准Transformer的局限性开始显现。
# 本脚本将概念性地介绍一些用于解决这些挑战的高级技术，为你进一步的学习提供一个路线图。

# --- 1. 处理长序列 (Handling Long Sequences) ---
# **问题**: 标准自注意力的复杂度是 O(seq_len²)。当序列长度(seq_len)从512增加到4096时，
# 计算量和内存占用会增加 4096²/512² = 64倍，这在计算上是难以承受的。

# **解决方案类别**: 高效Transformer (Efficient Transformers)
# 这些模型试图在保持强大性能的同时，将注意力复杂度降低到 O(seq_len * log(seq_len)) 或 O(seq_len)。

# **A. 稀疏注意力 (Sparse Attention)**
# - **代表模型**: Longformer, BigBird
# - **思想**: 与其让每个token关注所有其他token，不如只关注一个精心选择的、稀疏的子集。
#   例如，一个token可以关注它周围的几个“局部”token，再加上几个预先选定的“全局”token（如CLS）。

# **B. 线性化注意力 (Linearized Attention)**
# - **代表模型**: Linformer, Performer
# - **思想**: 通过数学变换（如低秩近似）来避免直接计算巨大的 (seq_len, seq_len) 注意力矩阵，
#   从而将复杂度从 O(seq_len²) 降低到 O(seq_len)。

# **C. 递归 (Recurrence)**
# - **代表模型**: Transformer-XL, Compressive Transformer
# - **思想**: 将长序列切分成段落（segment），然后逐段处理。在处理当前段落时，
#   通过一种循环机制来复用前一个段落的隐藏状态，从而在段落之间传递信息。

print("--- 1. Long sequence handling requires efficient attention variants. ---")
print("-"*30)

# --- 2. 内存与性能优化 ---

# **A. 梯度检查点 (Gradient Checkpointing / Activation Checkpointing)**
# - **问题**: 在前向传播时，PyTorch会存储所有的中间激活值（activations），以便在反向传播时计算梯度。
#   对于非常深或非常宽的模型，这些激活值会占用巨大的显存。
# - **解决方案**: “用计算换内存”。在前向传播时，不保存中间激活值。在反向传播需要它们时，
#   再重新从前一个检查点开始，进行一次小范围的前向传播来重新计算它们。
# - **效果**: 可以极大地减少显存占用（例如，减少到原来的平方根），代价是增加了约20-30%的训练时间。
# - **实现**: `torch.utils.checkpoint.checkpoint`

#   ```python
#   # 原始代码
#   def forward(self, x):
#       for layer in self.layers:
#           x = layer(x)
#       return x
#
#   # 使用checkpoint的代码
#   def forward(self, x):
#       for layer in self.layers:
#           # checkpoint会负责在反向传播时重新计算layer的前向传播
#           x = checkpoint(layer, x)
#       return x
#   ```

# **B. 混合精度训练 (Mixed-Precision Training)**
# - **问题**: 默认情况下，所有计算都使用32位浮点数（FP32）。
# - **解决方案**: 使用16位浮点数（FP16）和32位浮点数的混合。在前向和反向传播时使用FP16进行计算，
#   同时保留一个FP32的主权重副本用于更新，以保持数值稳定性。
# - **效果**: 在NVIDIA的Tensor Core GPU上，训练速度可以提升2-3倍，显存占用减少近一半。
# - **实现**: `torch.cuda.amp` (Automatic Mixed Precision)

#   ```python
#   from torch.cuda.amp import GradScaler, autocast
#
#   scaler = GradScaler()
#
#   for data in dataloader:
#       optimizer.zero_grad()
#       with autocast(): # 在autocast上下文中，运算会自动切换到FP16
#           loss = model(data)
#       
#       # scaler会缩放损失，以防止FP16的梯度下溢
#       scaler.scale(loss).backward()
#       scaler.step(optimizer)
#       scaler.update()
#   ```

print("--- 2. Use checkpointing for memory and mixed-precision for speed. ---")
print("-"*30)

# --- 3. 并行训练 (Parallel Training) ---
# **问题**: 当模型巨大到单个GPU无法容纳时，需要将其拆分到多个GPU上。

# **A. 数据并行 (Data Parallelism)** (最常用)
# - **思想**: 在每个GPU上都放一个完整的模型副本。将一个大的batch数据切分后，分发给每个GPU独立进行前向/反向传播。最后同步所有GPU上的梯度，并更新模型。
# - **实现**: `nn.DataParallel` (简单但不推荐), `nn.DistributedDataParallel` (更快、更标准)。

# **B. 流水线并行 (Pipeline Parallelism)**
# - **思想**: 将模型的不同**层**分布到不同的GPU上。例如，GPU-0负责1-8层，GPU-1负责9-16层。
#   数据像流水线一样依次通过所有GPU。
# - **应用**: 适用于层数非常多的模型。

# **C. 张量并行 (Tensor Parallelism)**
# - **思想**: 将模型中单个巨大的**权重矩阵**（例如，一个大的nn.Linear层）本身进行切分，分布到不同的GPU上，然后在不同的GPU上并行地执行矩阵运算。
# - **应用**: 适用于模型宽度非常大（权重矩阵巨大）的情况。Megatron-LM等巨型模型的核心技术。

print("--- 3. Advanced parallelism splits the model itself across GPUs. ---")

# 总结:
# 1. **高效Transformer**: 通过稀疏化、线性化等方法解决自注意力的平方复杂度问题，是处理长序列的关键。
# 2. **梯度检查点**: 一种用时间换空间（显存）的实用技术。
# 3. **混合精度训练**: 在现代GPU上加速训练、节省显存的标准实践。
# 4. **模型并行**: 当模型大到单卡无法容纳时，通过流水线或张量并行等技术将其拆分到多卡上，是训练千亿级参数模型的基础。
