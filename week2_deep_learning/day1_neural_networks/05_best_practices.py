
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# --- 前言 --- 
# 本脚本总结了一些在设计、训练和调试神经网络时被广泛遵循的最佳实践和实用技巧。
# 遵循这些原则可以帮助你更快地构建出性能更好、更稳定的模型。

# --- 1. 网络设计原则 (Network Design) ---

# **原则1: 从简单开始，逐步增加复杂度**
# - 不要一开始就构建一个巨大而复杂的网络。从一个简单的模型（例如1-2个隐藏层）开始，
#   确保整个训练流程（数据加载、训练、评估）是通畅的。如果简单模型能工作，再逐步增加其深度或宽度。

# **原则2: 使用标准的层结构（“漏斗”结构）**
# - 一个常见且有效的结构是让网络层的大小逐渐减小，形成一个“漏斗”或“金字塔”形状。
#   例如：Input -> 256 units -> 128 units -> 64 units -> Output
#   这有助于网络从原始输入中提取广泛的模式，并逐渐将其压缩为更抽象、更具体的特征。

# **原则3: 在全连接层之间使用批标准化 (Batch Normalization)**
# - `nn.BatchNorm1d` (用于全连接层) 是一个极其有用的层。
# - **作用**: 它对每一批(batch)的数据在网络层之间进行重新标准化，使其均值为0，方差为1。
# - **优点**:
#   1. 加速模型收-敛。
#   2. 允许使用更高的学习率。
#   3. 提供轻微的正则化效果。
#   4. 降低了对参数初始化的敏感度。

class ModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64), # 在线性层之后，激活函数之前使用
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(x)

# --- 2. 参数初始化 (Parameter Initialization) ---

# **原则: 优先使用PyTorch的默认初始化**
# - PyTorch的 `nn.Linear` 和 `nn.Conv2d` 等层使用了经过理论证明的、非常有效的默认初始化方法（Kaiming He 和 Xavier 初始化）。
# - 对于ReLU激活函数，Kaiming初始化是最佳实践。对于tanh，Xavier初始化更合适。
# - 在绝大多数情况下，你**不需要**手动修改参数初始化。

# 如果确实需要自定义初始化（例如，复现论文或进行实验）：
def weights_init(m):
    if isinstance(m, nn.Linear):
        # 使用Xavier均匀分布进行初始化
        nn.init.xavier_uniform_(m.weight)
        # 将偏置初始化为0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# model = MyModel()
# model.apply(weights_init) # .apply()会递归地将函数应用到所有子模块

# --- 3. 调试技巧 (Debugging) ---

# **技巧1: 过拟合一小批数据 (Overfit a small batch)**
# - 这是验证模型是否具有学习能力的黄金法则。
# - **步骤**: 
#   1. 从你的数据集中取出一小批数据（例如32个样本）。
#   2. 让你的模型在这个小批次上反复训练几百个周期。
#   3. 观察损失函数是否能下降到一个非常小的值（接近0）。
# - **结果**: 
#   - 如果损失能降到接近0，说明你的模型结构、前向传播和反向传播逻辑是正确的，它有能力学习。
#   - 如果损失不下降或来回震荡，说明你的模型、学习率或数据处理流程中存在严重问题。

# **技巧2: 监控梯度范数 (Gradient Norm)**
# - 在训练过程中，监控梯度的范数（大小）可以帮助诊断梯度消失或梯度爆炸问题。
#   `torch.nn.utils.clip_grad_norm_` 是一个非常有用的工具，它可以在梯度爆炸时将其“裁剪”到一个合理的范围内。
# optimizer.step()
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# --- 4. 可视化工具 (Visualization) ---

# **工具: TensorBoard**
# - TensorBoard 是一个强大的可视化工具，可以用来跟踪实验的几乎所有方面。
# - PyTorch通过 `torch.utils.tensorboard.SummaryWriter` 与其集成。

# **用法**: 
# 1. 创建一个 writer 对象
# writer = SummaryWriter('runs/my_experiment_1') # 日志会保存在这个文件夹

# 2. 在训练循环中，记录你关心的指标
# for epoch in range(num_epochs):
#     # ... training ...
#     loss = ...
#     accuracy = ...
#     writer.add_scalar('Loss/train', loss, epoch) # 记录训练损失
#     writer.add_scalar('Accuracy/train', accuracy, epoch) # 记录训练准确率
#     
#     # 记录模型参数的分布
#     for name, param in model.named_parameters():
#         writer.add_histogram(name, param, epoch)

# 3. 在终端中运行 `tensorboard --logdir=runs` 来启动TensorBoard服务。

# 总结:
# 1. **设计**: 从简到繁，使用BatchNorm。
# 2. **初始化**: 相信PyTorch的默认设置。
# 3. **调试**: 首先尝试在一个小批次上过拟合。
# 4. **可视化**: 使用TensorBoard来跟踪和比较你的实验，这是进行系统性研究和调优的基础。
