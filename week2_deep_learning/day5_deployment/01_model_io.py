
import torch
import torch.nn as nn
import torch.optim as optim

# --- 前言 --- 
# 训练一个模型可能需要数小时甚至数天。我们不希望训练完成后模型就消失了，
# 或者在训练中途因意外中断而前功尽弃。因此，学会如何正确地保存和加载模型至关重要。

# --- 1. 创建一个简单的模型和优化器 ---
# 我们将使用这个模型作为示例
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- 1. A Simple Model and Optimizer ---")
print(model)
print("-"*30)

# --- 2. 方法一: 保存和加载模型权重 (State Dictionary) ---
# 这是最推荐、最灵活的方法。`state_dict` 是一个Python字典，它将模型的每一层映射到其可学习的参数（权重和偏置）。
# 只保存模型的参数，而不是整个模型的结构，使得代码更可移植、更不容易出错。

print("--- 2. Saving and Loading State Dict ---")

# **保存 State Dict**
# 定义文件路径
PATH_STATE_DICT = "simple_model_state_dict.pth"
# 使用 torch.save() 保存
torch.save(model.state_dict(), PATH_STATE_DICT)
print(f"Model state_dict saved to {PATH_STATE_DICT}")

# **加载 State Dict**
# 1. 首先，你需要创建一个与你保存权重时结构完全相同的模型实例。
model_to_load = SimpleModel()

# 2. 使用 .load_state_dict() 加载参数
model_to_load.load_state_dict(torch.load(PATH_STATE_DICT))

# 3. 务必调用 model.eval() 来将模型设置为评估模式，特别是如果模型中有Dropout或BatchNorm层。
model_to_load.eval()
print("Model loaded successfully from state_dict.")
print("-"*30)

# --- 3. 方法二: 保存和加载整个模型 ---
# 这种方法将整个模型对象（包括其结构和参数）使用Python的pickle进行序列化。
# - **优点**: 代码简单。
# - **缺点**: 序列化的数据与特定的类和目录结构绑定，如果你的项目代码发生变化，
#   或者你想在另一个项目中使用这个模型，可能会导致无法加载。

print("--- 3. Saving and Loading Entire Model ---")

# **保存整个模型**
PATH_FULL_MODEL = "simple_full_model.pth"
torch.save(model, PATH_FULL_MODEL)
print(f"Entire model saved to {PATH_FULL_MODEL}")

# **加载整个模型**
# 你不需要先创建模型实例
loaded_full_model = torch.load(PATH_FULL_MODEL)
loaded_full_model.eval()
print("Entire model loaded successfully.")
print("-"*30)

# --- 4. 保存和加载训练检查点 (Checkpoint) ---
# 在长时间的训练中，我们不仅想保存模型，还想保存训练的状态，以便能从中断的地方恢复训练。
# 这就需要保存一个“检查点”，它通常是一个字典，包含：
# - 当前的周期数 (epoch)
# - 模型的 state_dict
# - 优化器的 state_dict
# - 当前的损失值等

print("--- 4. Saving and Loading Checkpoints for Resuming Training ---")

# 假设我们训练到了第50个周期
epoch = 50
current_loss = 0.5
PATH_CHECKPOINT = "training_checkpoint.pth"

# **保存检查点**
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
}
torch.save(checkpoint, PATH_CHECKPOINT)
print(f"Checkpoint saved to {PATH_CHECKPOINT}")

# **加载检查点以恢复训练**
# 1. 创建模型和优化器实例
model_to_resume = SimpleModel()
optimizer_to_resume = optim.Adam(model_to_resume.parameters(), lr=0.001)

# 2. 加载检查点字典
checkpoint_loaded = torch.load(PATH_CHECKPOINT)

# 3. 分别恢复模型和优化器的状态
model_to_resume.load_state_dict(checkpoint_loaded['model_state_dict'])
optimizer_to_resume.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
start_epoch = checkpoint_loaded['epoch']
last_loss = checkpoint_loaded['loss']

# 4. 将模型设置为训练模式，然后就可以从 start_epoch + 1 开始继续训练了
model_to_resume.train()
print("Checkpoint loaded. Training can be resumed from epoch", start_epoch + 1)

# 总结:
# 1. **首选方法**: 使用 `state_dict` 来保存和加载模型权重，这是最安全、最灵活的方式。
# 2. **完整模型**: 简单方便，但可移植性差。
# 3. **检查点**: 用于需要恢复训练的场景，务必保存优化器状态和周期数等关键信息。
