
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 前言 --- 
# 知识蒸馏是一种模型压缩技术，其核心思想是训练一个小的“学生模型”来模仿一个大的、
# 性能强大的“教师模型”的行为。学生模型不仅学习真实的标签，还学习教师模型输出的“软标签”
# （即logits或概率分布），这些软标签包含了教师模型学到的类别之间的“暗知识”。

# --- 1. 准备教师和学生模型 ---
print("--- 1. Preparing Teacher and Student Models ---")

# **教师模型**: 一个更大、更强大的模型
# 假设我们使用一个已经微调好的bert-base模型
teacher_model_name = "bert-base-uncased"
# 在真实场景中，这个模型应该是在你的任务上微调过的
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)

# **学生模型**: 一个更小、更快的模型
# DistilBERT是BERT的蒸馏版本，是作为学生模型的绝佳选择
student_model_name = "distilbert-base-uncased"
student_model = AutoModelForSequenceClassification.from_pretrained(student_model_name, num_labels=2)

# 在蒸馏时，教师模型始终处于评估模式
teacher_model.eval()

print(f"Teacher: {teacher_model_name}")
print(f"Student: {student_model_name}")
print("-"*30)

# --- 2. 蒸馏损失函数 (Distillation Loss) ---
# 蒸馏的总损失函数通常是两部分的加权和：
# Total Loss = α * L_hard + (1 - α) * L_soft
#
# - **L_hard**: 硬损失。即学生模型与真实标签之间的标准损失（例如，交叉熵损失）。
#   这保证了学生模型在学习真实的任务。
# - **L_soft**: 软损失。即学生模型的输出与教师模型输出之间的损失。
#   这使得学生模型能够模仿教师模型的“思考方式”。

# 为了让教师模型的输出更“软”，包含更多类别间的信息，我们使用一个“温度(temperature)”参数T来平滑它。
# q_i = exp(z_i / T) / Σ_j exp(z_j / T)
# 当T > 1时，概率分布会变得更平滑；当T -> ∞时，趋向于均匀分布。

class DistillationLoss(nn.Module):
    def __init__(self, student_loss_fn, distillation_loss_fn, alpha, temperature):
        super().__init__()
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        # 1. 计算硬损失 (学生 vs. 真实标签)
        hard_loss = self.student_loss_fn(student_logits, labels)
        
        # 2. 计算软损失 (学生 vs. 教师)
        # 使用温度T来平滑教师和学生的logits
        soft_student_logits = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_logits = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 计算KL散度损失
        # `kl_div`的输入log_probs和目标probs的单位需要匹配
        distillation_loss = self.distillation_loss_fn(soft_student_logits, soft_teacher_logits)
        
        # 3. 组合损失
        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * distillation_loss
        return total_loss

print("--- 2. The Distillation Loss ---")
# 实例化损失函数
# 对于多分类任务，硬损失是交叉熵，软损失是KL散度
loss_fn = DistillationLoss(
    student_loss_fn=nn.CrossEntropyLoss(),
    distillation_loss_fn=nn.KLDivLoss(reduction='batchmean'),
    alpha=0.3, # 硬损失和软损失的权重
    temperature=4.0
)
print("Distillation loss combines a hard loss (vs. labels) and a soft loss (vs. teacher). ")
print("-"*30)

# --- 3. 概念性训练流程 ---
print("--- 3. Conceptual Training Loop ---")

# `optimizer = AdamW(student_model.parameters(), lr=5e-5)`
# `student_model.train()`
# `teacher_model.eval()`
#
# `for batch in dataloader:`
# `    inputs, labels = batch`
# `    `
# `    # 1. 获取教师模型的logits (不计算梯度)`
# `    with torch.no_grad():`
# `        teacher_outputs = teacher_model(**inputs)`
# `        teacher_logits = teacher_outputs.logits`
# `    `
# `    # 2. 获取学生模型的logits`
# `    student_outputs = student_model(**inputs)`
# `    student_logits = student_outputs.logits`
# `    `
# `    # 3. 计算蒸馏损失`
# `    loss = loss_fn(student_logits, teacher_logits, labels)`
# `    `
# `    # 4. 反向传播和优化 (只更新学生模型的权重)`
# `    loss.backward()`
# `    optimizer.step()`
# `    optimizer.zero_grad()`

print("In the training loop, the student learns from both the true labels and the teacher's logits.")

# 总结:
# 1. **知识蒸馏**是一种有效的模型压缩方法，它将大模型的“知识”迁移到小模型中。
# 2. **教师模型**提供“软目标”，指导学生模型的学习方向，帮助其学习到更泛化的特征。
# 3. **总损失**是学生模型与真实标签的**硬损失**和与教师模型的**软损失**的加权和。
# 4. 在训练时，只有**学生模型**的参数被更新。
# 5. DistilBERT, TinyBERT等都是使用知识蒸馏技术从BERT压缩而来的优秀小型模型。
