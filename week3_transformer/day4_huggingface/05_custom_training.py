
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm # 用于显示漂亮的进度条

# --- 前言 --- 
# `Trainer` API非常方便，但它将许多细节抽象掉了。
# 当你需要实现一些`Trainer`不支持的复杂逻辑时，或者当你就是想深入理解训练过程的每一个步骤时，
# 就需要编写自己的训练循环。本脚本将展示如何用标准的PyTorch流程来微调一个Hugging Face模型。

# --- 1. 准备工作 (与之前类似) --- 
print("---" + " 1. Preparing Data, Model, and Tokenizer ---")

# 加载数据和分词器
dataset = load_dataset("imdb").shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 应用预处理并设置格式
tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 只取一小部分用于演示
small_train_dataset = tokenized_dataset["train"].select(range(1000))
small_eval_dataset = tokenized_dataset["test"].select(range(1000))

# --- 2. 创建原生PyTorch DataLoader --- 
# 这是与`Trainer`不同的第一步：我们自己创建DataLoader。
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

print("DataLoaders and Model are ready.")
print("---" + "-" * 30)

# --- 3. 创建优化器和学习率调度器 --- 
print("---" + " 2. Creating Optimizer and LR Scheduler ---")

# AdamW是Adam优化器的一个变体，它能更好地处理权重衰减
optimizer = AdamW(model.parameters(), lr=5e-5)

# 定义训练参数
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# `get_scheduler` 是一个方便的函数，用于创建带有预热的调度器
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

# 将模型移动到正确的设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

print(f"Training on device: {device}")
print("---" + "-" * 30)

# --- 4. 自定义训练与评估循环 --- 
print("---" + " 3. Custom Training and Evaluation Loop ---")

# 使用tqdm来可视化训练进度
progress_bar = tqdm(range(num_training_steps))

# **训练循环**
model.train() # 设置为训练模式
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 1. 将批次数据移动到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 2. 前向传播
        outputs = model(**batch)
        
        # 3. 计算损失
        # 模型会自动返回损失，因为我们提供了`labels`参数
        loss = outputs.loss
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 更新权重和学习率
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)

# **评估循环**
metric = evaluate.load("accuracy")
model.eval() # 设置为评估模式
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): # 在评估时不计算梯度
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

final_metric = metric.compute()
print(f"\nFinal evaluation accuracy: {final_metric}")

# 总结:
# 1. 自定义训练循环给予你最大程度的灵活性和控制力。
# 2. 核心流程与标准的PyTorch训练循环完全一致：
#    DataLoader -> for loop -> move to device -> forward pass -> loss -> backward -> step。
# 3. Hugging Face的`transformers`库与PyTorch生态无缝集成，
#    你可以轻松地混合使用来自两个库的组件（如`AdamW`, `get_scheduler`）。
# 4. 当你需要实现`Trainer`不支持的复杂功能时，或者想要完全理解底层发生了什么时，
#    编写自定义训练循环是一项必备技能。

