
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate # Hugging Face提供的评估指标库

# --- 前言 --- 
# 本脚本将展示使用Hugging Face `transformers`库微调一个预训练模型
# 来完成文本分类任务的完整流程。我们将使用功能强大的`Trainer` API，
# 它为我们处理了大部分训练循环的样板代码。
#
# 安装所需库: pip install transformers datasets evaluate scikit-learn

# --- 1. 加载和预处理数据 ---
print("---" + "1. Loading and Preprocessing Data" + "---")

# 加载IMDB电影评论数据集
# `datasets`库会自动下载并缓存数据
dataset = load_dataset("imdb")

# 为了节省时间，我们只使用一小部分数据进行演示
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

# 加载分词器
# 我们选择distilbert-base-uncased，这是一个更小、更快的BERT变体，非常适合微调
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建一个预处理函数，用于对文本进行分词
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# 使用.map()方法将预处理函数应用到整个数据集
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

print("Data tokenized.")
print("-" * 30)

# --- 2. 加载模型和定义评估指标 ---
print("---" + "2. Loading Model and Defining Metrics" + "---")

# 加载带有序列分类“头”的预训练模型
# `num_labels=2` 告诉模型我们正在进行一个二分类任务
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 定义一个函数来计算评估指标
# `evaluate`库可以轻松加载像accuracy, f1, precision, recall等标准指标
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

print("Model and metrics ready.")
print("-" * 30)

# --- 3. 使用 `Trainer` API 进行训练 ---
print("---" + "3. Training with the `Trainer` API" + "---")

# **步骤A: 定义训练参数 `TrainingArguments`**
# 这是一个包含了所有训练配置的类
training_args = TrainingArguments(
    output_dir="./results",          # 训练结果（模型、检查点等）的输出目录
    num_train_epochs=3,              # 训练的总轮数
    per_device_train_batch_size=16,  # 每个GPU/CPU上的训练批次大小
    per_device_eval_batch_size=16,   # 评估批次大小
    warmup_steps=500,                # 学习率预热的步数
    weight_decay=0.01,               # 权重衰减（L2正则化）
    logging_dir="./logs",            # 日志目录
    logging_steps=10,
    evaluation_strategy="epoch",     # 在每个epoch结束时进行评估
    save_strategy="epoch",           # 在每个epoch结束时保存模型
    load_best_model_at_end=True,     # 训练结束后加载性能最佳的模型
)

# **步骤B: 实例化 `Trainer`**
# `Trainer` 接收模型、训练参数、训练/测试数据集、分词器和评估函数
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# **步骤C: 开始训练**
# 只需一行代码！`Trainer`会处理所有事情：
# - 训练循环 (包括梯度、优化器、学习率调度)
# - 将数据移动到正确的设备 (GPU/CPU)
# - 评估循环
# - 模型保存和检查点管理
print("Starting training... (This may take a while on CPU)")
# trainer.train() # 取消注释以开始训练

# **步骤D: 评估**
# print("\nEvaluating the finetuned model...")
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")

print("\nConceptual training complete. Uncomment `trainer.train()` to run for real.")

# 总结:
# 1. `datasets`库可以轻松加载和处理各种NLP数据集。
# 2. `AutoModelForSequenceClassification` 是进行文本分类微调的标准模型类。
# 3. `Trainer` API 是Hugging Face提供的强大工具，它将训练流程高度自动化，
#    让研究人员和开发者能更专注于数据和模型本身，而不是训练循环的样板代码。
# 4. 整个流程是：加载数据 -> 分词 -> 加载模型 -> 定义训练参数 -> 创建Trainer -> 开始训练。
