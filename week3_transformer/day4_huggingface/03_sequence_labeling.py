
import torch
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# --- 前言 ---
# 序列标注任务的目标是为输入序列中的每个token分配一个类别标签。
# 命名实体识别 (NER) 是其典型应用，用于识别文本中的人名、地名、组织名等。
# 本脚本将展示如何使用Hugging Face微调一个模型来完成NER任务。
#
# 安装所需库: pip install transformers datasets evaluate seqeval

# --- 1. 加载和预处理数据 ---
print("---" + " 1. Loading and Preprocessing Data " + "---")

# 加载CoNLL-2003数据集，这是一个标准的NER基准
dataset = load_dataset("conll2003")
ner_feature = dataset["train"].features["ner_tags"]
label_names = ner_feature.feature.names

# 加载分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **核心：对齐标签与子词 (Aligning Labels with Subwords)**
# 分词器可能会将一个词拆分成多个子词（例如，“Washington” -> “Washing”, “##ton”）。
# 我们需要确保标签能与这些子词正确对齐。
# 一个常见的策略是：只为每个原始单词的第一个子词分配标签，而将其余的子词标签设为一个特殊值（如-100），
# 这样它们在计算损失时就会被自动忽略。

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: # 对于特殊token如[CLS], [SEP]
                label_ids.append(-100)
            elif word_idx != previous_word_idx: # 单词的第一个子词
                label_ids.append(label[word_idx])
            else: # 同一个单词的后续子词
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 应用预处理函数
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

print("Data tokenized and labels aligned.")
print("---"+"-"*30)

# --- 2. 加载模型和定义评估指标 ---
print("---" + " 2. Loading Model and Defining Metrics " + "---")

# 加载带有token分类“头”的预训练模型
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_names))

# 定义评估指标
# `seqeval`是评估序列标注任务（如NER, POS）的标准库
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 移除被忽略的标签 (-100)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

print("Model and metrics ready.")
print("---"+"-"*30)

# --- 3. 使用 `Trainer` API 进行训练 ---
print("---" + " 3. Training with the `Trainer` API " + "---")

training_args = TrainingArguments(
    output_dir="./results_ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1, # 为快速演示，只训练1个epoch
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training for NER... (This may take a while on CPU)")
# trainer.train() # 取消注释以开始训练

print("\nConceptual training complete. Uncomment `trainer.train()` to run for real.")

# 总结:
# 1. 序列标注（如NER）是为序列中的每个token分配标签的任务。
# 2. 微调流程与文本分类类似，但数据预处理更复杂，核心是**对齐子词与标签**。
# 3. `AutoModelForTokenClassification` 是用于此类任务的模型类。
# 4. 评估时，应使用`seqeval`等专用库来计算实体级别的F1分数，而不是简单的token准确率。

