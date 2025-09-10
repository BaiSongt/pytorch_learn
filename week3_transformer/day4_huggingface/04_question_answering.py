
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# --- 前言 ---
# 抽取式问答是NLP中的一个核心任务。与需要模型生成答案的“生成式问答”不同，
# 抽取式问答的任务是在给定的上下文中，找到答案所在的文本片段（span）。
# 本脚本将展示如何使用Hugging Face微调一个模型来完成此任务。
#
# 安装所需库: pip install transformers datasets evaluate

# --- 1. 加载和预处理数据 ---
print("---" + " 1. Loading and Preprocessing Data ---")

# 加载SQuAD数据集
# squad = load_dataset("squad") # 完整数据集较大
# 为快速演示，我们创建一个虚拟的、结构相同的小数据集
from datasets import Dataset, DatasetDict
dummy_data = {
    "id": ["1", "2"],
    "title": ["dummy_title", "dummy_title"],
    "context": [
        "Hugging Face is a company based in New York City. Its headquarters are in DUMBO, Brooklyn.",
        "The quick brown fox jumps over the lazy dog."
    ],
    "question": [
        "Where is Hugging Face based?",
        "What does the fox jump over?"
    ],
    "answers": [
        {"text": ["New York City"], "answer_start": [30]},
        {"text": ["the lazy dog"], "answer_start": [30]}
    ]
}
dummy_dataset = Dataset.from_dict(dummy_data)
squad = DatasetDict({"train": dummy_dataset, "validation": dummy_dataset})

# 加载分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **核心：预处理问答数据**
# 1. 将问题和上下文拼接起来，格式通常是 `[CLS] question [SEP] context [SEP]`。
# 2. 找到答案在拼接后文本中的起始和结束**token**的位置。
# 3. 处理长上下文：如果一个上下文太长，可以将其切分成多个有重叠的块。
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second", # 只截断上下文，不截断问题
        return_offsets_mapping=True, # 返回每个token在原文中的起始和结束位置
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # 找到答案覆盖的第一个和最后一个token
        idx = 0
        while sequence_ids[idx] != 1: # 1代表上下文部分
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 应用预处理函数
tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

print("Data preprocessed for Question Answering.")
print("---"+"-"*30)

# --- 2. 加载模型 ---
print("---" + " 2. Loading Model ---")

# 加载带有问答“头”的预训练模型
# 这个头的输出是start_logits和end_logits
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

print("Model ready.")
print("---"+"-"*30)

# --- 3. 使用 `Trainer` API 进行训练 ---
print("---" + " 3. Training with the `Trainer` API ---")

training_args = TrainingArguments(
    output_dir="./results_qa",
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
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
)

print("Starting training for QA... (This may take a while on CPU)")
# trainer.train() # 取消注释以开始训练

print("\nConceptual training complete. Uncomment `trainer.train()` to run for real.")

# 总结:
# 1. 抽取式问答的目标是在上下文中找到答案的起始和结束位置。
# 2. 数据预处理是关键且复杂的，需要将答案的字符级别位置映射到token级别的位置。
# 3. `AutoModelForQuestionAnswering` 是用于此类任务的模型类，它会输出每个token是起点和终点的logits。
# 4. 模型的损失函数会同时计算起点和终点位置的交叉熵损失。
# 5. 推理时，需要一个后处理步骤，从start_logits和end_logits中解码出最可能的文本片段作为最终答案。
