# --- 前言 ---
# Hugging Face `transformers` 库是NLP领域事实上的标准工具库。
# 它极大地简化了下载、加载和使用预训练模型（如BERT, GPT-2等）的流程。
# 本脚本将介绍该库的三个核心组件：`pipeline`, `AutoTokenizer`, 和 `AutoModel`。
#
# 安装所需库: pip install transformers torch

from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# --- 1. `pipeline`: 最简单的开箱即用工具 ---
# `pipeline` 是一个高级封装，它将模型、分词器和前后处理逻辑全部打包好，
# 让你只需两行代码就能完成一个特定的任务。

print("--- 1. Using `pipeline` for zero-shot inference ---")

# 1. 创建一个情感分析的pipeline
# 第一次运行时，它会自动下载并缓存所需的模型和分词器
classifier = pipeline("sentiment-analysis")

# 2. 使用pipeline进行预测
results = classifier([
    "I love this library, it is amazing!", 
    "I hate waiting, this is so slow."
])

print("Pipeline results:")
for result in results:
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")

# pipeline支持多种任务，例如：
# - "text-generation" (文本生成)
# - "fill-mask" (掩码填充，即MLM)
# - "ner" (命名实体识别)
# - "question-answering" (问答)
# - "summarization" (摘要)
print("---"*30)

# --- 2. `AutoTokenizer`: 智能分词器 ---
# 每个预训练模型都有一个与之对应的、用相同方式训练的分词器。
# `AutoTokenizer` 可以根据你指定的模型名称（checkpoint），自动从Hugging Face Hub上加载正确的分词器。

print("--- 2. Using `AutoTokenizer` ---")

# 指定一个模型名称
model_name = "bert-base-uncased" # 一个基础的、不区分大小写的BERT模型

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用分词器
raw_texts = ["Hello, this is a test sentence for BERT."]
encoded_input = tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt")

print(f"Tokenizer for: {model_name}")
print("Encoded input:")
# `input_ids`: 数值化的token索引
# `token_type_ids`: 用于区分不同句子（在BERT的NSP任务中）
# `attention_mask`: 用于区分真实的token和填充的[PAD] token
for key, value in encoded_input.items():
    print(f"  {key}: {value}")

# 我们可以将ID转换回token
decoded_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print("\nDecoded tokens:")
print(decoded_tokens)
print("---"*30)

# --- 3. `AutoModel`: 智能模型加载器 ---
# 与`AutoTokenizer`类似，`AutoModel` 可以根据指定的模型名称，自动加载正确的模型架构和预训练权重。

print("--- 3. Using `AutoModel` ---")

# 加载预训练模型
# 这会下载模型权重（通常几百MB），所以第一次会比较慢
model = AutoModel.from_pretrained(model_name)

# `AutoModel` 加载的是基础的、没有特定任务“头”的模型。
# 它的输出是最后一层的隐藏状态 (last_hidden_state)。
print(f"Loaded model: {model.__class__.__name__}")

# 将分词后的输入喂给模型
with torch.no_grad():
    outputs = model(**encoded_input)

last_hidden_states = outputs.last_hidden_state
print(f"Shape of last hidden state: {last_hidden_states.shape}")
print("(batch_size, sequence_length, hidden_size)")

# 如果你想加载一个带有特定任务“头”的模型，可以使用其他`AutoModel`变体，例如：
# - `AutoModelForSequenceClassification` (用于文本分类)
# - `AutoModelForTokenClassification` (用于命名实体识别)
# - `AutoModelForQuestionAnswering` (用于问答)
# - `AutoModelWithLMHead` (用于语言模型任务，如MLM或文本生成)

# 总结:
# 1. **`pipeline`**: 用于快速、无代码的推理，非常适合演示和简单的应用。
# 2. **`AutoTokenizer`**: 加载与模型匹配的分词器，负责将文本转换为模型可接受的输入格式。
# 3. **`AutoModel`**: 加载模型的架构和权重。使用不同的`AutoModelFor...`变体可以加载带有特定任务头的模型，这是进行微调的基础。
# 这三个组件是使用Hugging Face `transformers`库的基石。
