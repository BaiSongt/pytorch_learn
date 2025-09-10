
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# --- 前言 --- 
# 神经网络只能处理数字，不能直接处理原始文本。
# 因此，在进行任何NLP任务之前，我们必须将文本数据转换成神经网络可以理解的数值格式。
# 这个过程通常包括：分词、构建词汇表、数值化和填充。

# --- 1. 原始文本数据 ---
# 假设我们有以下几句评论作为我们的数据集
raw_texts = [
    "I love PyTorch, it is amazing!",
    "PyTorch is the best deep learning framework.",
    "I am learning NLP with PyTorch.",
    "NLP is fun and amazing."
]

print("--- 1. Raw Text Data ---")
print(raw_texts)
print("-"*30)

# --- 2. 分词 (Tokenization) ---
# 分词是将文本字符串分解成一个个独立的单元（称为“token”）的过程。
# 最简单的分词方法是按空格和标点符号切分。
# 在实际应用中，通常会使用更成熟的分词库，如 NLTK, spaCy, or Hugging Face Tokenizers。

# 简单的分词实现
tokenized_texts = [text.lower().replace('.', '').replace('!', '').replace(',', '').split()
                   for text in raw_texts]

print("--- 2. Tokenization ---")
print(tokenized_texts)
print("-"*30)

# --- 3. 构建词汇表 (Building a Vocabulary) ---
# 词汇表是一个映射，它将每个唯一的token与一个整数索引相关联。
# 我们需要定义一些特殊的token：
# - `<pad>` (padding token): 用于将短序列填充到与长序列相同的长度。
# - `<unk>` (unknown token): 用于表示在训练时未见过、词汇表中不存在的词。

# 统计所有词的频率
word_counts = Counter(token for text in tokenized_texts for token in text)

# 创建词汇表，只包含出现次数超过min_freq的词
min_freq = 1
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
vocab = [word for word in vocab if word_counts[word] >= min_freq]

# 添加特殊token到词汇表的最前面
vocab = ['<pad>', '<unk>'] + vocab

# 创建 word -> index 的映射字典
word_to_idx = {word: i for i, word in enumerate(vocab)}

print("--- 3. Building Vocabulary ---")
print(f"Vocabulary size: {len(vocab)}")
print(f"First 10 words in vocab: {vocab[:10]}")
print(f"Mapping for 'pytorch': {word_to_idx['pytorch']}")
print("-"*30)

# --- 4. 数值化 (Numericalization) ---
# 使用构建好的词汇表，将每个分词后的文本序列转换为整数序列。

numericalized_texts = []
for tokens in tokenized_texts:
    numericalized_texts.append([word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens])

print("--- 4. Numericalization ---")
print("Original: ", tokenized_texts[0])
print("Numericalized: ", numericalized_texts[0])
print("-"*30)

# --- 5. 填充 (Padding) ---
# 为了将多个序列打包成一个批次(batch)进行高效计算，它们必须具有相同的长度。
# 填充是在较短的序列末尾（或开头）添加特殊的`<pad>` token，直到所有序列长度一致。

# 首先，将数值化后的列表转换为Tensor
sequences_as_tensors = [torch.tensor(seq) for seq in numericalized_texts]

# `pad_sequence` 是一个方便的工具
# - `batch_first=True`: 使输出的形状为 (batch_size, sequence_length)，这是RNN层通常接受的格式。
# - `padding_value`: 用于填充的值，我们使用`<pad>` token的索引。
padded_sequences = pad_sequence(
    sequences_as_tensors, 
    batch_first=True, 
    padding_value=word_to_idx['<pad>']
)

print("--- 5. Padding ---")
print("Padded sequences tensor:")
print(padded_sequences)
print(f"Shape of the padded tensor: {padded_sequences.shape}")

# 总结:
# 文本预处理是将原始字符串转换为模型可以处理的、整齐的数值张量的过程。
# 1. **分词**: 文本 -> token列表。
# 2. **构建词汇表**: token -> 整数索引 的映射。
# 3. **数值化**: token列表 -> 整数索引列表。
# 4. **填充**: 使同一批次中的所有整数序列具有相同的长度。
# 这些步骤是几乎所有NLP任务的起点。
