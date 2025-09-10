
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

# --- 前言 ---
# 自注意力机制本身并不处理序列的顺序信息。它将输入看作一个无序的“集合”，
# 这意味着“I am learning NLP”和“NLP am I learning”在自注意力看来是一样的。
# 为了让模型能够利用序列的顺序，我们需要向输入中注入关于token位置的信息。
# 这就是位置编码 (Positional Encoding) 的作用。

# --- 1. 核心思想 ---
# 位置编码的思想是创建一个与词嵌入维度相同的向量，这个向量能够表示token在序列中的位置。
# 然后，将这个位置编码向量与原始的词嵌入向量相加。
#
#   Final_Embedding = Word_Embedding + Positional_Encoding
#
# 这样，每个token的最终表示就既包含了其语义信息，也包含了其位置信息。

# --- 2. 正弦/余弦位置编码 (Sinusoidal Positional Encoding) ---
# 这是原版Transformer论文中提出的方法。它不是通过学习得到的，而是使用一个固定的数学公式来生成。
#
# 公式:
# PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
#
# - `pos`: token在序列中的位置 (0, 1, 2, ...)
# - `i`: 嵌入向量中的维度索引 (0, 1, 2, ...)
# - `d_model`: 嵌入向量的总维度
#
# **为什么这个公式有效？**
# 1. 它为每个位置生成了独一无二的编码。
# 2. 不同长度的序列，其位置编码的计算方式是一致的。
# 3. 模型可以很容易地学习到相对位置关系，因为对于任意固定的偏移k，PE(pos+k)可以由PE(pos)线性表示。

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 嵌入向量的维度
            max_len: 预先计算编码的最大序列长度
        """
        super().__init__()

        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建位置张量 (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母部分，即 10000^(2i / d_model)
        # 使用log空间可以避免数值溢出
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算偶数维度的sin编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算奇数维度的cos编码
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个批次维度，并注册为buffer
        # buffer是不被视为模型参数，但又希望随模型保存的张量
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: 词嵌入向量, shape (batch_size, seq_len, d_model)
        """
        # 将位置编码加到输入上
        # self.pe[:, :x.size(1)] 截取所需长度的位置编码
        x = x + self.pe[:, :x.size(1)]
        return x

# --- 3. 可视化位置编码 ---

print("--- Visualizing Positional Encodings ---")

d_model = 128
max_len = 100

pe_module = PositionalEncoding(d_model, max_len)
positional_encodings = pe_module.pe.squeeze(0).numpy()

plt.figure(figsize=(10, 8))
cax = plt.matshow(positional_encodings, cmap='viridis')
plt.gcf().colorbar(cax)
plt.xlabel('Embedding Dimension')
plt.ylabel('Position in Sequence')
plt.title('Sinusoidal Positional Encoding Matrix')
plt.show()

print("Each row is a unique positional vector. The columns show the sinusoidal patterns.")
print("-"*30)

# --- 4. 应用位置编码 ---

print("--- Applying Positional Encoding ---")

# 1. 创建一个词嵌入层和一个位置编码层
vocab_size = 1000
embedding_dim = 128

word_embedding = nn.Embedding(vocab_size, embedding_dim)
pos_encoder = PositionalEncoding(embedding_dim)

# 2. 准备输入
# (batch_size, seq_len)
input_seq_indices = torch.randint(0, vocab_size, (2, 50)) # 2个句子，每个长50

# 3. 应用
# a. 获取词嵌入
embedded_words = word_embedding(input_seq_indices)
print(f"Shape after word embedding: {embedded_words.shape}")

# b. 添加位置编码
final_embedding = pos_encoder(embedded_words)
print(f"Shape after adding positional encoding: {final_embedding.shape}")

# 这个 final_embedding 就是最终送入Transformer编码器第一层的输入。

# --- 5. 可学习的位置编码 (Learnable Positional Encoding) ---
# - **思想**: 将位置编码也看作是模型的可学习参数，而不是使用固定的公式。
# - **实现**: 通常通过再创建一个 `nn.Embedding` 层来实现。
#   `pos_embedding = nn.Embedding(max_len, d_model)`
#   `positions = torch.arange(0, seq_len)`
#   `final_embedding = embedded_words + pos_embedding(positions)`
# - **应用**: BERT, GPT-2 等模型采用了这种方法。它的灵活性更高，但需要模型从数据中自己学习到位置关系。

# 总结:
# 1. 自注意力机制本身不感知顺序，必须通过位置编码来注入位置信息。
# 2. 经典的方法是使用固定的**正弦/余弦位置编码**，它为每个位置生成唯一的编码，并且模型能轻易学习到相对位置关系。
# 3. 另一种方法是**可学习的位置编码**，它将位置信息作为模型参数的一部分进行训练。
# 4. 位置编码向量最终与词嵌入向量**相加**，形成Transformer编码器的最终输入。
