
import torch
import torch.nn as nn
import math

# --- 前言 ---
# 自注意力 (Self-Attention) 是注意力机制的一种特殊形式，它的Query, Key, 和 Value都来自于同一个输入序列。
# 这使得模型在处理序列中的某个token时，能够计算出序列中所有其他token对当前token的“重要性”或“相关性”，
# 从而生成一个对上下文感知的、更丰富的表示。

# --- 1. 准备输入数据 ---
# 假设我们有一个批次，序列长度为4，每个token用一个6维的嵌入向量表示。
# 这可以看作是一个句子经过了 nn.Embedding 层的输出。

seq_len = 4
embedding_dim = 6

# (batch_size, seq_len, embedding_dim)
input_sequence = torch.randn(1, seq_len, embedding_dim)

print("--- 1. Input Sequence ---")
print(f"Shape of input sequence: {input_sequence.shape}")
print("-"*30)

# --- 2. 从输入生成 Q, K, V ---
# 我们通过三个独立的全连接层（线性变换）来将输入序列分别映射到Query, Key, 和 Value 空间。
# 这三个线性层拥有各自独立的、可学习的权重。

# d_q, d_k, d_v 是Q, K, V向量的维度。在标准的Transformer中，它们通常是相等的。
d_k = 8 # Key和Query的维度
d_v = 8 # Value的维度

# 定义三个线性层
W_q = nn.Linear(embedding_dim, d_k, bias=False)
W_k = nn.Linear(embedding_dim, d_k, bias=False)
W_v = nn.Linear(embedding_dim, d_v, bias=False)

# 进行线性投射
query = W_q(input_sequence) # (1, 4, 6) -> (1, 4, 8)
key = W_k(input_sequence)   # (1, 4, 6) -> (1, 4, 8)
value = W_v(input_sequence) # (1, 4, 6) -> (1, 4, 8)

print("--- 2. Projecting Input to Q, K, V ---")
print(f"Shape of Q: {query.shape}")
print(f"Shape of K: {key.shape}")
print(f"Shape of V: {value.shape}")
print("-"*30)

# --- 3. 执行缩放点积注意力 ---
# 现在我们有了Q, K, V，可以应用上一脚本中学到的注意力公式了。

print("--- 3. Applying Scaled Dot-Product Attention ---")

# 计算分数
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

# 计算注意力权重
attention_weights = torch.softmax(scores, dim=-1)

# 计算输出
output = torch.matmul(attention_weights, value)

print(f"Shape of attention scores: {scores.shape}")
print(f"Shape of attention weights: {attention_weights.shape}")
print(f"Shape of final output: {output.shape}")
print("-"*30)

# --- 4. 封装成一个完整的自注意力模块 ---
# 将上述逻辑封装成一个 nn.Module，以便于复用。

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, d_k, d_v):
        super().__init__()
        self.W_q = nn.Linear(embedding_dim, d_k, bias=False)
        self.W_k = nn.Linear(embedding_dim, d_k, bias=False)
        self.W_v = nn.Linear(embedding_dim, d_v, bias=False)
        self.d_k = d_k

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, embedding_dim)
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # 计算注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

# 使用模块进行计算
self_attention_module = SelfAttention(embedding_dim, d_k, d_v)
output_module, weights_module = self_attention_module(input_sequence)

print("--- 4. Encapsulated Self-Attention Module ---")
print(f"Output shape from module: {output_module.shape}")
print(f"Weights shape from module: {weights_module.shape}")

# 总结:
# 1. 自注意力的核心是“自己和自己做注意力计算”，即 Q, K, V 均来自同一个输入序列。
# 2. 这是通过三个独立的线性层 (W_q, W_k, W_v) 将输入序列投射到三个不同的空间来实现的。
# 3. 自注意力机制使得模型能够动态地、基于上下文地为序列中的每个元素生成一个丰富的表示，
#    这个表示考虑了整个序列的信息。
# 4. 这种机制是并行的，计算一个元素的输出时，可以同时计算它与所有其他元素的关联，
#    这是Transformer相比于RNN的一个巨大优势。
