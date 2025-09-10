
import torch
import torch.nn.functional as F
import math

# --- 前言 ---
# 注意力机制 (Attention Mechanism) 的核心思想是，当处理一个序列时（例如，翻译一个句子），
# 输出的每一步都应该将“注意力”集中在输入序列的特定部分。
#
# Transformer 使用的“缩放点积注意力 (Scaled Dot-Product Attention)”是其中最高效、最主流的一种。
# 它涉及三个核心概念：Query, Key, 和 Value。
#
# - **Query (Q)**: 代表当前我们关注的焦点。可以理解为“我在寻找什么？”
# - **Key (K)**: 代表序列中可以被检索的信息。可以理解为“这里有什么？”
# - **Value (V)**: 代表序列中实际包含的内容。可以理解为“具体的内容是什么？”
#
# **计算过程**: 注意力机制通过计算Query和所有Key的“相似度”（通常是点积），
# 来决定给每个Key对应的Value分配多少“注意力权重”，最后对所有Value进行加权求和。

# --- 1. 准备输入数据 ---
# I am a Cat
# 假设我们有一个批次，序列长度为4，每个token用一个6维的向量表示。
# 在这个基础示例中，我们假设Q, K, V是外部给定的，并且形状相同。

seq_len = 4
embedding_dim = 6

# (batch_size, seq_len, embedding_dim)
query = torch.randn(1, seq_len, embedding_dim)
key = torch.randn(1, seq_len, embedding_dim)
value = torch.randn(1, seq_len, embedding_dim)

print("---" + "-" * 10 + " 1. Input Tensors (Query, Key, Value) ---")
print(f"Shape of Q, K, V: {query.shape}")
print("-" * 30)

# --- 2. 注意力计算的分步实现 ---
# 公式: Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

print("---" + "-" * 10 + " 2. Step-by-step Attention Calculation ---")

# **步骤 1: 计算Query和Key的点积 (Dot Product)**
# - `query` 形状: (1, 4, 6)
# - `key.transpose(-2, -1)` (K^T) 形状: (1, 6, 4)
# - `scores` (Q @ K^T) 形状: (1, 4, 4)
# `scores` 矩阵中的每个元素 scores[i, j] 表示第 i 个Query与第 j 个Key的相似度。
scores = torch.matmul(query, key.transpose(-2, -1))
print("Step 1: Dot Product Scores (Q @ K^T)")
print(f"Shape of scores: {scores.shape}")
# print(scores)

# **步骤 2: 缩放 (Scaling)**
# - **原因**: 当 embedding_dim (即 d_k) 的维度很大时，点积的结果也可能变得非常大，
#   这会将Softmax函数的输入推向梯度非常小的区域，导致梯度消失，不利于训练。
# - **方法**: 将点积分数除以 embedding_dim 的平方根。
d_k = key.shape[-1]
scaled_scores = scores / math.sqrt(d_k)
print("\nStep 2: Scaling")
print(f"Scaling factor (sqrt(d_k)): {math.sqrt(d_k):.4f}")
# print(scaled_scores)

# **步骤 3: Softmax**
# - 将缩放后的分数转换为概率分布。每一行的和为1。
# - `attention_weights[i, j]` 表示在计算第 i 个输出时，应该对第 j 个Value赋予多大的注意力。
attention_weights = F.softmax(scaled_scores, dim=-1)
print("\nStep 3: Softmax for Attention Weights")
print(f"Shape of attention_weights: {attention_weights.shape}")
# print(attention_weights)
# 验证每一行的和是否为1
# print(attention_weights.sum(dim=-1))

# **步骤 4: 对Value进行加权求和**
# - `attention_weights` 形状: (1, 4, 4)
# - `value` 形状: (1, 4, 6)
# - `output` (weights @ V) 形状: (1, 4, 6)
# `output`中的第 i 个向量，是所有Value向量根据第 i 行注意力权重的加权和。
# 它融合了整个序列的信息，但重点关注了与第 i 个Query最相关的部分。
output = torch.matmul(attention_weights, value)
print("\nStep 4: Weighted Sum of Values")
print(f"Shape of final output: {output.shape}")
# print(output)
print("-" * 30)

# --- 3. 封装成一个函数 ---

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Mask (可选): 在解码器中会用到，用于防止未来的token影响当前的预测。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 将被mask的位置设为一个极小的数

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# 使用函数进行计算
output_fn, weights_fn = scaled_dot_product_attention(query, key, value)

print("---" + "-" * 10 + " 3. Encapsulated Function ---")
print(f"Output shape from function: {output_fn.shape}")
print(f"Weights shape from function: {weights_fn.shape}")

# 验证结果是否一致
assert torch.allclose(output, output_fn)
print("Function output matches step-by-step output.")

# 总结:
# 1. 注意力机制的核心是为序列中的每个元素（Value）计算一个权重。
# 2. 这个权重是通过Query和所有Key的相似度计算得出的。
# 3. **缩放点积注意力**是计算这种相似度的标准方法：点积 -> 缩放 -> Softmax。
# 4. 最终的输出是所有Value根据注意力权重的加权和，它是一个融合了全局信息但有所侧重的上下文向量。
