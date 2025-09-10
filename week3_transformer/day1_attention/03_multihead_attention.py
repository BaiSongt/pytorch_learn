
import torch
import torch.nn as nn
import math

# --- 前言 --- 
# 多头注意力机制 (Multi-Head Attention) 是对简单自注意力机制的扩展。
# 它不是只计算一次注意力，而是将Q, K, V通过不同的线性变换投射多次，
# 然后并行地为每个投射后的版本计算注意力，最后将所有结果拼接并再次进行线性变换。
# 
# **核心思想**: 与其让模型用一套Q,K,V学习一种关系，不如给它多套Q,K,V（即多个“头”），
# 让每个头都能学习到输入序列中不同方面的关系（例如，一个头可能关注语法依赖，另一个头可能关注同义词关系）。

# --- 1. 准备输入数据 ---
# (batch_size, seq_len, embedding_dim)
batch_size = 2
seq_len = 4
embedding_dim = 12 # 为了能被头数整除，我们设为12

input_sequence = torch.randn(batch_size, seq_len, embedding_dim)

# --- 2. 多头注意力模块的实现 ---

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads # 每个头的维度
        
        # 我们需要4个线性层：分别用于Q, K, V的投射，以及最后输出的投射
        self.W_q = nn.Linear(embedding_dim, embedding_dim) # Q, K, V的投射维度与输入/输出维度相同
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim) # 用于最后输出
        
    def split_heads(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        # 目标: 将 embedding_dim 拆分为 (num_heads, head_dim)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 交换维度，让 num_heads 成为批处理的一部分，以便并行计算
        # -> (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        # 在自注意力中, query, key, value 都是同一个输入
        # query, key, value shape: (batch_size, seq_len, embedding_dim)
        
        # 1. 线性投射
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 拆分成多个头
        Q = self.split_heads(Q) # -> (batch_size, num_heads, seq_len, head_dim)
        K = self.split_heads(K) # -> (batch_size, num_heads, seq_len, head_dim)
        V = self.split_heads(V) # -> (batch_size, num_heads, seq_len, head_dim)
        
        # 3. 并行计算缩放点积注意力
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        # attention_output shape: (batch_size, num_heads, seq_len, head_dim)
        
        # 4. 拼接所有头的输出并进行最终的线性变换
        # 首先，需要将 attention_output 的维度恢复成 (batch_size, seq_len, embedding_dim)
        # contiguous() 确保张量在内存中是连续的，然后才能调用 .view()
        concatenated_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        final_output = self.W_o(concatenated_output)
        
        return final_output, attention_weights

# --- 3. 使用多头注意力模块 ---

print("--- Using the MultiHeadAttention Module ---")

num_heads = 4 # 假设我们使用4个头
mha_module = MultiHeadAttention(embedding_dim, num_heads)

# 在自注意力场景下，Q, K, V是相同的
output, weights = mha_module(input_sequence, input_sequence, input_sequence)

print(f"Input shape: {input_sequence.shape}")
print(f"Final output shape: {output.shape}")
# 注意力权重的形状: (batch_size, num_heads, seq_len, seq_len)
print(f"Attention weights shape: {weights.shape}")

# 总结:
# 1. 多头注意力是Transformer的核心计算单元。
# 2. 它通过将输入投射并拆分到多个子空间（头），让模型能并行地关注不同方面的信息。
# 3. 实现的关键在于维度的变换：
#    a. `split_heads`: (batch, seq, embed) -> (batch, heads, seq, head_dim)
#    b. `concatenate_heads`: (batch, heads, seq, head_dim) -> (batch, seq, embed)
# 4. 最后的线性层 `W_o` 负责将所有头的信息融合起来。
# 5. PyTorch官方也提供了优化好的 `nn.MultiheadAttention` 模块，在实际项目中应优先使用，
#    但理解其内部实现对于深入掌握Transformer至关重要。
