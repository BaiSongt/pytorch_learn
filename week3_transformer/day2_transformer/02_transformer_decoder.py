
import torch
import torch.nn as nn
import math

# --- 前言 --- 
# Transformer的解码器 (Decoder) 的任务是生成目标序列。
# 它与编码器结构相似，也是由N个相同的“解码器层 (Decoder Layer)”堆叠而成。
# 但解码器层比编码器层要复杂，因为它需要处理两组输入：
# 1. 已经生成的目标序列（用于自注意力）。
# 2. 编码器的输出（用于交叉注意力）。

# --- 1. 解码器层 (Decoder Layer) ---
# 一个解码器层由三个子层组成：
# 1. 带掩码的多头自注意力机制 (Masked Multi-Head Self-Attention)。
# 2. 多头交叉注意力机制 (Multi-Head Cross-Attention)。
# 3. 位置前馈网络 (Position-wise Feed-Forward Network)。

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential( # 简化版的前馈网络
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_outputs, trg_mask, src_mask=None):
        # trg: 目标序列的嵌入, shape: (batch_size, trg_len, d_model)
        # encoder_outputs: 编码器的输出, shape: (batch_size, src_len, d_model)
        # trg_mask: 目标序列的掩码，用于防止关注未来的token
        # src_mask: 源序列的掩码（可选，用于padding）
        
        # 1. 带掩码的多头自注意力
        # Q, K, V 都来自目标序列 trg
        # `attn_mask=trg_mask` 是这里的关键
        attn_output, _ = self.self_attn(trg, trg, trg, attn_mask=trg_mask)
        trg = self.norm1(trg + self.dropout(attn_output))
        
        # 2. 交叉注意力
        # Query 来自解码器的上一个子层 (trg)
        # Key 和 Value 来自编码器的输出 (encoder_outputs)
        # 这是解码器“关注”输入序列的地方
        cross_attn_output, _ = self.cross_attn(trg, encoder_outputs, encoder_outputs, key_padding_mask=src_mask)
        trg = self.norm2(trg + self.dropout(cross_attn_output))
        
        # 3. 位置前馈网络
        ff_output = self.feed_forward(trg)
        trg = self.norm3(trg + self.dropout(ff_output))
        
        return trg

# --- 2. 完整的解码器 (Decoder) ---
# 解码器将N个解码器层堆叠起来，并包含自己的词嵌入和位置编码层。

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model)) # 简化版位置编码
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) 
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size) # 最后的线性层，用于生成词汇表大小的logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, encoder_outputs, trg_mask, src_mask=None):
        # trg: 目标序列的token索引, shape: (batch_size, trg_len)
        # encoder_outputs: 编码器的输出, shape: (batch_size, src_len, d_model)
        
        # 1. 词嵌入和位置编码
        embedded = self.embedding(trg) * math.sqrt(self.d_model)
        pos_encoded = embedded + self.pos_encoding[:, :trg.size(1)]
        trg_encoded = self.dropout(pos_encoded)
        
        # 2. 逐层通过N个解码器层
        for layer in self.layers:
            trg_encoded = layer(trg_encoded, encoder_outputs, trg_mask, src_mask)
            
        # 3. 最终的线性输出层
        output = self.fc_out(trg_encoded)
        return output

# --- 3. 创建和理解掩码 (Masking) ---
# **目标序列掩码 (Target Sequence Mask)**
# - **目的**: 在自注意力中，确保一个token的预测只能依赖于它之前的token，而不能“偷看”未来的token。
# - **实现**: 创建一个上三角矩阵，对角线以上的位置为True（或一个极小的负数），表示这些位置需要被掩盖。

def create_target_mask(trg_seq):
    # trg_seq shape: (batch_size, trg_len)
    trg_len = trg_seq.size(1)
    # `torch.triu` 创建一个上三角矩阵
    trg_mask = torch.triu(torch.ones((trg_len, trg_len)), diagonal=1).bool()
    return trg_mask # shape: (trg_len, trg_len)

# --- 4. 使用解码器 ---
print("--- Using the Transformer Decoder ---")

# 定义超参数
TRG_VOCAB_SIZE = 1200
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
MAX_LEN = 100

# 实例化解码器
decoder = Decoder(TRG_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN)

# 创建虚拟输入
# 编码器的输出
encoder_output_sample = torch.randn(2, 50, D_MODEL) # (batch_size, src_len, d_model)
# 目标序列 (解码器已经生成的部分)
trg_input_sample = torch.randint(0, TRG_VOCAB_SIZE, (2, 60)) # (batch_size, trg_len)

# 创建目标序列掩码
trg_mask_sample = create_target_mask(trg_input_sample)

# 前向传播
decoder_output = decoder(trg_input_sample, encoder_output_sample, trg_mask_sample)

print(f"Encoder output shape: {encoder_output_sample.shape}")
print(f"Target sequence input shape: {trg_input_sample.shape}")
print(f"Target mask shape: {trg_mask_sample.shape}")
print(f"Final decoder output shape: {decoder_output.shape}")
print("(batch_size, trg_len, trg_vocab_size)")

# 总结:
# 1. 解码器层比编码器层多一个**交叉注意力**模块，这是它与编码器交互的桥梁。
# 2. 解码器的自注意力是**带掩码的**，以保证其自回归（auto-regressive）的特性，即一次生成一个词。
# 3. 解码器的最终输出是一个在整个目标词汇表上的概率分布（logits），用于预测下一个词。
