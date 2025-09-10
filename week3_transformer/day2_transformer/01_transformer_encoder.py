
import torch
import torch.nn as nn
import math

# --- 前言 ---
# Transformer的编码器 (Encoder) 是一个由N个相同的“编码器层 (Encoder Layer)”堆叠而成的结构。
# 它的主要作用是接收一个经过词嵌入和位置编码的序列，然后为序列中的每一个token
# 生成一个富含上下文信息的、深度的表示（representation）。

# --- 1. 位置前馈网络 (Position-wise Feed-Forward Network) ---
# 这是构成编码器层的一个简单组件。它是一个小型的、由两个线性层组成的网络，
# 它独立地、相同地作用于序列中的每一个token。
# FFN(x) = max(0, x @ W1 + b1) @ W2 + b2

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型的维度，也是输入和输出的维度
            d_ff: 中间隐藏层的维度，通常远大于d_model (例如 d_ff = 4 * d_model)
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

# --- 2. 层归一化 (Layer Normalization) ---
# - **与BatchNorm的区别**: BatchNorm在批次(batch)维度上对特征进行归一化，
#   而LayerNorm在每个样本的特征(feature)维度上进行归一化。
# - **为何在Transformer中使用**: LayerNorm与序列长度无关，更适合处理可变长度的序列数据（NLP任务），
#   在RNN和Transformer中是标准选择。

# --- 3. 编码器层 (Encoder Layer) ---
# 一个编码器层由两个子层组成：
# 1. 一个多头自注意力机制。
# 2. 一个位置前馈网络。
# 每个子层的输出都经过了 Dropout, 残差连接 (Add) 和层归一化 (Norm) 的处理。

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # PyTorch内置了MultiheadAttention，我们直接使用它
        # 注意：PyTorch的实现需要 (seq_len, batch_size, d_model) 的输入，除非 batch_first=True
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len, d_model)

        # 1. 多头自注意力子层
        # Q, K, V都来自同一个src
        attn_output, _ = self.self_attn.forward(src, src, src, key_padding_mask=src_mask)
        # 残差连接和层归一化
        # x = x + dropout(sublayer(x))
        src = self.norm1(src + self.dropout(attn_output))

        # 2. 位置前馈网络子层
        ff_output = self.feed_forward(src)
        # 残差连接和层归一化
        src = self.norm2(src + self.dropout(ff_output))

        return src

# --- 4. 完整的编码器 (Encoder) ---
# 编码器就是将N个编码器层堆叠起来。

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 复用day1的位置编码模块 (这里简化实现)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        # 1. 词嵌入和位置编码
        embedded = self.embedding(src) * math.sqrt(self.d_model) # 乘以sqrt(d_model)是论文中的一个细节
        pos_encoded = embedded + self.pos_encoding[:, :src.size(1)]
        src_encoded = self.dropout(pos_encoded)

        # 2. 逐层通过N个编码器层
        for layer in self.layers:
            src_encoded = layer(src_encoded, src_mask)

        return src_encoded

# --- 5. 使用编码器 ---
print("--- Using the Transformer Encoder ---")

# 定义超参数
VOCAB_SIZE = 1000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
MAX_LEN = 100

# 实例化编码器
encoder = Encoder(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN)

# 创建虚拟输入
# (batch_size, seq_len)
src_input = torch.randint(0, VOCAB_SIZE, (2, 50))

# 前向传播
encoder_output = encoder(src_input)

print(f"Input shape: {src_input.shape}")
print(f"Final encoder output shape: {encoder_output.shape}")
print("(batch_size, src_len, d_model)")

# 总结:
# 1. Transformer编码器由N个相同的编码器层堆叠而成。
# 2. 每个编码器层包含一个**多头自注意力**模块和一个**位置前馈网络**模块。
# 3. **残差连接**和**层归一化**是连接这两个子层的关键，它们保证了深度网络的稳定训练。
# 4. 编码器的最终输出是一个与输入序列等长的序列，但其中每个token的向量表示都融合了整个输入序列的上下文信息。
