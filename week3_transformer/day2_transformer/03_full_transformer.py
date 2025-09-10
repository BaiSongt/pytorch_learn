
import torch
import torch.nn as nn
import math

# --- 前言 --- 
# 我们将把之前实现的编码器和解码器组装成一个完整的Transformer模型。
# 这个模型将能够处理一个序列到序列的任务，例如机器翻译。

# --- 复用之前的模块 (为保持脚本独立，在此重新定义简化版) ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first)
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        return self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_mask):
        src = self.norm1(src + self.dropout(self.self_attn(src, src, src, key_padding_mask=src_mask)))
        src = self.norm2(src + self.dropout(self.ffn(src)))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.norm1(trg + self.dropout(self.self_attn(trg, trg, trg, attn_mask=trg_mask)))
        trg = self.norm2(trg + self.dropout(self.cross_attn(trg, enc_src, enc_src, key_padding_mask=src_mask)))
        trg = self.norm3(trg + self.dropout(self.ffn(trg)))
        return trg

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- 1. 完整的Transformer架构 ---

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        
        # 词嵌入和位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器和解码器
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # 最终的线性输出层
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def create_src_padding_mask(self, src, pad_idx):
        # src shape: (batch_size, src_len)
        # 如果一个位置是pad_idx，则为True，否则为False
        return src == pad_idx

    def create_trg_look_ahead_mask(self, trg):
        # trg shape: (batch_size, trg_len)
        trg_len = trg.shape[1]
        # 返回一个上三角矩阵
        return torch.triu(torch.ones((trg_len, trg_len)), diagonal=1).bool().to(trg.device)

    def forward(self, src, trg, src_pad_idx=0):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        
        # 1. 创建掩码
        src_padding_mask = self.create_src_padding_mask(src, src_pad_idx)
        trg_look_ahead_mask = self.create_trg_look_ahead_mask(trg)
        
        # 2. 编码器部分
        src_embedded = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        enc_src = src_embedded
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, src_padding_mask)
        
        # 3. 解码器部分
        trg_embedded = self.dropout(self.pos_encoding(self.trg_embedding(trg) * math.sqrt(self.d_model)))
        dec_output = trg_embedded
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_src, trg_look_ahead_mask, src_padding_mask)
            
        # 4. 最终输出
        output = self.fc_out(dec_output)
        return output

# --- 2. 使用完整的Transformer模型 ---
print("--- Using the Full Transformer Model ---")

# 定义超参数
SRC_VOCAB_SIZE = 5000
TRG_VOCAB_SIZE = 5000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
MAX_LEN = 100

# 实例化模型
model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN)

# 创建虚拟输入
# (batch_size, seq_len)
src_input = torch.randint(1, SRC_VOCAB_SIZE, (2, 50)) # padding_idx=0
trg_input = torch.randint(1, TRG_VOCAB_SIZE, (2, 60))

# 前向传播
output_logits = model(src_input, trg_input)

print(f"Source input shape: {src_input.shape}")
print(f"Target input shape: {trg_input.shape}")
print(f"Final output shape (logits): {output_logits.shape}")
print("(batch_size, trg_len, trg_vocab_size)")

# 总结:
# 1. Transformer模型由一个编码器和一个解码器组成。
# 2. 编码器负责处理输入序列，生成上下文感知的表示。
# 3. 解码器接收编码器的输出和已生成的目标序列，来预测下一个token。
# 4. 掩码（Padding Mask 和 Look-ahead Mask）在训练过程中至关重要，以确保模型不会关注到无效信息。
# 5. 整个模型是一个端到端的、巨大的神经网络，通过反向传播进行训练。
