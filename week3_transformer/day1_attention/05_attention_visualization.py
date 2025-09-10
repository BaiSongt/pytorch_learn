
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# --- 前言 --- 
# 注意力权重矩阵 `attention_weights` 的形状为 (batch_size, num_heads, seq_len, seq_len)。
# 这个矩阵告诉我们，对于序列中的每一个词（Query），模型在多大程度上关注了序列中的其他所有词（Key）。
# 将这个矩阵可视化，可以让我们直观地理解模型的行为。

# --- 1. 准备一个多头注意力模块和输入 ---
# 我们复用前一个脚本中的MultiHeadAttention模块
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        concatenated = attention_output.transpose(1, 2).contiguous().view(query.shape[0], -1, self.embedding_dim)
        final_output = self.W_o(concatenated)
        return final_output, attention_weights

# 准备输入
# 假设我们有一个经过分词和嵌入的句子
sentence = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
seq_len = len(sentence)
embedding_dim = 128
num_heads = 8

# 创建虚拟的词嵌入
input_embeddings = torch.randn(1, seq_len, embedding_dim)

# --- 2. 获取注意力权重 ---

mha_module = MultiHeadAttention(embedding_dim, num_heads)
_, attention_weights = mha_module(input_embeddings, input_embeddings, input_embeddings)

print("--- Attention Weights ---")
print(f"Shape of attention weights: {attention_weights.shape}")
print("(batch_size, num_heads, seq_len, seq_len)")
print("-"*30)

# --- 3. 可视化注意力权重 ---

def plot_attention_heatmap(attention_weights, sentence_tokens, head_idx=0):
    """绘制单个头的注意力热力图"""
    # 从批次中取出第一个样本，并选择一个头
    head_attention = attention_weights[0, head_idx].detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(head_attention, cmap='viridis')
    fig.colorbar(cax)
    
    # 设置坐标轴的刻度和标签
    ax.set_xticks(range(len(sentence_tokens)))
    ax.set_yticks(range(len(sentence_tokens)))
    ax.set_xticklabels(sentence_tokens, rotation=90)
    ax.set_yticklabels(sentence_tokens)
    
    # 设置刻度线位置，使其位于单元格之间
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.xlabel("Keys (Words being attended to)")
    plt.ylabel("Queries (Words doing the attending)")
    plt.title(f"Attention Heatmap for Head {head_idx}")
    plt.tight_layout()
    plt.show()

# 可视化第一个头的注意力
plot_attention_heatmap(attention_weights, sentence)

# --- 4. 如何解读热力图 ---
# - **Y轴 (Queries)**: 代表当前正在计算表示的词。
# - **X轴 (Keys)**: 代表句子中所有可以被关注的词。
# - **颜色**: 单元格 (i, j) 的颜色越亮，表示第 i 个词（Query）在计算其新表示时，
#   对第 j 个词（Key）的关注度越高。
#
# **可以观察到的模式 (在真实训练好的模型中)**:
# - **对角线**: 通常会很亮，因为一个词最关注的往往是它自己。
# - **特定关系**: 代词（如'it'）可能会高度关注它所指代的名词（如'cat'）。
# - **短语结构**: 一个形容词可能会关注它所修饰的名词。
# - **分隔符**: 像句号这样的分隔符可能会关注句子中的所有词，或不关注任何词。
#
# 不同的头会学习到不同的注意力模式，这正是多头注意力的强大之处。

# 我们可以绘制所有头的注意力图
def plot_all_head_attentions(attention_weights, sentence_tokens):
    num_heads = attention_weights.shape[1]
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        head_attention = attention_weights[0, i].detach().numpy()
        cax = ax.matshow(head_attention, cmap='viridis')
        ax.set_xticks(range(len(sentence_tokens)))
        ax.set_yticks(range(len(sentence_tokens)))
        ax.set_xticklabels(sentence_tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(sentence_tokens, fontsize=8)
        ax.set_title(f"Head {i}")
    fig.suptitle("Multi-Head Attention Patterns")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_all_head_attentions(attention_weights, sentence)

# 总结:
# 1. 可视化注意力权重是理解Transformer模型内部工作机制的强大工具。
# 2. 通过热力图，我们可以看到模型在处理序列时，是如何动态地分配其“计算焦点”的。
# 3. 不同的头学习不同的注意力模式，共同为模型提供丰富的上下文信息。
