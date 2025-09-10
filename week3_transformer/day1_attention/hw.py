## 作业
# 1. 实现基础注意力机制
# 2. 构建多头注意力模块
# 3. 完成注意力可视化项目


import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class SelfAttention(nn.Module):
    """自注意力"""

    def __init__(self, embedding_dim: int, k_dim: int, v_dim: int):
        """初始化

        Args:
            ebd_dim (int): 嵌入向量维度
            k_dim (int): query 和 key 的维度
            v_dim (int): value 的维度
        """
        super().__init__()

        # 初始化训练权重
        self.W_q = nn.Linear(in_features=embedding_dim, out_features=k_dim, bias=False)
        self.W_k = nn.Linear(in_features=embedding_dim, out_features=k_dim, bias=False)
        self.W_v = nn.Linear(in_features=embedding_dim, out_features=v_dim, bias=False)
        self.k_dim = k_dim

        print(f"{'-' * 20} init {'-'*20}")
        print(f"W q: {self.W_q}")
        print(f"W k: {self.W_k}")
        print(f"W v: {self.W_v}")

    def forward(self, input:torch.Tensor, mask=None):
        query = self.W_q(input)
        key = self.W_k(input)
        value = self.W_v(input)

        # 计算注意力
        # score =  Q @ K.T  / sqrt(k_dim)
        # 转置 seq_len 和 ebd_dim 用于矩阵乘法
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_dim)

        # mask for decoder
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        # att_w = softmax(att_score)
        attention_weights = torch.softmax(scores, dim=-1)
        # o = softmax( Q @ K.T / sqrt(d_k)) @ V
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

# 多头注意力机制的核心在于把自注意力的 Q K V 按照数目进行拆分， 然后再每个分别计算输出

class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, head_num: int, batch_size:int, seq_len:int, embedding_dim: int):
        """初始化

        Args:
            head_num (int): 注意力头数目
            embedding_dim (int): 嵌入向量维度
        """
        super().__init__()

        assert embedding_dim % head_num == 0, "Embedding dimension must be divisible by number of heads"

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = head_num
        self.head_dim = embedding_dim // head_num # 每个头的维度

        # 初始化训练权重
        self.W_q = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.W_k = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.W_v = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.W_o = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)

        print(f"{'-' * 20} init {'-'*20}")
        print(f"W q: {self.W_q}")
        print(f"W k: {self.W_k}")
        print(f"W v: {self.W_v}")

    def splite_head(self, x : torch.Tensor):
        """头拆分

        Args:
           x: 输入 Q, K, V的 张量， 然后按照  self.num_heads, self.head_dim 进行拆分
        Transpose:
            batch_size, seq_len, num_heads, head_dim
            -> batch_size, num_heads, seq_len, head_dim

        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None):
        # 1 线性投射
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 2 拆分
        Q = self.splite_head(Q)
        K = self.splite_head(K)
        V = self.splite_head(V)

        # 3 计算注意力分数
        k_dim = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(k_dim)

        if mask is not None:
            scores = torch.masked_fill( mask==0, -1e9)

        # 计算attention weights 和 b(output)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # 拼接所有头
        concatenated_output = output.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.embedding_dim)

        # 变换输出权重
        final_output = self.W_o(concatenated_output)

        return final_output, weights

def plot_attention_heatmap(attention_weights:torch.Tensor, sentence_tokens:torch.Tensor, head_idx = 0):
    """绘制单个头的注意力热力图
    """
    head_np = attention_weights[head_idx].detach().numpy()

    fig, ax = plt.subplots(figsize = (8, 8))
    cax = ax.matshow(head_np, cmap = 'viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(len(sentence_tokens)))
    ax.set_yticks(range(len(sentence_tokens)))
    ax.set_xticklabels(sentence_tokens)
    ax.set_yticklabels(sentence_tokens)

    # 设置刻度线位置，使其位于单元格之间
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel("Keys (Words being attended to)")
    plt.ylabel("Queries (Words doing the attending)")
    plt.title(f"Attention Heatmap for Head {head_idx}")
    plt.tight_layout()
    plt.show()

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


if __name__ == "__main__":

    sentence = ['I', 'am', 'a', 'red', 'cat', '.']
    seq_len = len(sentence)
    embedding_dim = 128
    num_heads = 8
    batch_size = 2

    # 创建虚拟的词嵌入
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    print(f"{'='*20} SelfAttention {'='*20}")
    self_att_module = SelfAttention(embedding_dim, seq_len, seq_len)
    output, weights = self_att_module(input_embeddings)
    print(f"input_embeddings  shape: {output.shape}")
    print(f"weight shape: {weights.shape}")

    plot_attention_heatmap(weights, sentence)

    print(f"\n{'='*20} Multi-Head Attention {'='*20}")
    multi_head_att_module = MultiHeadAttention(num_heads, batch_size, seq_len, embedding_dim)
    output, weights = multi_head_att_module(input_embeddings, input_embeddings, input_embeddings)
    print(f"input  shape: {output.shape}")
    print(f"weight shape: {weights.shape}")

    # 可视化
    plot_all_head_attentions(weights, sentence)
