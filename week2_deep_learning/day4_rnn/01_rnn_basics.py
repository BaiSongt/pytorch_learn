
import torch
import torch.nn as nn

# --- 前言 --- 
# 循环神经网络 (Recurrent Neural Network, RNN) 是一类专门用于处理“序列数据”的神经网络。
# 序列数据包括：时间序列（股票价格）、文本（单词序列）、音频、视频等。
# 与MLP和CNN不同，RNN具有“记忆”能力。它内部有一个循环结构，允许信息从一个时间步持续传递到下一个时间步。

# --- 1. RNN的核心思想 --- 
# 在每个时间步 t，RNN会接收两个输入：
# 1. 当前时间步的输入数据 x_t (例如，句子中的一个词)。
# 2. 上一个时间步的隐藏状态 h_{t-1} (这就是“记忆”)。
# 然后，它会计算出两个输出：
# 1. 当前时间步的输出 o_t。
# 2. 传递给下一个时间步的新隐藏状态 h_t。
# h_t = f(W * x_t + U * h_{t-1} + b)，其中f是激活函数（如tanh）。
# 所有时间步共享同一套权重 W, U 和偏置 b。

# --- 2. 准备输入数据 ---
# RNN的输入通常是一个3D张量，形状为 (N, L, H_in):
# - N: Batch size (批次大小)
# - L: Sequence length (序列长度，例如句子中的单词数)
# - H_in: Input size (输入特征的维度，例如每个词的嵌入向量维度)

batch_size = 2
seq_len = 3
input_size = 4 # 假设每个词被一个4维向量表示

# 创建一个虚拟的输入批次
input_seq = torch.randn(batch_size, seq_len, input_size)

print("--- 1. Input Sequence Tensor ---")
print(f"Shape of the input sequence: {input_seq.shape}")
print("-"*30)

# --- 3. 使用 `nn.RNN` 层 ---
# - `input_size`: 输入特征的维度 (H_in)。
# - `hidden_size`: 隐藏状态的特征维度 (H_out)。
# - `num_layers`: RNN的层数。堆叠多个RNN层可以构建更深的网络。
# - `batch_first=True`: 一个非常重要的参数，它让输入的形状变为我们常用的 (N, L, H_in)。
#   如果为False（默认），则输入形状应为 (L, N, H_in)。

hidden_size = 5
num_layers = 1

rnn_layer = nn.RNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    batch_first=True
)

print("--- 2. nn.RNN Layer ---")
print(rnn_layer)
print("-"*30)

# --- 4. 前向传播与输出解析 ---

# 我们可以选择提供一个初始的隐藏状态 h_0。如果不提供，它将默认为全0。
# h_0 的形状: (num_layers, batch_size, hidden_size)
h_0 = torch.randn(num_layers, batch_size, hidden_size)

# 前向传播
# `nn.RNN` 返回两个东西：`outputs` 和 `h_n`
outputs, h_n = rnn_layer(input_seq, h_0)

print("--- 3. Forward Pass and Outputs ---")

# **解析 `outputs`**
# - `outputs` 包含了序列中**每一个**时间步的隐藏状态输出。
# - 形状: (batch_size, seq_len, hidden_size)
print(f"Shape of `outputs`: {outputs.shape}")
# `outputs[:, -1, :]` 应该与 `h_n` 的内容相同（对于单层RNN）

# **解析 `h_n`**
# - `h_n` 只包含了序列中**最后一个**时间步的隐藏状态。
# - 形状: (num_layers, batch_size, hidden_size)
print(f"Shape of `h_n` (final hidden state): {h_n.shape}")

# `h_n` 通常被用作整个输入序列的“语义摘要”，可以被送入一个全连接层进行分类或回归。
# `outputs` 则用于需要对序列中每个元素进行预测的任务（如词性标注）。

# --- 5. 简单RNN的局限性 ---
# - **梯度消失/爆炸**: 在反向传播时，梯度会通过时间步反复乘以相同的权重矩阵。
#   如果权重矩阵的特征值小于1，梯度会指数级缩小（梯度消失）；如果大于1，则会指数级增大（梯度爆炸）。
# - **长期依赖问题**: 由于梯度消失，简单RNN很难学习到序列中相距很远的两个元素之间的依赖关系（例如，一个长段落的开头和结尾）。
# 
# 为了解决这些问题，更复杂的门控RNN单元，如LSTM和GRU，被提了出来。

# 总结:
# 1. RNN通过内部的循环和隐藏状态来处理序列数据。
# 2. `nn.RNN` 是PyTorch中的基础RNN实现，`batch_first=True` 是一个常用且方便的设置。
# 3. RNN返回包含所有时间步输出的 `outputs` 和只包含最后一个时间步输出的 `h_n`。
# 4. 简单RNN存在梯度消失/爆炸问题，难以捕捉长期依赖，因此在实践中通常被LSTM或GRU所取代。
