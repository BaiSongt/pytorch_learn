
import torch
import torch.nn as nn

# --- 前言 --- 
# LSTM和GRU是为了解决简单RNN的梯度消失和长期依赖问题而设计的。
# 它们的核心思想是引入“门控机制 (gating mechanism)”，让网络能够有选择性地
# 学习需要保留或遗忘哪些信息。

# --- 1. 准备输入数据 ---
# 输入数据的格式与简单RNN完全相同。
# (N, L, H_in) -> (batch_size, sequence_length, input_size)
batch_size = 2
seq_len = 3
input_size = 4
hidden_size = 5
num_layers = 1

input_seq = torch.randn(batch_size, seq_len, input_size)

# --- 2. LSTM (长短期记忆网络) ---
# LSTM引入了一个“细胞状态 (Cell State)” c_t，它像一条传送带一样贯穿整个时间序列，
# 信息可以很容易地在上面流动而不发生改变。LSTM通过三个“门”来控制细胞状态：
#
# 1. **遗忘门 (Forget Gate)**: 决定从上一个细胞状态 c_{t-1} 中丢弃哪些信息。
# 2. **输入门 (Input Gate)**: 决定将哪些新的信息存入当前的细胞状态 c_t。
# 3. **输出门 (Output Gate)**: 决定从当前的细胞状态 c_t 中输出哪些信息到隐藏状态 h_t。

print("--- 1. LSTM (Long Short-Term Memory) ---")

lstm_layer = nn.LSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    batch_first=True
)

# LSTM的初始状态包括一个隐藏状态h_0和一个细胞状态c_0
h0_lstm = torch.randn(num_layers, batch_size, hidden_size)
c0_lstm = torch.randn(num_layers, batch_size, hidden_size)

# 前向传播
# LSTM返回 outputs, (h_n, c_n)
outputs_lstm, (hn_lstm, cn_lstm) = lstm_layer(input_seq, (h0_lstm, c0_lstm))

print(f"Shape of LSTM `outputs`: {outputs_lstm.shape}")
print(f"Shape of LSTM `h_n` (final hidden state): {hn_lstm.shape}")
print(f"Shape of LSTM `c_n` (final cell state): {cn_lstm.shape}")
print("-"*30)

# --- 3. GRU (门控循环单元) ---
# GRU是LSTM的一个简化版本，它将遗忘门和输入门合并为了一个单一的“更新门”。
# 它也合并了细胞状态和隐藏状态。GRU比LSTM参数更少，计算成本更低。
#
# 1. **更新门 (Update Gate)**: 类似于LSTM的遗忘门和输入门的组合。它决定了在多大程度上保留过去的隐藏状态，以及在多大程度上接收新的信息。
# 2. **重置门 (Reset Gate)**: 决定如何将新的输入信息与过去的隐藏状态相结合。

print("--- 2. GRU (Gated Recurrent Unit) ---")

gru_layer = nn.GRU(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    batch_first=True
)

# GRU只有一个隐藏状态，没有细胞状态
h0_gru = torch.randn(num_layers, batch_size, hidden_size)

# 前向传播
# GRU的返回格式与简单RNN相同: outputs, h_n
outputs_gru, hn_gru = gru_layer(input_seq, h0_gru)

print(f"Shape of GRU `outputs`: {outputs_gru.shape}")
print(f"Shape of GRU `h_n` (final hidden state): {hn_gru.shape}")
print("-"*30)

# --- 4. 如何选择：LSTM vs. GRU? ---
# - **性能**: 两者在大多数任务上的性能都非常相似，没有一个绝对的赢家。
# - **计算效率**: GRU的参数更少，计算速度比LSTM稍快。
# - **经验法则**:
#   1. 从LSTM开始，因为它更强大、更具表现力，是许多任务的默认选择。
#   2. 如果你非常关心计算效率，或者在你的特定任务上GRU表现更好，那么可以选择GRU。
#   3. 在数据量较少的情况下，GRU由于参数较少，可能更不容易过拟合。

# --- 5. 堆叠RNN (Stacked RNN) ---
# 我们可以通过设置 `num_layers > 1` 来构建一个“堆叠RNN”。
# 在堆叠RNN中，第一层的输出序列将作为第二层的输入序列。
# 这使得网络能够在不同的时间尺度上学习到更复杂的特征和模式。

print("--- 3. Stacked RNN ---")
stacked_lstm = nn.LSTM(input_size, hidden_size, num_layers=4, batch_first=True)
outputs_stacked, (hn_stacked, cn_stacked) = stacked_lstm(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Stacked LSTM output shape: {outputs_stacked.shape}")
print(f"Stacked LSTM final hidden shape: {hn_stacked.shape}")

# 总结:
# 1. LSTM和GRU通过引入门控机制，有效地解决了简单RNN的长期依赖问题。
# 2. LSTM使用一个独立的细胞状态和三个门（遗忘、输入、输出）来精细地控制信息流。
# 3. GRU是LSTM的简化版，使用两个门（更新、重置），计算效率更高。
# 4. 在实践中，LSTM和GRU是处理序列问题的标准和首选工具。
# 5. 堆叠多层RNN（`num_layers > 1`）可以提升模型的表达能力。
