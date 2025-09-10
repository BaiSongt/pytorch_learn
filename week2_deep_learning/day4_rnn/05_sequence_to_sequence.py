
import torch
import torch.nn as nn
import random

# --- 前言 --- 
# 序列到序列 (Seq2Seq) 模型是一种能将一个序列转换为另一个序列的神经网络架构。
# 它由两个核心组件构成：一个编码器 (Encoder) 和一个解码器 (Decoder)。
#
# 工作流程:
# 1. Encoder (编码器) 读取输入序列（例如，一个法语句子），并将其压缩成一个固定长度的“上下文向量”。
#    这个向量可以被认为是整个输入序列的“语义摘要”。
# 2. Decoder (解码器) 接收这个上下文向量，并根据它逐个地生成输出序列（例如，对应的英语句子）。
#
#      [Input Seq] -> Encoder -> [Context Vector] -> Decoder -> [Output Seq]
#

# --- 1. 编码器 (Encoder) ---
# 编码器是一个RNN（通常是LSTM或GRU），它遍历输入序列的每一个token。
# 它的最终隐藏状态 (hidden state) 和细胞状态 (cell state) 被用作上下文向量，传递给解码器。

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_len)
        embedded = self.embedding(input_seq) # -> (batch_size, seq_len, embedding_dim)
        
        # RNN的输出包括所有时间步的输出，以及最后一个时间步的隐藏状态和细胞状态
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # 我们只关心最后的隐藏和细胞状态，它们是上下文向量
        # hidden: (n_layers, batch_size, hidden_dim)
        # cell: (n_layers, batch_size, hidden_dim)
        return hidden, cell

# --- 2. 解码器 (Decoder) ---
# 解码器是另一个RNN。它接收编码器的上下文向量作为其初始隐藏状态。
# 它逐个生成输出token。在每个时间步，它接收前一个时间步生成的token作为输入，
# 并结合其当前的隐藏状态，来预测下一个token。

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, input_token, hidden, cell):
        # input_token: (batch_size, 1) - 当前时间步的输入token
        # hidden, cell: 上一个时间步的隐藏状态和细胞状态
        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output: (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1)) # -> (batch_size, output_vocab_size)
        return prediction, hidden, cell

# --- 3. Seq2Seq 模型 ---
# Seq2Seq模型将编码器和解码器包装在一起。

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_seq, trg_seq, teacher_forcing_ratio=0.5):
        # src_seq: 输入序列 (batch_size, src_len)
        # trg_seq: 目标序列 (batch_size, trg_len)
        batch_size = trg_seq.shape[0]
        trg_len = trg_seq.shape[1]
        trg_vocab_size = self.decoder.output_vocab_size
        
        # 用于存储解码器输出的张量
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. 将输入序列送入编码器，获取上下文向量
        hidden, cell = self.encoder(src_seq)
        
        # 2. 解码器的第一个输入是 <sos> (start of sentence) token
        decoder_input = trg_seq[:, 0].unsqueeze(1)
        
        # 3. 逐个token地进行解码
        for t in range(1, trg_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = decoder_output
            
            # **教师强制 (Teacher Forcing)**
            # 一种训练技巧：以一定概率，将“真实的目标token”作为解码器的下一个输入，
            # 而不是使用解码器自己刚刚生成的token。这能加速和稳定训练。
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = trg_seq[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

print("--- Seq2Seq Model Structure ---")
# 实例化模型 (参数为虚拟值)
INPUT_DIM = 1000
OUTPUT_DIM = 1200
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
model = Seq2Seq(enc, dec, device='cpu')
print(model)
print("-"*30)

# --- 4. 瓶颈问题与注意力机制 (Attention) ---
# - **瓶颈**: 基础的Seq2Seq模型必须将输入序列的所有信息压缩到一个固定长度的上下文向量中。
#   当输入序列很长时，这个向量很难承载所有信息，导致信息丢失。
#
# - **注意力机制 (Attention)**:
#   这是对基础Seq2Seq模型的关键改进。它允许解码器在生成每个输出token时，
#   都能“回头看”并“关注”输入序列的所有部分，并为最重要的部分分配更高的权重。
#   解码器不再只依赖于单一的上下文向量，而是依赖于一个根据当前解码步动态计算的、
#   加权的上下文向量。这使得模型能更好地处理长序列，并成为现代NLP（包括Transformer）的核心。
#   我们将在第三周详细学习注意力机制。

# 总结:
# 1. Seq2Seq模型通过编码器-解码器架构来处理序列到序列的任务。
# 2. 编码器将输入序列编码为上下文向量。
# 3. 解码器使用上下文向量来生成输出序列。
# 4. 教师强制是一种重要的训练技巧。
# 5. 注意力机制通过允许解码器关注输入序列的不同部分，解决了基础Seq2Seq模型的瓶颈问题。
