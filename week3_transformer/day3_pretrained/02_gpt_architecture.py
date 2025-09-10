
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 前言 ---
# GPT (Generative Pre-trained Transformer) 系列模型是基于Transformer解码器构建的、
# 强大的自回归语言模型。与BERT的双向性不同，GPT是单向的（从左到右），
# 这使得它在文本生成任务上表现得尤为出色。

# --- 1. GPT 架构 ---
# GPT的架构非常纯粹：它就是N个Transformer解码器层的堆叠。
# 它只使用了解码器，而没有使用编码器。
#
# - **核心组件**: 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)。
#   这个“掩码”是关键，它确保了在预测第 t 个位置的token时，模型只能关注到 t 之前的位置，
#   而不能“偷看”未来的信息。这正是“自回归”的体现。

print("---" + "1. GPT Architecture: A stack of Transformer Decoders. ---")
print("---" * 30)

# --- 2. 预训练任务: 标准语言建模 (Standard Language Modeling) ---
# GPT的预训练任务非常简单、统一且强大：预测下一个词。
# - **目标**: 给定一个文本序列 (x_1, x_2, ..., x_{t-1})，最大化预测下一个词 x_t 的概率。
# - **损失函数**: 本质上就是一个标准的多分类交叉熵损失。
#
#   例如，对于句子 "I am learning NLP"：
#   - 输入 "I"，模型需要预测 "am"。
#   - 输入 "I am"，模型需要预测 "learning"。
#   - 输入 "I am learning"，模型需要预测 "NLP"。
#
# 通过在海量的文本上进行这个简单的任务，模型被迫学习到语法、语义、事实知识等深刻的语言规律。

class SimpleGPT(nn.Module):
    """一个展示GPT核心思想的极简模型"""
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()
        # GPT的输入表示也是 Token Embedding + Position Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # 使用PyTorch内置的TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=d_ff, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 最终的线性层，用于预测词汇表中的下一个词
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src_seq):
        # src_seq: (batch_size, seq_len)
        seq_len = src_seq.shape[1]
        
        # 创建位置索引
        positions = torch.arange(0, seq_len).unsqueeze(0).to(src_seq.device)
        
        # 创建掩码，防止关注未来的token
        # `generate_square_subsequent_mask` 是PyTorch提供的便捷函数
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src_seq.device)
        
        # 嵌入和位置编码相加
        embedded = self.token_embedding(src_seq) + self.position_embedding(positions)
        
        # 通过解码器层
        # 在自回归任务中，target序列和memory序列是相同的
        decoder_output = self.decoder(tgt=embedded, memory=embedded, tgt_mask=tgt_mask)
        
        # 输出logits
        logits = self.fc_out(decoder_output)
        return logits

print("---" + "2. Pre-training Task: Standard (auto-regressive) Language Modeling. ---")
print("---" * 30)

# --- 3. 文本生成 (Text Generation) ---
# GPT的自回归特性使其天然地适合文本生成。
# **生成过程 (概念性)**:
# 1. 提供一个初始的文本“提示 (prompt)”，例如 "Once upon a time,"。
# 2. 将提示输入模型，模型会预测出下一个最可能的词，例如 "there"。
# 3. 将预测出的词拼接到提示后面，形成新的输入 "Once upon a time, there"。
# 4. 将这个新的、更长的序列再次输入模型，预测再下一个词，例如 "was"。
# 5. 循环往复，直到生成了特殊的中止符 `[EOS]` 或者达到了最大长度。
#
# **解码策略 (Decoding Strategy)**:
# - **Greedy Search**: 每次都选择概率最高的那个词。简单但可能导致重复、无趣的文本。
# - **Beam Search**: 在每一步都保留k个最可能的候选序列，并在下一步基于这k个序列进行扩展，可以生成更流畅的文本。
# - **Sampling (Top-k, Top-p/Nucleus)**: 从概率分布中进行随机采样，而不是总选择最优的。可以增加文本的多样性和创造性。

print("---" + "3. Text Generation: Predict one token at a time, and feed it back. ---")
print("---" * 30)

# --- 4. 零样本/少样本学习 (Zero-shot / Few-shot Learning) ---
# - **现象**: 当GPT模型变得足够大（例如GPT-3），它会展现出惊人的“上下文学习 (In-context Learning)”能力。
# - **零样本 (Zero-shot)**: 无需任何训练样本，直接通过指令来让模型完成任务。
#   - *提示*: "Translate English to French: sea otter => loutre de mer"
# - **少样本 (Few-shot)**: 在提示中提供几个示例，模型会“领会”任务的模式并完成新的示例。
#   - *提示*: "Translate English to French:\nsea otter => loutre de mer\ncheese =>"
#   - *模型输出*: "fromage"
#
# 这种能力表明，巨大的语言模型通过海量数据的预训练，已经内化了某种形式的“元学习”能力。

print("---" + "4. Zero/Few-shot Learning: Solving tasks via clever prompting. ---")

# 总结:
# 1. **架构**: GPT = 堆叠的Transformer解码器。
# 2. **核心思想**: 通过**自回归语言建模**（预测下一个词）进行预训练，是**单向**的。
# 3. **应用**: 天然适合**文本生成**任务。
# 4. **特性**: 大型GPT模型展现出强大的**上下文学习**能力，可以通过提示工程（Prompt Engineering）完成各种任务，而无需微调。
# 5. **BERT vs. GPT**: BERT是双向的，是优秀的**语言理解器**；GPT是单向的，是优秀的**语言生成器**。
