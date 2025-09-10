
import torch
import torch.nn as nn

# --- 前言 --- 
# BERT (Bidirectional Encoder Representations from Transformers) 是一种基于Transformer编码器的、
# 革命性的预训练语言模型。它的核心创新在于通过“掩码语言模型”任务，实现了真正的“双向”上下文理解。

# --- 1. BERT 架构 ---
# BERT的架构非常直观：它就是N个我们前一天实现的Transformer编码器层的堆叠。
# - BERT-Base: 12层编码器, d_model=768, 12个注意力头, 总参数约1.1亿
# - BERT-Large: 24层编码器, d_model=1024, 16个注意力头, 总参数约3.4亿
# 它没有解码器。

print("--- 1. BERT Architecture: A stack of Transformer Encoders. ---")
print("-"*30)

# --- 2. BERT的输入表示 ---
# BERT的输入不仅仅是词嵌入，它由三部分相加而成：
# 1. **Token Embeddings**: 词的嵌入向量。
# 2. **Segment Embeddings**: 用于区分两个不同句子（在NSP任务中）。例如，所有属于句子A的token共享一个段嵌入，所有属于句子B的token共享另一个。
# 3. **Position Embeddings**: 与Transformer不同，BERT使用“可学习的”位置嵌入，而不是固定的正弦编码。
#
#   Final_Input = Token_Embedding + Segment_Embedding + Position_Embedding
#
# BERT还在每个输入序列的开头添加一个特殊的 `[CLS]` token，这个token对应的最终输出被用作整个序列的聚合表示，通常用于分类任务。
# 当输入包含两个句子时，它们之间会用一个 `[SEP]` token 分隔。
# 例: [CLS] my dog is cute [SEP] he likes playing [SEP]

print("--- 2. BERT Input Representation: Token + Segment + Position Embeddings. ---")
print("-"*30)

# --- 3. 预训练任务一: 掩码语言模型 (Masked Language Model, MLM) ---
# 这是BERT最核心的创新。
# - **动机**: 标准的语言模型是单向的（从左到右或从右到左），这限制了模型在预训练时能利用的上下文。
#   为了实现双向理解，BERT不能简单地让模型预测下一个词，因为在双向的设定下，模型能“看到”要预测的词，任务变得毫无意义。
# - **方法**:
#   1. 随机选择输入序列中15%的token。
#   2. 对于这15%的token：
#      - 80%的概率，用一个特殊的 `[MASK]` token替换掉它。
#      - 10%的概率，用一个随机的词替换它。
#      - 10%的概率，保持原词不变。
#   3. **目标**: 训练模型去预测这些被掩盖或替换掉的原始token是什么。
# - **效果**: 这强迫模型必须依赖左右双向的上下文来推断被掩盖的词，从而学习到深刻的语言表示。

class BertMLMHead(nn.Module):
    """一个简化的MLM预测头"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_output):
        # encoder_output: (batch_size, seq_len, d_model)
        # 我们只关心被mask的位置的输出
        # ... (此处省略了只选择mask位置的逻辑) ...
        # 最终，将BERT的输出通过一个线性层，预测整个词汇表上的概率
        return self.linear(encoder_output)

print("--- 3. Pre-training Task 1: Masked Language Model (MLM). ---")
print("Predicting masked tokens forces deep bidirectional understanding.")
print("-"*30)

# --- 4. 预训练任务二: 下一句预测 (Next Sentence Prediction, NSP) ---
# - **动机**: 让模型能够理解句子与句子之间的关系，这对于问答、自然语言推理等任务至关重要。
# - **方法**:
#   1. 准备训练样本时，50%的概率，句子B是句子A的真实下一句。
#   2. 50%的概率，句子B是来自语料库的一个随机句子。
#   3. **目标**: 训练模型去预测句子B是否是句子A的下一句（一个二分类问题）。
# - **实现**: 通常使用 `[CLS]` token对应的最终输出，接一个简单的分类器来完成这个任务。

class BertNSPHead(nn.Module):
    """一个简化的NSP预测头"""
    def __init__(self, d_model):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2) # 2分类: IsNext vs. NotNext

    def forward(self, cls_output):
        # cls_output: [CLS] token的最终输出, shape (batch_size, d_model)
        return self.classifier(cls_output)

print("--- 4. Pre-training Task 2: Next Sentence Prediction (NSP). ---")
print("Teaches the model to understand sentence relationships.")
print("-"*30)

# --- 5. BERT的微调 (Fine-tuning) ---
# BERT的强大之处在于，经过MLM和NSP任务预训练后，它成为了一个强大的、通用的语言理解引擎。
# 我们可以根据不同的下游任务，在BERT的输出之上添加一个简单的分类层，然后用我们的有标签数据进行端到端的微调。
#
# - **文本分类**: 取 `[CLS]` token的输出，接一个分类器。
# - **命名实体识别**: 取每个token的输出，分别对每个token进行分类。
# - **问答**: 将问题和段落一起输入，训练模型在段落中找到答案的起始和结束位置。

# 总结:
# 1. **架构**: BERT = 堆叠的Transformer编码器。
# 2. **核心思想**: 通过**掩码语言模型 (MLM)** 实现真正的**双向**上下文表示学习。
# 3. **辅助任务**: 通过**下一句预测 (NSP)** 学习句子间的关系。
# 4. **应用**: 作为一个通用的语言理解模型，BERT可以通过简单的微调，在各种NLP下游任务上取得SOTA性能。
