
import torch
import torch.nn as nn

# --- 前言 ---
# 简单的整数索引无法表达词与词之间的关系（例如，索引5和索引6的两个词可能毫无关系）。
# One-Hot编码虽然能解决这个问题，但会导致极度高维和稀疏的向量，计算效率低下且无法捕捉语义。
# 词嵌入 (Word Embeddings) 将每个词（或token）表示为一个低维的、稠密的浮点数向量。
# 这些向量是可学习的，在训练过程中，模型会调整它们，使得意思相近的词在向量空间中的位置也相近。

# --- 1. 准备输入数据 ---
# 我们使用上一个脚本中得到的、经过填充的整数序列作为输入。
# (batch_size=4, sequence_length=7)
# 0是<pad>的索引
padded_sequences = torch.tensor([
    [ 3,  4,  2,  5,  6,  7,  8], # "i love pytorch it is amazing"
    [ 2,  5,  9, 10, 11, 12,  0], # "pytorch is the best deep learning framework"
    [ 3, 13, 14, 15, 16,  2,  0], # "i am learning nlp with pytorch"
    [15,  5, 17, 18,  7,  0,  0]  # "nlp is fun and amazing"
], dtype=torch.long)

print("---" + "1. Input: Padded Numericalized Sequences" + "---")
print(padded_sequences)
print(f"Input shape: {padded_sequences.shape}")
print("---" + "-"*30)

# --- 2. `nn.Embedding` 层 ---
# `nn.Embedding` 是PyTorch中实现词嵌入的核心层。
# 它本质上是一个巨大的查找表（lookup table），存储了每个词的向量。

# - `num_embeddings`: 词汇表的大小。即总共有多少个独立的词（或token）。
# - `embedding_dim`: 嵌入向量的维度。你希望用一个多长的向量来表示一个词。这是一个超参数，通常取值在50到300之间。
# - `padding_idx`: 可选参数。如果提供，该索引对应的向量在训练过程中将不会被更新，并且在反向传播时其梯度始终为0。
#   这对于处理填充token非常重要。

vocab_size = 20 # 假设我们的词汇表大小为20 (包括<pad>和<unk>)
embedding_dim = 5 # 为了方便演示，我们使用一个5维的嵌入向量
padding_idx = 0 # <pad> token的索引是0

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size, 
    embedding_dim=embedding_dim, 
    padding_idx=padding_idx
)

print("---" + "2. nn.Embedding Layer" + "---")
print(embedding_layer)

# 我们可以查看它的权重，这就是那个查找表
# 形状为 (vocab_size, embedding_dim)
print(f"Shape of embedding weights: {embedding_layer.weight.shape}")
# 我们可以看到，padding_idx=0对应的向量梯度是不会被计算的
# print(embedding_layer.weight)
print("---" + "-"*30)

# --- 3. 获取词嵌入向量 ---
# 将整数序列张量喂给embedding_layer，它会返回对应的嵌入向量张量。

embedded_sequences = embedding_layer(padded_sequences)

print("---" + "3. Looking up Embeddings" + "---")
print("Original input shape: ", padded_sequences.shape)
# 输出形状: (batch_size, sequence_length, embedding_dim)
print("Shape after embedding lookup: ", embedded_sequences.shape)

# 我们可以检查第一个句子的第一个词（索引为3）的向量
print("\nVector for the first word ('i', index 3) in the first sentence:")
print(embedded_sequences[0, 0, :])
# 它应该与直接从权重矩阵中查找索引为3的行是相同的
print("\nVector for index 3 in the embedding weight matrix:")
print(embedding_layer.weight[3])

# 同样，我们可以检查一个填充token的向量
print("\nVector for a <pad> token (index 0):")
print(embedded_sequences[1, 6, :]) # 第二个句子的最后一个词是<pad>
print("---" + "-"*30)

# --- 4. 训练和预训练词嵌入 ---
# - **从零开始训练**: 
#   `nn.Embedding`层的权重是模型的可训练参数。在训练过程中，损失函数的梯度会通过反向传播
#   一直传到embedding层，并更新这些词向量。模型会为了完成最终任务（如文本分类或情感分析）
#   而自动地学习到有意义的词表示。
#
# - **使用预训练词嵌入 (Pre-trained Embeddings)**:
#   我们可以加载在大规模语料库（如维基百科）上预训练好的词向量（如 GloVe, Word2Vec, FastText）。
#   这是一种强大的迁移学习，因为这些向量已经包含了丰富的语义信息。
#   **步骤**:
#   1. 加载预训练的词向量文件。
#   2. 创建一个与预训练向量维度和词汇表都匹配的`nn.Embedding`层。
#   3. 将预训练的向量加载到`nn.Embedding`层的权重矩阵中 (`embedding_layer.weight.data.copy_(pretrained_weights)`)
#   4. 在训练时，你可以选择“冻结”这些向量（不更新它们），或者用一个很小的学习率来“微调”它们。

# 总结:
# 1. 词嵌入是将词表示为稠密、低维、可学习的向量的关键技术。
# 2. `nn.Embedding` 层是PyTorch中实现此功能的标准工具，它作为一个可训练的查找表工作。
# 3. 将整数序列输入`nn.Embedding`层，即可得到形状为 (batch, seq_len, embedding_dim) 的嵌入向量，这通常是RNN或Transformer模型的输入。
# 4. 使用在大规模语料上预训练的词嵌入是一种非常有效的提升模型性能的方法。

