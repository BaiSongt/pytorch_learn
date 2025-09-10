# 第三周：Transformer与预训练模型

## 学习目标
深入理解 Transformer 架构和预训练模型的原理，掌握使用和微调预训练模型的技能。

## 1. Transformer架构 (3天)

### Day 1: 注意力机制
- 注意力机制基础
  - Self-Attention原理
  - Multi-Head Attention
  - 注意力可视化
- Position Encoding
  - 位置编码原理
  - 相对位置编码
  - 位置嵌入方法

### Day 2: Transformer详解
- Encoder架构
  - 多层堆叠
  - Feed-Forward Networks
  - Layer Normalization
- Decoder架构
  - 掩码注意力
  - 交叉注意力
  - 自回归生成

### Day 3: 预训练模型原理
- BERT
  - 模型架构
  - 预训练任务
  - NSP和MLM
- GPT系列
  - 自回归语言模型
  - 模型扩展
  - InstructGPT原理

## 2. 预训练模型实践 (2天)

### Day 4: HuggingFace应用
- Transformers库使用
  - 模型加载
  - Tokenizer使用
  - Pipeline快速应用
- 下游任务微调
  - 文本分类
  - 命名实体识别
  - 问答系统

### Day 5: 高级应用
- 知识蒸馏
  - 教师-学生模型
  - 损失函数设计
  - 蒸馏策略
- 模型压缩
  - 量化技术
  - 剪枝方法
  - 模型蒸馏

## 实践项目
1. 中文情感分析（BERT微调）
2. 文本生成系统（GPT实现）
3. 专业领域模型蒸馏

## 代码结构
```
week3_transformer/
├── day1_attention/
├── day2_transformer/
├── day3_pretrained/
├── day4_huggingface/
└── day5_advanced/
```

## 学习资源
- Attention is All You Need论文
- HuggingFace文档
- BERT/GPT相关论文
- Transformer可视化教程
