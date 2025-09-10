# Day 3: 预训练模型原理

## 学习目标
理解BERT和GPT等预训练模型的原理和应用

## 详细内容

### 1. BERT基础
```python
01_bert_basics.py
```
- 模型架构
- 预训练任务
- MLM实现
- NSP任务

### 2. GPT原理
```python
02_gpt_architecture.py
```
- 自回归语言模型
- GPT架构特点
- 训练目标
- 生成策略

### 3. 预训练过程
```python
03_pretraining.py
```
- 数据准备
- 训练策略
- 损失计算
- 训练优化

### 4. 模型适应
```python
04_model_adaptation.py
```
- 领域适应
- 任务适应
- 增量预训练
- 知识注入

### 5. 评估与分析
```python
05_evaluation.py
```
- 模型评估
- 错误分析
- 表示学习
- 知识探测

## 实践项目
1. BERT预训练
   - 小规模语料预训练
   - MLM任务实现
   - 模型评估

2. GPT文本生成
   - 模型实现
   - 生成策略
   - 质量评估

## 作业
1. 实现简化版BERT
2. 完成预训练任务
3. 分析模型表现

## 参考资源
- BERT/GPT论文
- 预训练最佳实践
- PyTorch实现
- 评估工具
