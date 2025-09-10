# 第四周：生成式AI与工程实践

## 学习目标
掌握最新的生成式AI技术，并学习工程化部署的最佳实践。

## 1. 生成式AI (3天)

### Day 1: Diffusion模型
- 扩散模型基础
  - DDPM原理
  - 噪声调度
  - 采样策略
- 架构设计
  - U-Net结构
  - 条件生成
  - 注意力机制

### Day 2: Stable Diffusion
- 模型组件
  - VAE编码器
  - U-Net主干
  - 文本编码器
- 实践应用
  - 文本到图像生成
  - 图像编辑
  - LoRA微调

### Day 3: LLM与RLHF
- LLM推理优化
  - KV Cache
  - Beam Search
  - 采样策略
- RLHF实现
  - PPO算法
  - 奖励模型
  - 人类反馈

## 2. 工程实践 (2天)

### Day 4: 分布式训练
- 并行训练策略
  - DataParallel
  - DistributedDataParallel
  - 模型并行
- 性能优化
  - 混合精度训练
  - 梯度累积
  - 梯度检查点

### Day 5: 部署优化
- 模型优化
  - 量化部署
  - 剪枝优化
  - TensorRT加速
- 服务部署
  - Flask API
  - FastAPI
  - 生产环境配置

## 实践项目
1. 图像生成系统（Stable Diffusion）
2. 对话机器人（LLM微调）
3. 高性能服务部署

## 代码结构
```
week4_genai/
├── day1_diffusion/
├── day2_stable_diffusion/
├── day3_llm_rlhf/
├── day4_distributed/
└── day5_deployment/
```

## 学习资源
- Stable Diffusion论文
- RLHF相关论文
- PyTorch分布式训练文档
- 部署最佳实践指南

## 扩展学习
- 模型量化技术
- 大规模训练优化
- 云平台部署
- 监控与日志
