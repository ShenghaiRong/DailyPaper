# LeWorldModel

LeWorldModel (LeWM) 是一个稳定、端到端的 Joint Embedding Predictive Architecture (JEPA)，可以直接从原始像素训练。

## 核心特点

- **极简损失函数**: 只有两个损失项
  - Next-embedding prediction loss
  - Gaussian-distributed latent 正则化
- **端到端训练**: 从原始像素训练，无需预训练 encoder
- **轻量级**: ~15M 参数，单 GPU 几小时训练完成
- **高效规划**: 比基于 foundation model 的 world model 快 48 倍

## 技术创新

证明 JEPA 不需要复杂的多项损失、EMA、预训练 encoder 或辅助监督就能稳定训练。

## 实验结果

- 在 2D 和 3D 控制任务上表现优秀
- Latent space 编码了有意义的物理结构（通过 probing 验证）
- Surprise evaluation 确认模型能检测物理上不可能的事件

## 相关链接

- [[JEPA]]
- [[DINO-WM]]
- [[PLDM]]
