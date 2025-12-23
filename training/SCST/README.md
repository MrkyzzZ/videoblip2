# SCST 训练入口

- 继承现有 CE 流水线，使用 `MultiModal_T5_Captioner` 作为基础策略网络。
- 奖励：CIDEr（简化版评估器放在 `evaluation/`）。
- 生成策略：beam search（默认 5 个样本）；通过 `SCSTTrainConfig` 调整。

## 快速开始

```bash
# SCST 微调
python -m torch.distributed.run --nproc_per_node=1 training/SCST/train_scst.py --gpu_ids 0
```

核心配置位于 `training/SCST/config_scst.py`，默认下调学习率与 batch size，并设置 SCST 专属的 beam/长度约束参数。
