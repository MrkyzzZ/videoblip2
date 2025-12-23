# SCST (Self-Critical Sequence Training) 训练代码提取

## 目录结构

```
code/
├── README.md                           # 本文档：SCST完整使用指南
├── models/
│   └── blip2_t5_scst.py               # BLIP2-T5 SCST核心模型代码
├── evaluation/
│   ├── cider.py                       # CIDEr评估器
│   └── cider_scorer.py                # CIDEr评分核心算法
├── configs/
│   ├── caption_scst_example.yaml      # SCST训练配置示例
│   └── caption_ce_example.yaml        # CE预训练配置示例（SCST前置步骤）
├── training/
│   └── train_scst.py                  # 训练入口脚本
└── docs/
    └── SCST_GUIDE.md                  # SCST详细原理与实现指南
```

---

## 🚀 快速开始

### 前置条件

SCST（Self-Critical Sequence Training）是一种强化学习方法，**必须在 CE（交叉熵）预训练模型的基础上进行**。

#### 步骤 1：使用 CE 训练基础模型

```bash
python -m torch.distributed.run --nproc_per_node=8 train.py \
    --cfg-path configs/caption_ce_example.yaml
```

#### 步骤 2：使用 SCST 微调

```bash
python -m torch.distributed.run --nproc_per_node=8 train.py \
    --cfg-path configs/caption_scst_example.yaml
```

---

## 📖 SCST 核心原理

### 什么是 SCST？

SCST 来自论文 ["Self-critical Sequence Training for Image Captioning"](https://arxiv.org/abs/1612.00563)。

传统 CE 训练使用教师强制（Teacher Forcing），直接优化预测 token 与 ground truth 的交叉熵损失。但这会导致：

- **Exposure Bias**：训练时模型看到的是 ground truth，推理时看到的是自己的预测
- **评估指标不一致**：训练优化 CE loss，但评估用 CIDEr/BLEU 等指标

SCST 通过强化学习直接优化 CIDEr 等评估指标，解决了这些问题。

### SCST 损失函数

```python
# 核心公式
loss = - log_prob(sampled_caption) * (reward - baseline)

# 其中：
# - sampled_caption: 采样生成的caption
# - reward: 采样caption的CIDEr分数
# - baseline: 贪婪解码caption的CIDEr分数（或beam search结果的平均分数）
```

### 为什么需要 CE 预训练？

1. **SCST 需要合理的初始策略**：如果模型生成的 caption 完全是垃圾，reward 都接近 0，梯度无法有效传播
2. **稳定性**：CE 预训练后，模型已经能生成合理的 caption，SCST 只需微调提升 CIDEr 分数
3. **收敛速度**：从头用 RL 训练需要大量样本，预训练后只需少量步骤即可收敛

---

## 💡 关键实现细节

### 1. 模型 forward 中的 SCST 分支

```python
def forward(self, samples):
    if not self.scst:  # 标准CE训练
        # ... 计算交叉熵损失
        return {"loss": ce_loss}
    else:  # SCST训练
        # 1. 使用beam search采样多个caption
        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # 2. 计算每个生成序列的log概率
        transition_scores = self.t5_model.compute_transition_scores(...)
        sequences_scores = transition_scores.sum(dim=1) / output_length

        # 3. 计算CIDEr reward
        reward = Cider().compute_score(caps_gt, caps_gen)

        # 4. 计算SCST损失
        reward_baseline = torch.mean(reward, dim=-1, keepdim=True)
        loss = - sequences_scores * (reward - reward_baseline)

        return {"loss": loss.mean()}
```

### 2. Reward 计算

使用 CIDEr 作为 reward 是最常见的选择，因为：

- CIDEr 与人类评价相关性高
- CIDEr 可微分（通过 REINFORCE 梯度估计）
- CIDEr 对长度有惩罚，避免生成过长或过短的 caption

### 3. Baseline 策略

本代码使用 **self-critical baseline**：使用同一 batch 内 beam search 结果的平均 CIDEr 分数作为 baseline。

```python
reward_baseline = torch.mean(reward, -1, keepdim=True)
loss = - sequences_scores * (reward - reward_baseline)
```

这样做的好处：

- 减小方差
- 不需要额外的 baseline 网络
- 同一个 batch 内的样本作为彼此的对比

---

## ⚙️ 超参数调优建议

### SCST 训练关键超参数

| 参数           | 建议值      | 说明                                |
| -------------- | ----------- | ----------------------------------- |
| `init_lr`      | 1e-5 ~ 1e-6 | SCST 学习率应比 CE 低 1-2 个数量级  |
| `batch_size`   | 2-4         | SCST 内存占用大（需要采样多个序列） |
| `beam_size`    | 5           | beam 数量，也是每个样本采样的序列数 |
| `max_iters`    | 15000-20000 | 根据数据集大小调整                  |
| `warmup_steps` | 1000        | 预热步数                            |

### 常见问题

#### Q: 训练时 CIDEr 分数剧烈波动？

A: 降低学习率，增加 warmup 步数

#### Q: 训练后 CIDEr 反而下降？

A: 检查 CE 预训练模型是否足够好（建议 CE 模型 CIDEr > 60 再进行 SCST）

#### Q: 显存不足？

A: 减小 batch_size，减小 beam_size，使用 gradient accumulation

---

## 📁 文件说明

### models/blip2_t5_scst.py

核心模型代码，包含：

- CE 训练分支
- SCST 训练分支
- CIDEr reward 计算
- 序列 log 概率计算

### evaluation/cider.py & cider_scorer.py

CIDEr 评估器实现，用于：

- 训练时计算 reward
- 验证/测试时评估模型

### configs/\*.yaml

训练配置文件示例：

- `caption_ce_example.yaml`: CE 预训练配置
- `caption_scst_example.yaml`: SCST 微调配置

---

## 🔧 如何将 SCST 应用到你自己的 LLM 项目

详细指南请参阅 [docs/SCST_GUIDE.md](docs/SCST_GUIDE.md)

简要步骤：

1. 确保你的 LLM 支持 `generate()` 方法且能返回 log 概率
2. 实现 CIDEr 计算（或其他 reward 函数）
3. 在 forward 中添加 SCST 分支
4. 先用 CE 训练出基础模型
5. 加载 CE 模型，切换到 SCST 模式继续训练

---

## 📚 参考文献

1. [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563)
2. [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
3. [CIDEr: Consensus-based Image Description Evaluation](https://arxiv.org/abs/1411.5726)
