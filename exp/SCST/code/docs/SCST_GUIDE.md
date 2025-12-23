# SCST (Self-Critical Sequence Training) 详细实现指南

## 目录

1. [SCST 原理深度解析](#1-scst原理深度解析)
2. [为什么需要 CE 预训练](#2-为什么需要ce预训练)
3. [将 SCST 应用于你自己的 LLM 项目](#3-将scst应用于你自己的llm项目)
4. [完整代码实现步骤](#4-完整代码实现步骤)
5. [常见问题与调优技巧](#5-常见问题与调优技巧)
6. [数学推导](#6-数学推导)

---

## 1. SCST 原理深度解析

### 1.1 问题背景

传统的图像/视频描述生成使用**交叉熵损失（Cross-Entropy Loss）**进行训练：

```python
loss = CrossEntropyLoss(predicted_logits, ground_truth_tokens)
```

这种方法存在两个主要问题：

1. **Exposure Bias（曝光偏差）**

   - 训练时：模型看到的是 ground truth token 作为上文
   - 推理时：模型看到的是自己之前预测的 token
   - 这种训练-推理的不一致导致错误累积

2. **评估指标不一致**
   - 训练优化的是 token 级别的 CE loss
   - 评估使用的是句子级别的 CIDEr、BLEU 等指标
   - 优化目标与评估目标不一致

### 1.2 SCST 解决方案

SCST 使用强化学习（Reinforcement Learning）直接优化评估指标：

```
目标：最大化 E[reward(sampled_caption)]
```

核心思想是将 captioning 模型视为一个**策略（policy）**，生成的 caption 视为**动作（action）**，CIDEr 分数视为**奖励（reward）**。

### 1.3 REINFORCE 算法

SCST 使用 REINFORCE 算法估计梯度：

```
∇θ J(θ) ≈ (r(w^s) - b) ∇θ log p(w^s | I; θ)

其中：
- w^s: 采样生成的caption
- r(w^s): caption的CIDEr reward
- b: baseline（用于减小方差）
- p(w^s | I; θ): 在给定图像I下生成w^s的概率
```

### 1.4 Self-Critical Baseline

SCST 的"Self-Critical"体现在 baseline 的选择：

```python
# 使用beam search生成多个候选
candidates = model.beam_search(image, num_beams=5)

# 计算每个候选的reward
rewards = [CIDEr(cand, ground_truth) for cand in candidates]

# Baseline = 所有候选的平均reward
baseline = mean(rewards)

# Self-Critical Loss
loss = -log_prob(candidates) * (rewards - baseline)
```

**为什么叫"Self-Critical"？**

- 模型用自己的输出（beam search 结果）作为 baseline
- 如果采样结果比自己的平均水平好，增加其概率
- 如果采样结果比自己的平均水平差，减少其概率
- 模型在"批评"自己的输出

---

## 2. 为什么需要 CE 预训练

### 2.1 冷启动问题

如果直接从随机初始化开始 SCST 训练：

```
问题：
1. 模型生成的caption完全是乱码
2. 所有caption的CIDEr分数都接近0
3. reward - baseline ≈ 0，梯度几乎为0
4. 模型无法学习任何有用信息
```

### 2.2 CE 预训练的作用

CE 预训练提供了一个"合理的初始策略"：

```
CE预训练后：
1. 模型能生成语法正确的句子
2. 生成的caption与图像有一定相关性
3. CIDEr分数分布合理（不全是0）
4. SCST可以在此基础上微调优化
```

### 2.3 训练流程

```
Step 1: CE预训练
├── 使用标准CrossEntropyLoss
├── 训练直到验证集CIDEr稳定（建议>60）
└── 保存checkpoint

Step 2: SCST微调
├── 加载CE预训练的checkpoint
├── 切换到SCST模式（scst=True）
├── 使用较小的学习率
└── 继续训练优化CIDEr
```

### 2.4 预训练模型质量要求

```python
# 建议的CE预训练模型指标
if ce_model_cider > 60:
    print("可以进行SCST训练")
elif ce_model_cider > 40:
    print("可以尝试，但可能效果不佳")
else:
    print("建议继续CE训练，模型尚未ready")
```

---

## 3. 将 SCST 应用于你自己的 LLM 项目

### 3.1 前置要求

你的 LLM 需要支持以下功能：

```python
# 1. 生成方法需要返回log概率
outputs = model.generate(
    inputs,
    return_dict_in_generate=True,
    output_scores=True,  # 关键：返回每步的score
)

# 2. 能够计算序列的log概率
# 对于HuggingFace模型，可以使用compute_transition_scores
log_probs = model.compute_transition_scores(
    outputs.sequences,
    outputs.scores,
    outputs.beam_indices,
)
```

### 3.2 核心修改步骤

#### Step 1: 添加 SCST 开关

```python
class YourLLM(nn.Module):
    def __init__(self, scst=False, beam_size=5, ...):
        self.scst = scst
        self.beam_size = beam_size
```

#### Step 2: 实现 SCST forward

```python
def forward(self, samples):
    if not self.scst:
        return self._forward_ce(samples)
    else:
        return self._forward_scst(samples)
```

#### Step 3: 实现 reward 计算

```python
def compute_reward(self, generated_captions, gt_captions):
    """计算CIDEr reward"""
    from evaluation import Cider, tokenize

    # Tokenize
    refs_tok, cands_tok = tokenize(gt_captions, generated_captions)

    # 计算CIDEr
    _, scores = Cider().compute_score(refs_tok, cands_tok)

    return torch.from_numpy(scores).to(self.device)
```

#### Step 4: 实现 SCST 损失

```python
def _forward_scst(self, samples):
    # 1. 编码输入
    inputs = self.encode(samples)

    # 2. 生成多个候选
    outputs = self.model.generate(
        inputs,
        num_beams=self.beam_size,
        num_return_sequences=self.beam_size,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # 3. 计算log概率
    transition_scores = self.model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        outputs.beam_indices,
    )
    log_probs = transition_scores.sum(dim=1) / (transition_scores < 0).sum(dim=1)
    log_probs = log_probs.view(batch_size, self.beam_size)

    # 4. 解码并计算reward
    captions = self.decode(outputs.sequences)
    reward = self.compute_reward(captions, samples["ground_truth"])
    reward = reward.view(batch_size, self.beam_size)

    # 5. 计算SCST损失
    baseline = reward.mean(dim=-1, keepdim=True)
    loss = -log_probs * (reward - baseline)

    return {"loss": loss.mean()}
```

### 3.3 完整示例：为 GPT-2 添加 SCST

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from evaluation import Cider, tokenize
import itertools
import numpy as np


class GPT2WithSCST(nn.Module):
    """示例：为GPT-2添加SCST训练能力"""

    def __init__(self, scst=False, beam_size=5):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scst = scst
        self.beam_size = beam_size

    def forward(self, input_ids, attention_mask, labels=None, ground_truth=None):
        if not self.scst:
            # 标准CE训练
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return {"loss": outputs.loss}
        else:
            # SCST训练
            return self._forward_scst(input_ids, attention_mask, ground_truth)

    def _forward_scst(self, input_ids, attention_mask, ground_truth):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. 使用beam search生成多个候选
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=self.beam_size,
            num_return_sequences=self.beam_size,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # 2. 计算log概率
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            outputs.beam_indices,
            normalize_logits=False,
        )
        output_length = (transition_scores < 0).sum(dim=1)
        log_probs = transition_scores.sum(dim=1) / output_length
        log_probs = log_probs.view(batch_size, self.beam_size)

        # 3. 解码生成的文本
        generated_texts = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )

        # 4. 扩展ground truth以匹配生成数量
        gt_expanded = list(itertools.chain(
            *([gt] * self.beam_size for gt in ground_truth)
        ))
        gt_expanded = [[gt] for gt in gt_expanded]

        # 5. 计算CIDEr reward
        refs_tok, cands_tok = tokenize(gt_expanded, generated_texts)
        _, scores = Cider().compute_score(refs_tok, cands_tok)
        reward = torch.from_numpy(scores.astype(np.float32)).to(device)
        reward = reward.view(batch_size, self.beam_size)

        # 6. 计算SCST损失
        baseline = reward.mean(dim=-1, keepdim=True)
        loss = -log_probs * (reward - baseline)

        return {"loss": loss.mean()}


# 使用示例
if __name__ == "__main__":
    # CE预训练
    model_ce = GPT2WithSCST(scst=False)
    # ... 训练代码 ...
    torch.save(model_ce.state_dict(), "gpt2_ce_pretrained.pth")

    # SCST微调
    model_scst = GPT2WithSCST(scst=True, beam_size=5)
    model_scst.load_state_dict(torch.load("gpt2_ce_pretrained.pth"))
    # ... 继续训练 ...
```

---

## 4. 完整代码实现步骤

### 4.1 项目结构

```
your_project/
├── models/
│   └── your_model_scst.py    # 添加SCST的模型
├── evaluation/
│   ├── __init__.py
│   ├── cider.py              # CIDEr评估器
│   ├── cider_scorer.py       # CIDEr核心算法
│   └── ptbtokenizer.py       # 文本tokenizer
├── configs/
│   ├── ce_train.yaml         # CE训练配置
│   └── scst_train.yaml       # SCST训练配置
├── train.py                   # 训练脚本
└── requirements.txt
```

### 4.2 关键代码清单

1. **SCST Forward（必须实现）**

   ```python
   def _forward_scst(self, samples):
       # 见上面的完整实现
       pass
   ```

2. **Log 概率计算（必须实现）**

   ```python
   # HuggingFace模型使用compute_transition_scores
   # 其他框架需要自行实现
   ```

3. **Reward 计算（必须实现）**

   ```python
   def compute_reward(self, generated, ground_truth):
       # 使用CIDEr或其他指标
       pass
   ```

4. **配置文件（必须正确设置）**
   ```yaml
   model:
     scst: True # 启用SCST
     pretrained: "path/to/ce_checkpoint.pth" # CE预训练权重
   ```

---

## 5. 常见问题与调优技巧

### 5.1 训练不稳定

**问题**：CIDEr 分数剧烈波动

**解决方案**：

```yaml
# 降低学习率
init_lr: 1e-6  # 从1e-5降到1e-6

# 增加预热步数
warmup_steps: 2000  # 从1000增加到2000

# 使用梯度裁剪（在代码中添加）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5.2 显存不足

**问题**：SCST 需要更多显存

**解决方案**：

```yaml
# 减小batch size
batch_size_train: 1

# 使用梯度累积
accum_grad_iters: 4 # 等效batch_size = 1 * 4 = 4

# 减小beam size
# 在模型中修改beam_size参数
beam_size: 3 # 从5减到3
```

### 5.3 CIDEr 不提升

**问题**：SCST 训练后 CIDEr 没有提升甚至下降

**可能原因与解决方案**：

1. **CE 模型不够好**

   ```
   解决：继续CE训练，直到CIDEr > 60
   ```

2. **学习率太大**

   ```yaml
   init_lr: 1e-7 # 尝试更小的学习率
   ```

3. **Beam size 太小**
   ```python
   beam_size: 5  # 至少使用5个beam
   ```

### 5.4 超参数推荐

| 参数         | CE 训练       | SCST 训练     | 说明                  |
| ------------ | ------------- | ------------- | --------------------- |
| init_lr      | 1e-4 ~ 5e-5   | 1e-5 ~ 1e-6   | SCST 用更小的学习率   |
| batch_size   | 32 ~ 64       | 2 ~ 4         | SCST 显存占用大       |
| warmup_steps | 500 ~ 1000    | 1000 ~ 2000   | SCST 需要更多预热     |
| beam_size    | N/A           | 5             | beam 数量             |
| max_iters    | 10000 ~ 15000 | 15000 ~ 20000 | SCST 可能需要更多步骤 |

---

## 6. 数学推导

### 6.1 Policy Gradient

目标函数：
$$J(\theta) = \mathbb{E}_{w^s \sim p_\theta}[r(w^s)]$$

使用 REINFORCE 估计梯度：
$$\nabla_\theta J(\theta) = \mathbb{E}_{w^s \sim p_\theta}[r(w^s) \nabla_\theta \log p_\theta(w^s)]$$

### 6.2 Baseline 减小方差

引入 baseline $b$不改变梯度期望，但减小方差：
$$\nabla_\theta J(\theta) = \mathbb{E}_{w^s \sim p_\theta}[(r(w^s) - b) \nabla_\theta \log p_\theta(w^s)]$$

最优 baseline 是 reward 的期望，但计算困难。Self-Critical 使用同一 batch 的平均 reward 作为近似。

### 6.3 SCST 损失函数

$$L_{SCST} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(w_i^s) \cdot (r(w_i^s) - \bar{r})$$

其中：

- $N$: batch 中的样本数 × beam 数
- $w_i^s$: 第$i$个采样的 caption
- $r(w_i^s)$: 对应的 CIDEr 分数
- $\bar{r}$: 同一图像所有候选的平均 CIDEr

---

## 参考文献

1. Rennie, S. J., et al. "Self-critical sequence training for image captioning." CVPR 2017. [[Paper]](https://arxiv.org/abs/1612.00563)
2. Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023. [[Paper]](https://arxiv.org/abs/2301.12597)
3. Vedantam, R., et al. "CIDEr: Consensus-based Image Description Evaluation." CVPR 2015. [[Paper]](https://arxiv.org/abs/1411.5726)
