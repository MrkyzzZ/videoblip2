# VideoBLIP2 模型设计文档

## 1. 模型概览 (Model Overview)

本项目提出了一个基于 **BLIP-2** 和 **T5** 架构的多模态视听问答与描述生成模型。该模型旨在通过高效的参数微调（Parameter-Efficient Fine-Tuning），将预提取的视频和音频特征对齐到冻结的大型语言模型（LLM）空间中，从而实现高质量的视频内容理解与文本生成。

核心设计理念是利用 **Q-Former** 作为模态桥梁，将非文本模态（视频、音频）转换为 LLM 可理解的 Token 序列，并结合 **LoRA (Low-Rank Adaptation)** 技术对 T5 模型进行轻量级微调。

---

## 2. 详细架构设计 (Architecture Design)

模型整体架构可以分为三个主要层次：**特征编码层**、**模态对齐层 (Q-Former)** 和 **生成解码层 (LLM)**。

### 2.1 特征编码层 (Feature Encoding Layer)
模型不直接处理原始像素或波形，而是利用强大的预训练模型提取高层语义特征。

*   **视频特征 (Video Features)**:
    *   **来源**: 使用 **CLIP-ViT (Vision Transformer)** 提取的帧级特征。
    *   **维度**: `[Batch, Time_Video, 768]` (默认)。
    *   **作用**: 捕捉视频中的视觉对象、动作和场景信息。
*   **音频特征 (Audio Features)**:
    *   **来源**: 使用 **CLAP (Contrastive Language-Audio Pretraining)** 提取的音频特征。
    *   **维度**: `[Batch, Time_Audio, 512]` (默认)。
    *   **预处理**: 通过一个线性层 (`Audio Upscaler`) 将维度从 512 映射到 768，以便与 Q-Former 的隐藏层维度匹配。
    *   **作用**: 捕捉视频中的环境音、语音情感和背景音乐等听觉信息。

### 2.2 模态对齐层 (Modality Alignment Layer)
这是模型的核心组件，负责将视听特征压缩并翻译为 LLM 的输入格式。模型使用了两个独立的 **Q-Former** 模块。

*   **结构**: 基于 BERT 的 Transformer Encoder，包含 Cross-Attention 机制。
*   **初始化**: 权重初始化自预训练的 **BLIP-2** 模型，继承了其强大的图文对齐能力。
*   **双路 Q-Former 设计**:
    1.  **Video Q-Former**:
        *   **输入**: 可学习的 `Video Query Tokens` (数量: 32) + 文本提示 (Text Prompt)。
        *   **交互**: Query Tokens 通过 Cross-Attention 与 **视频特征** 交互。
        *   **输出**: 提取了视觉信息的 Query Embeddings。
    2.  **Audio Q-Former**:
        *   **输入**: 可学习的 `Audio Query Tokens` (数量: 32) + 文本提示 (Text Prompt)。
        *   **交互**: Query Tokens 通过 Cross-Attention 与 **音频特征** 交互。
        *   **输出**: 提取了听觉信息的 Query Embeddings。
*   **投影层 (Projection)**: Q-Former 的输出经过线性投影层 (`proj_video`, `proj_audio`)，变换为 T5 模型的输入维度 (`d_model`)。

### 2.3 生成解码层 (Generative Decoding Layer)
使用冻结的大型语言模型作为“大脑”，负责理解多模态信息并生成最终文本。

*   **基座模型**: **Flan-T5-Base** (Encoder-Decoder 架构)。
*   **输入序列构造**:
    *   将投影后的 **视频 Tokens**、**音频 Tokens** 和 **文本提示 Tokens** 拼接。
    *   序列形式: `[Video_Queries, Audio_Queries, Text_Prompt]`。
*   **微调策略 (LoRA)**:
    *   **冻结参数**: T5 的大部分参数保持冻结，保留预训练知识。
    *   **可训练参数**: 在 T5 的 Attention 层 (`q`, `v` 矩阵) 中注入 **LoRA (Low-Rank Adaptation)** 适配器。
    *   **配置**: Rank=16, Alpha=32, Dropout=0.05。
    *   **优势**: 极大减少了显存占用和训练时间，同时避免了灾难性遗忘。

---

## 3. 训练策略 (Training Strategy)

训练过程分为两个阶段，旨在逐步提升生成质量。

### 3.1 第一阶段：交叉熵预训练 (Cross-Entropy Pre-training)
*   **目标**: 让模型学会基本的视听理解和语法生成。
*   **数据**: MSR-VTT 原始训练集 (`train.json`)。
*   **损失函数**: 标准的语言模型损失 (Cross-Entropy Loss)。
*   **优化器**: AdamW，配合余弦学习率衰减 (Cosine Decay)。

### 3.2 第二阶段：自我批判序列训练 (SCST - Reinforcement Learning)
*   **目标**: 针对评估指标 (CIDEr) 进行直接优化，提升生成的丰富度和准确性。
*   **数据**: 精选的 Top-5 高质量描述子集 (`train_top5_detail.json`)。
*   **方法**: **Self-Critical Sequence Training (SCST)**。
    *   **Baseline**: 使用 Greedy Search 生成句子，计算 CIDEr 分数作为基准。
    *   **Sample**: 使用随机采样生成句子，计算 CIDEr 分数。
    *   **Reward**: 优化目标是最大化 (Sample Score - Baseline Score)。
*   **效果**: 鼓励模型生成更符合人类偏好、细节更丰富的描述。

---

## 4. 预处理与依赖 (Preprocessing & Dependencies)

*   **预训练权重**:
    *   **BERT**: `bert-base-uncased` (用于 Q-Former 的文本编码)。
    *   **T5**: `flan-t5-base` (用于文本生成)。
    *   **BLIP-2**: `blip2_pretrained_flant5xl.pth` (用于初始化 Q-Former)。
*   **特征提取**:
    *   视频特征需预先通过 ViT 提取并保存为 `.npy` 文件。
    *   音频特征需预先通过 CLAP 提取并保存为 `.npy` 文件。
*   **库依赖**: `PyTorch`, `Transformers`, `PEFT` (用于 LoRA), `pycocoevalcap` (用于 SCST 评估)。

---

## 5. 总结 (Summary)

该模型通过结合 **BLIP-2 的 Q-Former** 和 **LoRA 微调的 T5**，构建了一个轻量级但强大的多模态理解框架。双路 Q-Former 设计使其能同时处理视频和音频信息，而 SCST 训练策略则进一步提升了生成文本的质量和细节丰富度。
