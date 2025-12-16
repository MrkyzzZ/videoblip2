# Video Captioning Incremental Plan (Baseline → Qwen)

目标：在不大改动代码的前提下，分阶段提升 caption 质量，并平滑切换文本骨干到 Qwen。

## 阶段 0：基线健康检查（无代码改动）
- 确认数据完整：`VIDEO_FEATURES_DIR`/`AUDIO_FEATURES_DIR` 下不存在缺失文件；过滤标注中缺特征的样本。
- 训练/推理一致：`DEFAULT_QUESTION_PROMPT` 保持不变；`max_target_length` 与生成 `max_length` 对齐（32 如现配置）。
- 预训练权重：确保 `saved_models/best_lora_adapters` 与 `best_other_params.pth` 可正常加载；若加载报适配器同名冲突，改用 `set_peft_model_state_dict` 覆盖默认适配器即可（局部改动，后续阶段执行）。

## 阶段 1：XE 训练微调（小步修改）
- 在 `TrainConfig` 中开启/检查：
  - `USE_COSINE_DECAY=True`（已开）; `WARMUP_RATIO` 保持 0.1。
  - 可加 `LABEL_SMOOTHING=0.1`（需在 `trainer.py` 的 CE 调用中增加）以减弱过拟合常见词。
- 生成侧小改动（`models/multimodal_t5.py`）：
  - `generate` 时追加 `no_repeat_ngram_size=3`，`min_length=6`，`repetition_penalty=1.1`，防止空泛重复。
  - 保持 `num_beams=1` 作为基线，后续 SCST 再调整。
- 日志：保留 `sample_text` 输出，检查是否出现高频“people are ...”模式以早停。

## 阶段 2：修复/启用 SCST（强化学习）
- 数据：`SCST_ANNOTATIONS_PATH` 已指向 top5 captions，开启时将 `ENABLE_SCST=True`。
- 代码小改动（局部）：
  - `_prepare_batch` 在 SCST 模式下不要随机抽一条 caption，需保留全部参考作为 `all_refs`（目前训练态随机，会削弱 CIDEr 奖励）。
  - `load_checkpoint`：用 `set_peft_model_state_dict` 覆盖现有 `default` 适配器，避免重复加载失败；若找不到 safetensors，则退回 `load_adapter`。
- SCST 超参：
  - `SCST_LEARNING_RATE=5e-6`（已设），`SCST_BATCH_SIZE=16`；可加梯度累积以稳住显存。
  - Baseline 解码：`num_beams=3`；Sample 解码：`do_sample=True, top_p=0.9, temperature=0.7, top_k=50`。
  - 奖励：CIDEr 为主，可混合 0.7*CIDEr + 0.3*BLEU-4（可在 reward 部分线性组合）。
  - 长度归一化：计算 sample log-prob 时除以有效 token 数，防止奖励偏向短句。

## 阶段 3：解码与评测细节（无大改）
- 推理时使用与 SCST 一致的生成参数（min_length、no_repeat_ngram_size、temperature/top_p 或 beam），保证训练-评估一致。
- 对动画/体育/真人类别做分组评测（logs/test 下额外保存分组样本），定位语义漂移来源。
- 如果 Rouge-L/METEOR 不提升而 CIDEr 提升，适当提高 `max_length` 至 48，并在 SCST 中加入长度惩罚系数（如 -0.01*|len-32|）。

## 阶段 4：平滑切换文本骨干到 Qwen（最小侵入式）
- 选择模型：`Qwen2.5-1.8B-Instruct` 或 `Qwen2.5-7B-Instruct`（按显存）。准备本地路径，如 `/root/autodl-tmp/videoblip2/pretrained/qwen2.5-7b-instruct`。
- 兼容改动（集中在 `multimodal_t5.py` 等）：
  - 将 T5 tokenizer/model 替换为对应 Qwen tokenizer/model（`AutoTokenizer`, `AutoModelForCausalLM`）。
  - 统一维度：若 Qwen hidden size 与当前投影不符，调整 `proj_video/proj_audio` 输出维度与 Qwen embedding dim 对齐。
  - 解码接口从 `generate` 的 seq2seq 变为 causal LM：保持 encoder 输出作为前缀 embedding，或改为构造前缀 prompt（最小改动：将 encoder输出映射到 Qwen 的输入嵌入并拼接文本 token）。
  - LoRA 目标模块更新为 Qwen 的注意力/FFN 名称（如 `q_proj`, `k_proj`, `v_proj`, `o_proj`）。
- 训练策略：
  - 先进行 3–5 epoch XE 微调（lr≈1e-5，LoRA rank 8–16），确认收敛后再开启 SCST（lr≈5e-6）。
  - 继续使用现有 Q-Former/特征，不改视频/音频侧，保证改动局部。

## 阶段 5：验证与对比
- 统一评测：MSRVTT val/test 上记录 BLEU-1/4、ROUGE-L、CIDEr，比较 T5 基线 vs Qwen 版本。
- 误差分析：从 `sample_text` 抽取分类错误（动画→真人、体育类别混淆），针对性做小批 hard-neg 再训练 1–2 epoch。
- 若 Qwen 显存紧张，可尝试 4-bit quant + LoRA（QLoRA），保持其他流程不变。

## 快速检查清单
- [ ] `ENABLE_SCST` 切换后能正常加载旧 checkpoint（LoRA 覆盖无报错）。
- [ ] SCST 奖励确实使用多参考 caption（非随机单条）。
- [ ] 生成参数在训练/推理一致（min_length、no_repeat_ngram_size、beam/sample 设置）。
- [ ] Qwen 版本的投影维度和 LoRA 目标名已对齐，训练能跑通首个 epoch。
