"""SCST 训练器，基于现有 CE Trainer 扩展。"""

import logging
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from ..trainer import Trainer
from .evaluation import Cider, tokenize


class SCSTTrainer(Trainer):
    """Self-Critical Sequence Training (CIDEr reward)。"""

    def __init__(self, config, gpu_ids=None, enable_logging=True):
        super().__init__(config, gpu_ids=gpu_ids, enable_logging=enable_logging)
        self.beam_size = getattr(config, "SCST_BEAM_SIZE", 5)
        self.max_new_tokens = getattr(config, "SCST_MAX_NEW_TOKENS", 32)
        self.length_penalty = getattr(config, "SCST_LENGTH_PENALTY", 1.0)
        self.repetition_penalty = getattr(config, "SCST_REPETITION_PENALTY", 1.05)
        self.do_sample = getattr(config, "SCST_DO_SAMPLE", False)
        self.temperature = getattr(config, "SCST_TEMPERATURE", 1.0)
        self.pad_token_id = self.t5_tokenizer.pad_token_id

        # 只训练 T5 的 LoRA，冻结 Q-Former、投影和嵌入层
        self._freeze_non_lora()
        self._reset_optimizer_and_scheduler()

        # 加载 CE 预训练权重（显式指定优先，否则回退 SAVE_DIR）
        ckpt_dir = getattr(config, "CE_CKPT_DIR", None) or getattr(config, "SAVE_DIR", None)
        if ckpt_dir:
            loaded = self.load_checkpoint(ckpt_dir)
            if loaded:
                logging.info("已加载 CE 预训练权重: %s", ckpt_dir)
            else:
                raise RuntimeError(f"未能从 {ckpt_dir} 加载 CE 预训练权重，请确认路径/文件是否存在。")
        else:
            raise RuntimeError("未指定 CE_CKPT_DIR，不允许在未加载 CE 权重的情况下运行 SCST。")

        logging.info(
            "SCST 启动: beam=%d, max_new_tokens=%d, lr=%.2e",
            self.beam_size, self.max_new_tokens, self.optimizer.param_groups[0]['lr']
        )

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _encode_inputs(self, video_feats, audio_feats, video_mask, audio_mask, questions):
        core = self._unwrap_model()
        encoder_outputs, encoder_mask, _ = core.encode_for_t5(
            video_features=video_feats,
            audio_features=audio_feats,
            video_attention_mask=video_mask,
            audio_attention_mask=audio_mask,
            questions=questions,
        )
        return encoder_outputs, encoder_mask

    def _reset_optimizer_and_scheduler(self):
        """在冻结后重建优化器/调度器，仅保留可训练参数。"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = AdamW(trainable_params, lr=self.config.LEARNING_RATE)

        if self.use_cosine_decay:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps,
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps,
            )

    def _freeze_non_lora(self):
        """将非 LoRA 部件全部冻结，只训练 T5 的 LoRA 权重。"""
        core = self._unwrap_model()
        freeze_targets = [
            getattr(core, "video_qformer", None),
            getattr(core, "audio_qformer", None),
            getattr(core, "video_patch_downsampler", None),
            getattr(core, "audio_upscaler", None),
            getattr(core, "proj_video", None),
            getattr(core, "proj_audio", None),
            getattr(core, "bert_embedding", None),
        ]

        for module in freeze_targets:
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False

        # 冻结可学习 query tokens（Parameter 类型不在 module.parameters() 里单独出现）
        for param_name in ["video_query_tokens", "audio_query_tokens"]:
            if hasattr(core, param_name):
                getattr(core, param_name).requires_grad = False

        # 确保 T5 的 LoRA 权重仍可训练，若 base 权重被标记 requires_grad=True 也可按需继续冻结
        for name, p in core.t5_model.named_parameters():
            if "lora" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        # 记录当前可训练参数，便于核对
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        logging.info("可训练参数列表(仅 LoRA)：%s", ", ".join(trainable))

    def _compute_log_probs(self, encoder_outputs, encoder_mask, sequences):
        core = self._unwrap_model()
        decoder_input_ids = sequences[:, :-1]
        labels = sequences[:, 1:]

        # 如果 generate 产生了 batch*beam 的序列，则需要重复 encoder 表示和 mask
        b_beam = sequences.size(0)
        beam = self.beam_size
        b_enc = encoder_mask.size(0)
        if b_beam == b_enc * beam:
            enc_hidden = encoder_outputs.last_hidden_state.repeat_interleave(beam, dim=0)
            enc_mask = encoder_mask.repeat_interleave(beam, dim=0)
            encoder_outputs = BaseModelOutput(last_hidden_state=enc_hidden)
            encoder_mask = enc_mask

        outputs = core.t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = labels.ne(self.pad_token_id)
        seq_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return seq_log_probs

    def _scst_step(self, batch):
        video_feats, audio_feats, video_mask, audio_mask, questions, captions, all_refs, _ = batch
        batch_size = video_feats.size(0)
        encoder_outputs, encoder_mask = self._encode_inputs(
            video_feats, audio_feats, video_mask, audio_mask, questions
        )
        core = self._unwrap_model()

        # 生成候选序列（无梯度，更省内存）
        with torch.no_grad():
            encoder_outputs_detached = type(encoder_outputs)(
                last_hidden_state=encoder_outputs.last_hidden_state.detach()
            )
            generated = core.t5_model.generate(
                encoder_outputs=encoder_outputs_detached,
                attention_mask=encoder_mask,
                num_beams=self.beam_size,
                num_return_sequences=self.beam_size,
                max_new_tokens=self.max_new_tokens,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.do_sample,
                temperature=self.temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = generated.sequences
        seq_log_probs = self._compute_log_probs(encoder_outputs, encoder_mask, sequences)
        seq_log_probs = seq_log_probs.view(batch_size, self.beam_size)

        captions_generated = core.t5_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        captions_generated = [c.strip() for c in captions_generated]

        expanded_refs = []
        for refs in all_refs:
            expanded_refs.extend([refs for _ in range(self.beam_size)])

        refs_tok, cands_tok = tokenize(expanded_refs, captions_generated)
        _, rewards_np = Cider().compute_score(refs_tok, cands_tok)
        rewards = torch.from_numpy(rewards_np.astype(np.float32)).to(self.device)
        rewards = rewards.view(batch_size, self.beam_size)

        baseline = rewards.mean(dim=1, keepdim=True)
        loss = -(seq_log_probs) * (rewards - baseline)
        loss = loss.mean()

        # 调试输出：仅首批打印少量样例
        if self.global_step_debug < getattr(self.config, "DEBUG_LOG_SAMPLES", 0):
            log_n = min(self.config.DEBUG_LOG_SAMPLES, len(captions_generated))
            for i in range(log_n):
                logging.info(
                    "[DBG] sample=%d reward=%.3f text=%s", i, rewards.view(-1)[i].item(), captions_generated[i]
                )
            self.global_step_debug += 1

        return loss, rewards.mean().item()

    def train(self):
        global_step = 0
        self.global_step_debug = 0
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            np.random.shuffle(self.train_data)
            steps_in_epoch = 0
            progress_bar = tqdm(
                range(0, len(self.train_data), self.config.BATCH_SIZE),
                desc=f"SCST Epoch {epoch + 1}/{self.config.EPOCHS}"
            )

            for idx in progress_bar:
                batch_anns = self.train_data[idx: idx + self.config.BATCH_SIZE]
                batch = self._prepare_batch(batch_anns, training=True)
                if batch is None:
                    continue

                self.optimizer.zero_grad()
                loss, reward_mean = self._scst_step(batch)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                global_step += 1
                steps_in_epoch += 1

                lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", reward=f"{reward_mean:.3f}", lr=f"{lr:.2e}")

                # 调试：限制每个 epoch 的训练步数
                max_steps = getattr(self.config, "DEBUG_MAX_STEPS_PER_EPOCH", None)
                if max_steps is not None and steps_in_epoch >= max_steps:
                    logging.info("DEBUG: 达到本 epoch 最大步数限制 %s，提前结束本 epoch。", max_steps)
                    break

            self.evaluate(epoch, save_best=True)
