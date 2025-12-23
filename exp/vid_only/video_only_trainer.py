"""
Video-only trainer that mirrors training/trainer.py but drops audio features
and writes logs/checkpoints under exp/.
"""

import os
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging
import random
from torch.optim import AdamW
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import PeftModel, set_peft_model_state_dict

from exp.vid_only.video_only_model import VideoOnly_T5_Captioner
from exp.vid_only.video_only_config import VideoOnlyTrainConfig


class VideoOnlyTrainer:
    """多模态 Caption 训练器（仅视频分支）。"""

    def __init__(self, config=None, gpu_ids=None, enable_logging=True):
        self.config = config or VideoOnlyTrainConfig()
        self.enable_logging = enable_logging

        if gpu_ids is None:
            num_cuda = torch.cuda.device_count()
            if num_cuda >= 4:
                gpu_ids = [0, 1, 2, 3]
            elif num_cuda > 0:
                gpu_ids = list(range(num_cuda))
            else:
                gpu_ids = []

        self.gpu_ids = gpu_ids if gpu_ids else []
        if torch.cuda.is_available() and self.gpu_ids:
            primary_gpu = self.gpu_ids[0]
            self.device = torch.device(f"cuda:{primary_gpu}")
        else:
            self.device = torch.device("cpu")
            self.gpu_ids = []

        # 日志目录迁移到 exp/
        if self.enable_logging:
            os.makedirs(self.config.LOG_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_log_dir = os.path.join(self.config.LOG_DIR, timestamp)
            os.makedirs(self.run_log_dir, exist_ok=True)
            self.sample_text_dir = os.path.join(self.run_log_dir, "test")
            os.makedirs(self.sample_text_dir, exist_ok=True)
            self.best_model_dir = os.path.join(self.run_log_dir, "best_model")
            os.makedirs(self.best_model_dir, exist_ok=True)
            gpu_tag = "_".join(str(g) for g in self.gpu_ids) if self.gpu_ids else "cpu"
            self.log_file_path = os.path.join(self.run_log_dir, f"training_log_gpu{gpu_tag}.txt")
            logging.info(f"本次训练的日志将记录在: {self.log_file_path}")
            self.log_hyperparameters()
        else:
            self.run_log_dir = None
            self.sample_text_dir = None
            self.best_model_dir = None
            self.log_file_path = None

        # 数据加载（不再检查音频特征，从而覆盖完整的 10k 样本）
        self.train_data = json.load(open(self.config.TRAIN_ANNOTATIONS_PATH, "r"))

        test_ann_path = self.config.TEST_ANNOTATIONS_PATH
        test_videodatainfo_path = os.path.join(os.path.dirname(test_ann_path), "test_videodatainfo.json")
        if os.path.exists(test_videodatainfo_path):
            logging.info(f"检测到完整的测试集标注文件: {test_videodatainfo_path}，正在加载并转换格式...")

            original_test_data = json.load(open(test_ann_path, "r"))
            valid_test_video_ids = {str(item.get("video_id") or item.get("id") or item.get("video")) for item in original_test_data if item.get("video_id") or item.get("id") or item.get("video")}

            raw_test_data = json.load(open(test_videodatainfo_path, "r"))
            if isinstance(raw_test_data, dict) and "sentences" in raw_test_data:
                raw_test_data = raw_test_data["sentences"]

            video_to_captions = {}
            for item in raw_test_data:
                vid = str(item.get("video_id"))
                cap = item.get("caption")
                if vid not in valid_test_video_ids:
                    continue
                if not vid or not cap:
                    continue
                video_to_captions.setdefault(vid, []).append(cap)

            self.test_anns = [
                {"video_id": vid, "caption": caps, "video": f"{vid}.mp4"}
                for vid, caps in video_to_captions.items()
            ]
            logging.info(f"成功加载并过滤测试集，共 {len(self.test_anns)} 个视频样本 (已过滤掉非测试集视频)。")
        else:
            logging.warning(f"未找到 {test_videodatainfo_path}，回退到使用 {test_ann_path}")
            self.test_anns = json.load(open(test_ann_path, "r"))

        self.max_target_length = getattr(self.config, "MAX_TARGET_LENGTH", 32)
        self.default_question_prompt = getattr(self.config, "DEFAULT_QUESTION_PROMPT", "Describe the video content in detail.")

        base_model = VideoOnly_T5_Captioner(
            num_classes=0,
            t5_model_path=self.config.T5_MODEL_PATH,
            qformer_num_layers=self.config.QFORMER_NUM_LAYERS,
            num_query_token=self.config.NUM_QUERY_TOKEN,
            lora_r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            blip2_model_path=self.config.BLIP2_MODEL_PATH,
            clip_patch_dim=getattr(self.config, "CLIP_PATCH_DIM", 1024),
            qformer_input_dim=getattr(self.config, "QFORMER_INPUT_DIM", 768),
            bert_tokenizer_name_or_path=getattr(self.config, "BERT_TOKENIZER_PATH", "bert-base-uncased"),
        ).to(self.device)

        self.bert_tokenizer = base_model.bert_tokenizer
        self.t5_tokenizer = base_model.t5_tokenizer

        if len(self.gpu_ids) > 1:
            logging.info(f"启用 DataParallel，多卡: {self.gpu_ids}")
            self.model = nn.DataParallel(base_model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
        else:
            self.model = base_model

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.LEARNING_RATE)

        self.use_cosine_decay = getattr(self.config, "USE_COSINE_DECAY", False)
        steps_per_epoch = max(1, math.ceil(len(self.train_data) / self.config.BATCH_SIZE))
        self.total_training_steps = steps_per_epoch * self.config.EPOCHS
        self.warmup_steps = int(self.config.WARMUP_RATIO * self.total_training_steps)
        if self.use_cosine_decay:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_training_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_training_steps
            )

        self.best_cider = float("-inf")
        self._nltk_ready = False
        logging.info("Video-only Trainer 初始化完成。")
        logging.info(f"总训练步数: {self.total_training_steps}, Warmup 步数: {self.warmup_steps}")

    def log_hyperparameters(self):
        log_message = "--- 实验配置 ---\n"
        for key in dir(self.config):
            if not key.startswith("__"):
                value = getattr(self.config, key)
                log_message += f"{key}: {value}\n"
        log_message += "-------------------\n\n"
        with open(self.log_file_path, "w") as f:
            f.write(log_message)

    def _prepare_batch(self, annotations, training=True):
        video_feats_list, questions, captions = [], [], []
        video_lengths = []
        valid_annotations = []
        for ann in annotations:
            video_id = ann.get("video_id") or ann.get("id") or ann.get("video")
            if video_id is None:
                video_name = ann.get("video") or ann.get("video_name") or ann.get("media")
                if video_name:
                    video_id = os.path.splitext(os.path.basename(str(video_name)))[0]
            if video_id is None:
                continue
            video_id = str(video_id)

            question = (
                ann.get("question_content")
                or ann.get("question")
                or ann.get("prompt")
                or ann.get("caption_prompt")
                or self.default_question_prompt
            )
            if isinstance(question, list):
                question = " ".join(str(q) for q in question if q)
            if question is None:
                question = self.default_question_prompt
            else:
                question = str(question)

            answer = ann.get("anser") or ann.get("caption") or ann.get("answer")
            if isinstance(answer, list):
                filtered_answers = [str(a) for a in answer if a]
                if not filtered_answers:
                    continue
                answer = random.choice(filtered_answers) if training else filtered_answers[0]
            if not answer:
                continue
            answer = str(answer)

            video_path = ann.get("video_feature_path")
            if video_path:
                video_path = video_path if os.path.isabs(video_path) else os.path.join(self.config.VIDEO_FEATURES_DIR, video_path)
            else:
                video_path = os.path.join(self.config.VIDEO_FEATURES_DIR, f"{video_id}.npy")

            if not os.path.exists(video_path):
                logging.warning(f"跳过缺失的视频特征文件 {video_path}")
                continue

            try:
                video_feat = torch.from_numpy(np.load(video_path)).float()
            except Exception as exc:
                logging.warning(f"跳过损坏的视频特征文件 {video_path}: {exc}")
                continue

            video_feats_list.append(video_feat)
            video_lengths.append(video_feat.size(0))
            questions.append(question)

            raw_answer = ann.get("anser") or ann.get("caption") or ann.get("answer")
            all_answers_for_sample = []
            if isinstance(raw_answer, list):
                all_answers_for_sample = [str(a) for a in raw_answer if a]
                if not all_answers_for_sample:
                    continue
                selected_answer = random.choice(all_answers_for_sample) if training else all_answers_for_sample[0]
            else:
                if not raw_answer:
                    continue
                selected_answer = str(raw_answer)
                all_answers_for_sample = [selected_answer]

            captions.append(selected_answer)
            valid_annotations.append(ann)
            if not hasattr(self, "_temp_all_captions_list"):
                self._temp_all_captions_list = []
            self._temp_all_captions_list.append(all_answers_for_sample)

        if not questions:
            return None

        all_captions_batch = getattr(self, "_temp_all_captions_list", [])
        self._temp_all_captions_list = []

        video_batch = nn.utils.rnn.pad_sequence(video_feats_list, batch_first=True).to(self.device)
        video_mask = torch.zeros(video_batch.size(0), video_batch.size(1), dtype=torch.long, device=self.device)
        for idx, v_len in enumerate(video_lengths):
            video_mask[idx, :v_len] = 1

        return video_batch, video_mask, questions, captions, all_captions_batch, valid_annotations

    def _build_target_tokens(self, captions):
        tokens = self.t5_tokenizer(
            captions, padding="longest", truncation=True, max_length=self.max_target_length, return_tensors="pt"
        ).input_ids.to(self.device)
        labels = tokens.clone()
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        return labels

    def _ensure_nltk_resources(self):
        if getattr(self, "_nltk_ready", False):
            return
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
        except Exception as exc:
            logging.warning(f"NLTK 'punkt' 资源不可用，自动指标可能失效: {exc}")
        finally:
            self._nltk_ready = True

    def train(self):
        global_step = 0
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            np.random.shuffle(self.train_data)
            progress_bar = tqdm(
                range(0, len(self.train_data), self.config.BATCH_SIZE), desc=f"Epoch {epoch + 1}/{self.config.EPOCHS}"
            )
            for i in progress_bar:
                batch_anns = self.train_data[i : i + self.config.BATCH_SIZE]
                batch = self._prepare_batch(batch_anns, training=True)
                if batch is None:
                    continue

                video_feats, video_mask, questions, captions, _, _ = batch
                labels = self._build_target_tokens(captions)

                self.optimizer.zero_grad()
                outputs = self.model(video_feats, video_mask, questions, labels=labels, generate=False)
                loss = outputs["lm_outputs"].loss if outputs["lm_outputs"] is not None else torch.tensor(0.0, device=self.device)
                loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                global_step += 1

                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            self.evaluate(epoch, save_best=True)

    def evaluate(self, epoch, save_best=True):
        self.model.eval()

        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        try:
            from rouge import Rouge
        except ImportError:
            Rouge = None
        try:
            from pycocoevalcap.meteor.meteor import Meteor
        except ImportError:
            Meteor = None
        try:
            from pycocoevalcap.cider.cider import Cider
        except ImportError:
            Cider = None

        self._ensure_nltk_resources()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_refs = []
        captured_batch_samples = None

        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_anns), self.config.BATCH_SIZE), desc="Evaluating"):
                batch_anns = self.test_anns[i : i + self.config.BATCH_SIZE]
                if not batch_anns:
                    continue
                batch = self._prepare_batch(batch_anns, training=False)
                if batch is None:
                    continue
                video_feats, video_mask, questions, captions, batch_all_captions, batch_valid_anns = batch
                if not captions:
                    continue
                labels = self._build_target_tokens(captions)
                if labels.size(0) == 0:
                    continue

                outputs = self.model(video_feats, video_mask, questions, labels=labels, generate=True)

                lm_out = outputs["lm_outputs"]
                if lm_out is not None and lm_out.loss is not None:
                    total_loss += lm_out.loss.item() * labels.size(0)
                total_samples += labels.size(0)

                preds = outputs.get("generated_text") or []
                if len(preds) != len(captions):
                    if len(preds) < len(captions):
                        logging.warning(
                            "生成结果数量({}) 与有效样本数({}) 不一致，使用空字符串填充缺失预测。".format(len(preds), len(captions))
                        )
                        preds = preds + [""] * (len(captions) - len(preds))
                    else:
                        preds = preds[: len(captions)]

                all_preds.extend(preds)
                all_refs.extend(batch_all_captions)

                if captured_batch_samples is None:
                    captured_batch_samples = []
                    align_iter = zip(batch_valid_anns, preds, captions)
                    for ann, pred, reference in align_iter:
                        vid = ann.get("video_id") or ann.get("id") or ann.get("video")
                        if vid is None:
                            video_name = ann.get("video") or ann.get("video_name") or ann.get("media")
                            if video_name:
                                vid = os.path.splitext(os.path.basename(str(video_name)))[0]
                        vid = str(vid) if vid is not None else "unknown"
                        captured_batch_samples.append({"video_id": vid, "reference": reference, "prediction": pred})

        avg_loss = total_loss / max(1, total_samples)

        bleu1s, bleu4s = [], []
        rouge_l_scores = []
        smooth = SmoothingFunction().method1
        rouge_eval = Rouge() if Rouge else None

        for pred, refs in zip(all_preds, all_refs):
            ref_tokens_list = [nltk.word_tokenize(r.lower()) for r in refs]
            pred_tokens = nltk.word_tokenize(pred.lower())
            bleu1s.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
            bleu4s.append(
                sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            )
            if rouge_eval:
                try:
                    scores = [rouge_eval.get_scores(pred, r)[0]["rouge-l"]["f"] for r in refs]
                    rouge_l_scores.append(max(scores) if scores else 0.0)
                except Exception:
                    rouge_l_scores.append(0.0)

        bleu1 = sum(bleu1s) / max(1, len(bleu1s))
        bleu4 = sum(bleu4s) / max(1, len(bleu4s))
        rouge_l = sum(rouge_l_scores) / max(1, len(rouge_l_scores)) if rouge_l_scores else 0.0

        meteor = 0.0
        if Meteor and all_preds and all_refs:
            try:
                meteor_eval = Meteor()
                refs_dict = {idx: refs for idx, refs in enumerate(all_refs)}
                preds_dict = {idx: [pred] for idx, pred in enumerate(all_preds)}
                meteor, _ = meteor_eval.compute_score(refs_dict, preds_dict)
            except Exception as exc:
                logging.warning(f"METEOR 计算失败: {exc}")

        cider = 0.0
        if Cider and all_preds and all_refs:
            try:
                cider_eval = Cider()
                refs_dict = {idx: refs for idx, refs in enumerate(all_refs)}
                preds_dict = {idx: [pred] for idx, pred in enumerate(all_preds)}
                cider, _ = cider_eval.compute_score(refs_dict, preds_dict)
            except Exception as exc:
                logging.warning(f"CIDEr 计算失败: {exc}")

        is_best = cider > self.best_cider
        best_suffix = "  <-- New Best!" if (save_best and is_best) else ""
        header = (
            f"-------------- Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, BLEU-1: {bleu1:.4f} BLEU-4: {bleu4:.4f} "
            f"METEOR: {meteor:.4f} ROUGE-L: {rouge_l:.4f} CIDEr: {cider:.4f}{best_suffix} --------------"
        )
        print("\n" + header)

        if self.enable_logging:
            try:
                with open(self.log_file_path, "a") as f:
                    f.write(header + "\n")
            except Exception as e:
                logging.error(f"无法将评估结果写入日志文件: {e}")

        if self.enable_logging and captured_batch_samples:
            sample_file = os.path.join(self.sample_text_dir, f"epoch_{epoch + 1:03d}.txt")
            try:
                with open(sample_file, "w") as sample_fp:
                    for idx, sample in enumerate(captured_batch_samples, start=1):
                        sample_fp.write(f"# Sample {idx} | Video: {sample['video_id']}\n")
                        sample_fp.write(f"Ground Truth: {sample['reference']}\n")
                        sample_fp.write(f"Prediction: {sample['prediction']}\n\n")
            except Exception as exc:
                logging.error(f"无法写入评估样本文件 {sample_file}: {exc}")

        if save_best and is_best:
            self.best_cider = cider
            logging.info(f"发现新的最佳 CIDEr: {self.best_cider:.4f}。正在保存模型...")
            self.save_model()

    def load_checkpoint(self, checkpoint_dir=None):
        ckpt_dir = checkpoint_dir or self.config.SAVE_DIR
        lora_dir = os.path.join(ckpt_dir, "best_lora_adapters")
        other_params_path = os.path.join(ckpt_dir, "best_other_params.pth")

        if not os.path.isdir(lora_dir):
            logging.error(f"未找到 LoRA 目录: {lora_dir}")
            return False
        if not os.path.isfile(other_params_path):
            logging.error(f"未找到参数文件: {other_params_path}")
            return False

        model_to_load = self.model.module if hasattr(self.model, "module") else self.model

        try:
            if isinstance(model_to_load.t5_model, PeftModel):
                try:
                    model_to_load.t5_model.load_adapter(lora_dir, adapter_name="default", is_trainable=True)
                    model_to_load.t5_model.set_adapter("default")
                    logging.info(f"已通过 load_adapter 加载 LoRA 适配器: {lora_dir}")
                except Exception as e:
                    logging.warning(f"load_adapter 失败，尝试重新 wrap: {e}")
                    from peft.utils.save_and_load import load_peft_weights

                    adapters_weights = load_peft_weights(lora_dir)
                    set_peft_model_state_dict(model_to_load.t5_model, adapters_weights)
                    logging.info(f"已通过 set_peft_model_state_dict 加载 LoRA 适配器: {lora_dir}")
            else:
                model_to_load.t5_model = PeftModel.from_pretrained(model_to_load.t5_model, lora_dir)
                logging.info(f"已通过 from_pretrained 加载 LoRA 适配器: {lora_dir}")

            model_to_load.t5_model.to(self.device)
        except Exception as exc:
            logging.error(f"加载 LoRA 适配器失败: {exc}")

        try:
            state = torch.load(other_params_path, map_location=self.device)
        except Exception as exc:
            logging.error(f"读取参数文件失败: {exc}")
            return False

        def _load_state(module, key):
            if module is None:
                return
            sd = state.get(key)
            if sd is None:
                logging.warning(f"参数 {key} 不存在于检查点，跳过加载。")
                return
            msg = module.load_state_dict(sd, strict=False)
            logging.info(f"加载 {key}: {msg}")

        _load_state(model_to_load.video_qformer, "video_qformer")
        _load_state(model_to_load.bert_embedding, "bert_embedding")
        _load_state(model_to_load.video_patch_downsampler, "video_patch_downsampler")
        _load_state(model_to_load.proj_video, "proj_video")

        def _load_param(param, key):
            tensor = state.get(key)
            if tensor is None:
                logging.warning(f"参数 {key} 不存在于检查点，跳过加载。")
                return
            try:
                param.data.copy_(tensor.to(self.device))
                logging.info(f"加载参数 {key} 成功")
            except Exception as exc:
                logging.warning(f"加载参数 {key} 失败: {exc}")

        _load_param(model_to_load.video_query_tokens, "video_query_tokens")

        self.best_model_dir = ckpt_dir
        logging.info(f"检查点加载完成: {ckpt_dir}")
        return True

    def save_model(self):
        lora_path = os.path.join(self.best_model_dir, "best_lora_adapters")
        other_params_path = os.path.join(self.best_model_dir, "best_other_params.pth")
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.t5_model.save_pretrained(lora_path)
        checkpoint = {
            "video_qformer": getattr(model_to_save, "video_qformer", None).state_dict()
            if hasattr(model_to_save, "video_qformer")
            else None,
            "bert_embedding": model_to_save.bert_embedding.state_dict(),
            "video_query_tokens": model_to_save.video_query_tokens,
            "video_patch_downsampler": model_to_save.video_patch_downsampler.state_dict()
            if hasattr(model_to_save, "video_patch_downsampler")
            else None,
            "proj_video": model_to_save.proj_video.state_dict() if hasattr(model_to_save, "proj_video") else None,
        }
        torch.save(checkpoint, other_params_path)
        logging.info(f"模型已保存至 {self.best_model_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    trainer = VideoOnlyTrainer()
    trainer.train()
