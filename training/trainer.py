"""
训练器模块
包含Trainer类，负责模型训练、评估和保存
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

from ..models.multimodal_t5 import MultiModal_T5_Classifier
from .config import TrainConfig


class Trainer:
    """多模态视听问答模型训练器"""

    def __init__(self, config, gpu_ids=None):
        self.config = config

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

        # 日志设置
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_dir = os.path.join(config.LOG_DIR, timestamp)
        os.makedirs(self.run_log_dir, exist_ok=True)
        self.sample_text_dir = os.path.join(self.run_log_dir, "test")
        os.makedirs(self.sample_text_dir, exist_ok=True)
        gpu_tag = "_".join(str(g) for g in self.gpu_ids) if self.gpu_ids else "cpu"
        self.log_file_path = os.path.join(self.run_log_dir, f"training_log_gpu{gpu_tag}.txt")
        logging.info(f"本次训练的日志将记录在: {self.log_file_path}")
        self.log_hyperparameters()

        # 数据加载
        train_anns      = json.load(open(config.TRAIN_ANNOTATIONS_PATH, 'r'))
        
        # 尝试加载完整的测试集标注 (test_videodatainfo.json)
        test_ann_path = config.TEST_ANNOTATIONS_PATH
        test_videodatainfo_path = os.path.join(os.path.dirname(test_ann_path), "test_videodatainfo.json")
        
        if os.path.exists(test_videodatainfo_path):
            logging.info(f"检测到完整的测试集标注文件: {test_videodatainfo_path}，正在加载并转换格式...")
            
            # 1. 加载原始 test.json 以获取正确的测试集 Video ID 列表 (避免数据泄漏)
            original_test_data = json.load(open(test_ann_path, 'r'))
            valid_test_video_ids = set()
            for item in original_test_data:
                vid = item.get('video_id') or item.get('id') or item.get('video')
                if vid:
                    valid_test_video_ids.add(str(vid))
            logging.info(f"原始测试集包含 {len(valid_test_video_ids)} 个视频 ID。")

            # 2. 加载包含完整字幕的 test_videodatainfo.json
            raw_test_data = json.load(open(test_videodatainfo_path, 'r'))
            
            # 将扁平的 caption 列表转换为按 video_id 聚合的格式
            video_to_captions = {}
            if isinstance(raw_test_data, dict) and "sentences" in raw_test_data:
                raw_test_data = raw_test_data["sentences"]
            
            for item in raw_test_data:
                vid = str(item.get('video_id'))
                cap = item.get('caption')
                
                # 关键：只保留在原始 test.json 中存在的视频，过滤掉可能存在于训练集中的视频
                if vid not in valid_test_video_ids:
                    continue
                    
                if not vid or not cap:
                    continue
                
                if vid not in video_to_captions:
                    video_to_captions[vid] = []
                video_to_captions[vid].append(cap)
            
            self.test_anns = []
            for vid, caps in video_to_captions.items():
                self.test_anns.append({
                    "video_id": vid,
                    "caption": caps,
                    "video": f"{vid}.mp4" 
                })
            logging.info(f"成功加载并过滤测试集，共 {len(self.test_anns)} 个视频样本 (已过滤掉非测试集视频)。")
            
        else:
            logging.warning(f"未找到 {test_videodatainfo_path}，回退到使用 {test_ann_path}")
            self.test_anns  = json.load(open(test_ann_path, 'r'))

        self.train_data = train_anns
        self.max_target_length = getattr(config, "MAX_TARGET_LENGTH", 32)
        self.default_question_prompt = getattr(
            config,
            "DEFAULT_QUESTION_PROMPT",
            "Describe the video content in detail."
        )

        # 初始化模型
        base_model = MultiModal_T5_Classifier(
            num_classes=0,
            t5_model_path=config.T5_MODEL_PATH,
            qformer_num_layers=config.QFORMER_NUM_LAYERS,
            num_query_token=config.NUM_QUERY_TOKEN,
            lora_r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            blip2_model_path=config.BLIP2_MODEL_PATH,
            bert_tokenizer_name_or_path=getattr(config, "BERT_TOKENIZER_PATH", "bert-base-uncased")
        ).to(self.device)

        self.bert_tokenizer = base_model.bert_tokenizer
        self.t5_tokenizer = base_model.t5_tokenizer

        if len(self.gpu_ids) > 1:
            logging.info(f"启用 DataParallel，多卡: {self.gpu_ids}")
            self.model = nn.DataParallel(base_model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
        else:
            self.model = base_model

        # 优化器设置
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.LEARNING_RATE)

        # 学习率调度器（可配置是否启用余弦退火）
        self.use_cosine_decay = getattr(self.config, "USE_COSINE_DECAY", False)
        steps_per_epoch          = max(1, math.ceil(len(self.train_data) / self.config.BATCH_SIZE))
        self.total_training_steps = steps_per_epoch * self.config.EPOCHS
        self.warmup_steps         = int(self.config.WARMUP_RATIO * self.total_training_steps)
        if self.use_cosine_decay:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_training_steps
            )

        self.best_accuracy = 0.0
        self._nltk_ready = False
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        logging.info("Trainer 初始化完成。")
        logging.info(f"总训练步数: {self.total_training_steps}, Warmup 步数: {self.warmup_steps}")

    def log_hyperparameters(self):
        """记录超参数到日志文件"""
        log_message = "--- 实验配置 ---\n"
        for key in dir(self.config):
            if not key.startswith('__'):
                value = getattr(self.config, key)
                log_message += f"{key}: {value}\n"
        log_message += "-------------------\n\n"
        with open(self.log_file_path, 'w') as f:
            f.write(log_message)

    def _prepare_batch(self, annotations, training=True):
        """准备训练批次数据"""
        video_feats_list, audio_feats_list, questions, captions = [], [], [], []
        for ann in annotations:
            video_id = ann.get('video_id') or ann.get('id') or ann.get('video')
            if video_id is None:
                video_name = ann.get('video') or ann.get('video_name') or ann.get('media')
                if video_name:
                    video_id = os.path.splitext(os.path.basename(str(video_name)))[0]
            if video_id is None:
                continue
            video_id = str(video_id)

            question = (
                ann.get('question_content')
                or ann.get('question')
                or ann.get('prompt')
                or ann.get('caption_prompt')
                or self.default_question_prompt
            )
            if isinstance(question, list):
                question = " ".join(str(q) for q in question if q)
            if question is None:
                question = self.default_question_prompt
            else:
                question = str(question)

            answer = ann.get('anser') or ann.get('caption') or ann.get('answer')
            if isinstance(answer, list):
                filtered_answers = [str(a) for a in answer if a]
                if not filtered_answers:
                    continue
                answer = random.choice(filtered_answers) if training else filtered_answers[0]
            if not answer:
                continue
            answer = str(answer)

            video_path = ann.get('video_feature_path')
            if video_path:
                video_path = video_path if os.path.isabs(video_path) else os.path.join(self.config.VIDEO_FEATURES_DIR, video_path)
            else:
                video_path = os.path.join(self.config.VIDEO_FEATURES_DIR, f"{video_id}.npy")

            audio_path = ann.get('clap_feature_path')
            if audio_path:
                audio_path = audio_path if os.path.isabs(audio_path) else os.path.join(self.config.AUDIO_FEATURES_DIR, audio_path)
            else:
                audio_path = os.path.join(self.config.AUDIO_FEATURES_DIR, f"{video_id}.npy")

            if not os.path.exists(video_path):
                continue

            if not os.path.exists(audio_path):
                continue

            video_feats_list.append(torch.from_numpy(np.load(video_path)).float())
            audio_feats_list.append(torch.from_numpy(np.load(audio_path)).float())
            questions.append(question)
            
            # 处理答案/字幕
            raw_answer = ann.get('anser') or ann.get('caption') or ann.get('answer')
            all_answers_for_sample = []
            
            if isinstance(raw_answer, list):
                all_answers_for_sample = [str(a) for a in raw_answer if a]
                if not all_answers_for_sample:
                    continue
                # 训练时随机采样一个作为目标；评估时取第一个用于计算Loss，但保留所有用于计算指标
                selected_answer = random.choice(all_answers_for_sample) if training else all_answers_for_sample[0]
            else:
                # 单个字符串的情况
                if not raw_answer:
                    continue
                selected_answer = str(raw_answer)
                all_answers_for_sample = [selected_answer]

            captions.append(selected_answer)
            # 额外返回该样本的所有参考答案，用于评估
            if not hasattr(self, '_temp_all_captions_list'):
                self._temp_all_captions_list = []
            self._temp_all_captions_list.append(all_answers_for_sample)

        if not questions:
            return None
            
        # 获取并清空临时列表
        all_captions_batch = getattr(self, '_temp_all_captions_list', [])
        self._temp_all_captions_list = []
        
        video_batch  = nn.utils.rnn.pad_sequence(video_feats_list, batch_first=True).to(self.device)
        audio_batch  = nn.utils.rnn.pad_sequence(audio_feats_list, batch_first=True).to(self.device)
        
        return video_batch, audio_batch, questions, captions, all_captions_batch

    def _build_target_tokens(self, captions):
        tokens = self.t5_tokenizer(
            captions,
            padding="longest",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        ).input_ids.to(self.device)
        labels = tokens.clone()
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        return labels

    def _ensure_nltk_resources(self):
        if self._nltk_ready:
            return
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except Exception as exc:
            logging.warning(f"NLTK 'punkt' 资源不可用，自动指标可能失效: {exc}")
        finally:
            self._nltk_ready = True

    @staticmethod
    def _parse_types(annotations):
        """解析问题类型"""
        parsed = []
        for ann in annotations:
            try:
                t = json.loads(ann['type'])
                if isinstance(t, list) and len(t) >= 2:
                    parsed.append((t[0], t[1]))
                else:
                    parsed.append(("Unknown", "Unknown"))
            except Exception:
                parsed.append(("Unknown", "Unknown"))
        return parsed

    def train(self):
        """执行训练过程"""
        global_step = 0
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            np.random.shuffle(self.train_data)
            progress_bar = tqdm(range(0, len(self.train_data), self.config.BATCH_SIZE), desc=f"Epoch {epoch+1}/{self.config.EPOCHS}")
            for i in progress_bar:
                batch_anns = self.train_data[i : i + self.config.BATCH_SIZE]
                batch = self._prepare_batch(batch_anns, training=True)
                if batch is None:
                    continue

                video_feats, audio_feats, questions, captions, _ = batch
                labels = self._build_target_tokens(captions)

                self.optimizer.zero_grad()
                outputs = self.model(video_feats, audio_feats, questions, labels=labels, generate=False)
                loss = outputs['lm_outputs'].loss if outputs['lm_outputs'] is not None else torch.tensor(0.0, device=self.device)
                loss.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                global_step += 1

                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            self.evaluate(epoch)

    def evaluate(self, epoch):
        """评估模型性能，包含 BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr"""
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
        exact_matches = 0
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
                video_feats, audio_feats, questions, captions, batch_all_captions = batch
                if not captions:
                    continue
                labels = self._build_target_tokens(captions)
                if labels.size(0) == 0:
                    continue

                outputs = self.model(video_feats, audio_feats, questions, labels=labels, generate=True)

                lm_out = outputs['lm_outputs']
                if lm_out is not None and lm_out.loss is not None:
                    total_loss += lm_out.loss.item() * labels.size(0)
                total_samples += labels.size(0)

                preds = outputs.get('generated_text') or []
                if len(preds) != len(captions):
                    if len(preds) < len(captions):
                        logging.warning(
                            "生成结果数量({}) 与有效样本数({}) 不一致，使用空字符串填充缺失预测。".format(len(preds), len(captions))
                        )
                        preds = preds + [""] * (len(captions) - len(preds))
                    else:
                        preds = preds[:len(captions)]
                
                # Exact Match 仍然只对比选中的那个（通常是第一个），或者也可以改为对比所有
                for pred, refs in zip(preds, batch_all_captions):
                    # 只要匹配任何一个参考答案就算 Exact Match
                    if any(pred.strip().lower() == r.strip().lower() for r in refs):
                        exact_matches += 1
                        
                all_preds.extend(preds)
                all_refs.extend(batch_all_captions)

                if captured_batch_samples is None:
                    captured_batch_samples = []
                    for ann, pred, reference in zip(batch_anns, preds, captions):
                        vid = ann.get('video_id') or ann.get('id') or ann.get('video')
                        if vid is None:
                            video_name = ann.get('video') or ann.get('video_name') or ann.get('media')
                            if video_name:
                                vid = os.path.splitext(os.path.basename(str(video_name)))[0]
                        vid = str(vid) if vid is not None else "unknown"
                        captured_batch_samples.append({
                            "video_id": vid,
                            "reference": reference,
                            "prediction": pred
                        })

        avg_loss = total_loss / max(1, total_samples)
        accuracy = (exact_matches / max(1, total_samples)) * 100.0
        is_best = accuracy > self.best_accuracy

        # 计算 BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr
        bleu1s, bleu4s = [], []
        rouge_l_scores = []
        smooth = SmoothingFunction().method1
        rouge_eval = Rouge() if Rouge else None
        
        # all_refs 现在是 list of lists (每个样本有多个参考答案)
        for pred, refs in zip(all_preds, all_refs):
            # refs 是一个字符串列表
            ref_tokens_list = [nltk.word_tokenize(r.lower()) for r in refs]
            pred_tokens = nltk.word_tokenize(pred.lower())
            
            # sentence_bleu 接受 list of list of tokens 作为 references
            bleu1s.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
            bleu4s.append(sentence_bleu(ref_tokens_list, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
            
            if rouge_eval:
                try:
                    # 计算与每个参考答案的 Rouge-L，取最大值
                    scores = [rouge_eval.get_scores(pred, r)[0]['rouge-l']['f'] for r in refs]
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
                # pycocoevalcap 期望 {idx: [ref1, ref2...]}
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

        header = f"-------------- Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Exact Match: {accuracy:.2f}% BLEU-1: {bleu1:.4f} BLEU-4: {bleu4:.4f} METEOR: {meteor:.4f} ROUGE-L: {rouge_l:.4f} CIDEr: {cider:.4f}{'  <-- New Best!' if is_best else ''} --------------"
        print("\n" + header)

        try:
            with open(self.log_file_path, 'a') as f:
                f.write(header + "\n")
        except Exception as e:
            logging.error(f"无法将评估结果写入日志文件: {e}")

        if captured_batch_samples:
            sample_file = os.path.join(self.sample_text_dir, f"epoch_{epoch+1:03d}.txt")
            try:
                with open(sample_file, 'w') as sample_fp:
                    for idx, sample in enumerate(captured_batch_samples, start=1):
                        sample_fp.write(f"# Sample {idx} | Video: {sample['video_id']}\n")
                        sample_fp.write(f"Ground Truth: {sample['reference']}\n")
                        sample_fp.write(f"Prediction: {sample['prediction']}\n\n")
            except Exception as exc:
                logging.error(f"无法写入评估样本文件 {sample_file}: {exc}")

        if is_best:
            self.best_accuracy = accuracy
            logging.info(f"发现新的最佳准确率: {self.best_accuracy:.2f}%。正在保存模型...")
            self.save_model()

    def save_model(self):
        """保存模型权重"""
        lora_path         = os.path.join(self.config.SAVE_DIR, "best_lora_adapters")
        other_params_path = os.path.join(self.config.SAVE_DIR, "best_other_params.pth")
        model_to_save     = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.t5_model.save_pretrained(lora_path)
        checkpoint = {
            'video_qformer': getattr(model_to_save, 'video_qformer', None).state_dict() if hasattr(model_to_save, 'video_qformer') else None,
            'audio_qformer': getattr(model_to_save, 'audio_qformer', None).state_dict() if hasattr(model_to_save, 'audio_qformer') else None,
            'bert_embedding': model_to_save.bert_embedding.state_dict(),
            'video_query_tokens': model_to_save.video_query_tokens,
            'audio_query_tokens': model_to_save.audio_query_tokens,
            'audio_upscaler': model_to_save.audio_upscaler.state_dict() if hasattr(model_to_save, 'audio_upscaler') else None,
            'proj_video': model_to_save.proj_video.state_dict() if hasattr(model_to_save, 'proj_video') else None,
            'proj_audio': model_to_save.proj_audio.state_dict() if hasattr(model_to_save, 'proj_audio') else None,
        }
        torch.save(checkpoint, other_params_path)
        logging.info(f"模型已保存至 {self.config.SAVE_DIR}")
