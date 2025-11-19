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
from torch.optim import AdamW
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup

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
        gpu_tag = "_".join(str(g) for g in self.gpu_ids) if self.gpu_ids else "cpu"
        self.log_file_path = os.path.join(config.LOG_DIR, f"training_log_{timestamp}_gpu{gpu_tag}.txt")
        logging.info(f"本次训练的日志将记录在: {self.log_file_path}")
        self.log_hyperparameters()

        # 数据加载
        train_anns      = json.load(open(config.TRAIN_ANNOTATIONS_PATH, 'r'))
        self.test_anns  = json.load(open(config.TEST_ANNOTATIONS_PATH, 'r'))
        self.answer_to_label = {ans: i for i, ans in enumerate(sorted(list(set(item['anser'] for item in train_anns))))}
        self.train_data = train_anns
        num_classes     = len(self.answer_to_label)

        # 初始化模型
        base_model = MultiModal_T5_Classifier(
            num_classes=num_classes,
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
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        steps_per_epoch          = max(1, math.ceil(len(self.train_data) / self.config.BATCH_SIZE))
        self.total_training_steps = steps_per_epoch * self.config.EPOCHS
        self.warmup_steps         = int(self.config.WARMUP_RATIO * self.total_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps
        )

        self.best_accuracy = 0.0
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

    def _prepare_batch(self, annotations):
        """准备训练批次数据"""
        video_feats_list, audio_feats_list, questions, labels = [], [], [], []
        for ann in annotations:
            video_id, question, answer = ann['video_id'], ann['question_content'], ann['anser']
            video_path = os.path.join(self.config.VIDEO_FEATURES_DIR, f"{video_id}.npy")
            audio_path = os.path.join(self.config.AUDIO_FEATURES_DIR, f"{video_id}.npy")
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue
            video_feats_list.append(torch.from_numpy(np.load(video_path)).float())
            audio_feats_list.append(torch.from_numpy(np.load(audio_path)).float())
            questions.append(question)
            labels.append(self.answer_to_label.get(answer, -1))
        if not questions:
            return None
        video_batch  = nn.utils.rnn.pad_sequence(video_feats_list, batch_first=True).to(self.device)
        audio_batch  = nn.utils.rnn.pad_sequence(audio_feats_list, batch_first=True).to(self.device)
        labels_batch = torch.tensor(labels, dtype=torch.long).to(self.device)
        return video_batch, audio_batch, questions, labels_batch

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
                batch = self._prepare_batch(batch_anns)
                if batch is None:
                    continue

                video_feats, audio_feats, questions, labels = batch

                self.optimizer.zero_grad()
                logits = self.model(video_feats, audio_feats, questions)
                loss   = self.criterion(logits, labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            self.evaluate(epoch)

    def evaluate(self, epoch):
        """评估模型性能"""
        self.model.eval()

        per_type = {
            ("Audio", "Counting"):      {"correct": 0, "total": 0},
            ("Audio", "Comparative"):   {"correct": 0, "total": 0},
            ("Visual", "Counting"):     {"correct": 0, "total": 0},
            ("Visual", "Location"):     {"correct": 0, "total": 0},
            ("Audio-Visual", "Existential"): {"correct": 0, "total": 0},
            ("Audio-Visual", "Counting"):    {"correct": 0, "total": 0},
            ("Audio-Visual", "Location"):    {"correct": 0, "total": 0},
            ("Audio-Visual", "Comparative"): {"correct": 0, "total": 0},
            ("Audio-Visual", "Temporal"):    {"correct": 0, "total": 0},
        }

        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_anns), self.config.BATCH_SIZE), desc="Evaluating"):
                batch_anns = self.test_anns[i : i + self.config.BATCH_SIZE]
                if not batch_anns:
                    continue
                types_this_batch = self._parse_types(batch_anns)

                batch = self._prepare_batch(batch_anns)
                if batch is None:
                    continue
                video_feats, audio_feats, questions, labels = batch

                valid_mask = labels != -1
                if not valid_mask.any():
                    continue

                logits = self.model(
                    video_feats[valid_mask],
                    audio_feats[valid_mask],
                    [q for j, q in enumerate(questions) if valid_mask[j]]
                )
                predictions = torch.argmax(logits, dim=1)

                total_correct += (predictions == labels[valid_mask]).sum().item()
                total_samples += valid_mask.sum().item()

                idxs = [idx for idx, ok in enumerate(valid_mask.tolist()) if ok]
                for k, idx in enumerate(idxs):
                    key = types_this_batch[idx]
                    if key in per_type:
                        per_type[key]["total"] += 1
                        if predictions[k].item() == labels[idx].item():
                            per_type[key]["correct"] += 1

        def _safe_acc(c, n): return (c / n * 100.0) if n > 0 else 0.0
        accuracy = _safe_acc(total_correct, total_samples)
        is_best  = accuracy > self.best_accuracy

        audio_count = per_type[("Audio", "Counting")]
        audio_comp  = per_type[("Audio", "Comparative")]
        audio_avg_c = audio_count["correct"] + audio_comp["correct"]
        audio_avg_n = audio_count["total"]   + audio_comp["total"]
        audio_avg   = _safe_acc(audio_avg_c, audio_avg_n)

        visual_count = per_type[("Visual", "Counting")]
        visual_loc   = per_type[("Visual", "Location")]
        visual_avg_c = visual_count["correct"] + visual_loc["correct"]
        visual_avg_n = visual_count["total"]   + visual_loc["total"]
        visual_avg   = _safe_acc(visual_avg_c, visual_avg_n)

        av_exist = per_type[("Audio-Visual", "Existential")]
        av_count = per_type[("Audio-Visual", "Counting")]
        av_loc   = per_type[("Audio-Visual", "Location")]
        av_comp  = per_type[("Audio-Visual", "Comparative")]
        av_temp  = per_type[("Audio-Visual", "Temporal")]
        av_avg_c = av_exist["correct"] + av_count["correct"] + av_loc["correct"] + av_comp["correct"] + av_temp["correct"]
        av_avg_n = av_exist["total"]   + av_count["total"]   + av_loc["total"]   + av_comp["total"]   + av_temp["total"]
        av_avg   = _safe_acc(av_avg_c, av_avg_n)

        header = f"-------------- Epoch: {epoch+1}, Accuracy: {accuracy:.4f}%{'  <-- New Best!' if is_best else ''} --------------"
        print("\n" + header)
        print("Audio QA:")
        print(f"  Count : {_safe_acc(audio_count['correct'], audio_count['total']):.2f}%  (n={audio_count['total']})")
        print(f"  Comp  : {_safe_acc(audio_comp['correct'],  audio_comp['total']):.2f}%  (n={audio_comp['total']})")
        print(f"  Avg   : {audio_avg:.2f}%  (n={audio_avg_n})")
        print("Visual QA:")
        print(f"  Count : {_safe_acc(visual_count['correct'], visual_count['total']):.2f}%  (n={visual_count['total']})")
        print(f"  Local : {_safe_acc(visual_loc['correct'],   visual_loc['total']):.2f}%  (n={visual_loc['total']})")
        print(f"  Avg   : {visual_avg:.2f}%  (n={visual_avg_n})")
        print("Audio-Visual QA:")
        print(f"  Exist : {_safe_acc(av_exist['correct'], av_exist['total']):.2f}%  (n={av_exist['total']})")
        print(f"  Count : {_safe_acc(av_count['correct'], av_count['total']):.2f}%  (n={av_count['total']})")
        print(f"  Local : {_safe_acc(av_loc['correct'],   av_loc['total']):.2f}%  (n={av_loc['total']})")
        print(f"  Comp  : {_safe_acc(av_comp['correct'],  av_comp['total']):.2f}%  (n={av_comp['total']})")
        print(f"  Temp  : {_safe_acc(av_temp['correct'],  av_temp['total']):.2f}%  (n={av_temp['total']})")
        print(f"  Avg   : {av_avg:.2f}%  (n={av_avg_n})")
        print(f"\nOverall: {accuracy:.2f}%\n")

        log_lines = [header, "Audio:", f"  Count : {_safe_acc(audio_count['correct'], audio_count['total']):.2f}%  (n={audio_count['total']})", f"  Comp  : {_safe_acc(audio_comp['correct'],  audio_comp['total']):.2f}%  (n={audio_comp['total']})", f"  Avg   : {audio_avg:.2f}%  (n={audio_avg_n})", "Visual:", f"  Count : {_safe_acc(visual_count['correct'], visual_count['total']):.2f}%  (n={visual_count['total']})", f"  Local : {_safe_acc(visual_loc['correct'],   visual_loc['total']):.2f}%  (n={visual_loc['total']})", f"  Avg   : {visual_avg:.2f}%  (n={visual_avg_n})", "Audio-Visual:", f"  Exist : {_safe_acc(av_exist['correct'], av_exist['total']):.2f}%  (n={av_exist['total']})", f"  Count : {_safe_acc(av_count['correct'], av_count['total']):.2f}%  (n={av_count['total']})", f"  Local : {_safe_acc(av_loc['correct'],   av_loc['total']):.2f}%  (n={av_loc['total']})", f"  Comp  : {_safe_acc(av_comp['correct'],  av_comp['total']):.2f}%  (n={av_comp['total']})", f"  Temp  : {_safe_acc(av_temp['correct'],  av_temp['total']):.2f}%  (n={av_temp['total']})", f"  Avg   : {av_avg:.2f}%  (n={av_avg_n})", f"Overall: {accuracy:.2f}%", ""]
        try:
            with open(self.log_file_path, 'a') as f: f.write("\n".join(log_lines))
        except Exception as e:
            logging.error(f"无法将准确率写入日志文件: {e}")

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
            'classifier_head': model_to_save.classifier_head.state_dict(),
            'answer_mapping': self.answer_to_label,
        }
        torch.save(checkpoint, other_params_path)
        logging.info(f"模型已保存至 {self.config.SAVE_DIR}")
