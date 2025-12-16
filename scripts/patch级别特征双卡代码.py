import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import T5TokenizerFast, T5ForConditionalGeneration, BertConfig, BertTokenizerFast
from transformers import BertModel
import math
import os
import numpy as np # 需要引入 numpy

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError("请先安装 PEFT 库: pip install peft")

# ==============================================================================
# --- 基础模块 (无改动) ---
# ==============================================================================
class MultiHeadAttention(nn.Module):
    # ... (保持不变)
    def __init__(self, hidden_size, num_heads, dropout_prob, is_cross_attention=False, encoder_hidden_size=None):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size)
        key_in_dim   = encoder_hidden_size if is_cross_attention else hidden_size
        value_in_dim = encoder_hidden_size if is_cross_attention else hidden_size
        self.key   = nn.Linear(key_in_dim, hidden_size)
        self.value = nn.Linear(value_in_dim, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, query, key, value, attention_mask=None):
        B, N, C = query.shape; Bk, S, Ck = key.shape
        assert B == Bk, "batch size mismatch"
        q = self.query(query).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(key).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(value).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None: attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.out(out)

class FeedForward(nn.Module):
    # ... (保持不变)
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x): return self.dropout(self.fc2(self.gelu(self.fc1(x))))

class DualQFormerLayer(nn.Module):
    # ... (保持不变)
    def __init__(self, hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size):
        super().__init__()
        self.self_attn   = MultiHeadAttention(hidden_size, num_heads, dropout_prob, is_cross_attention=False)
        self.norm1       = nn.LayerNorm(hidden_size)
        self.cross_attn  = MultiHeadAttention(hidden_size, num_heads, dropout_prob, is_cross_attention=True, encoder_hidden_size=encoder_hidden_size)
        self.norm_cross  = nn.LayerNorm(hidden_size)
        self.ffn         = FeedForward(hidden_size, intermediate_size, dropout_prob)
        self.norm2       = nn.LayerNorm(hidden_size)
    def forward(self, joint_embeds, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask):
        sa_out = self.self_attn(joint_embeds, joint_embeds, joint_embeds, attention_mask=self_attn_mask)
        joint  = self.norm1(joint_embeds + sa_out)
        q = joint[:, :num_query_tokens, :]
        ca_out = self.cross_attn(q, encoder_hidden_states, encoder_hidden_states, attention_mask=encoder_attention_mask)
        q = self.norm_cross(q + ca_out)
        joint = torch.cat([q, joint[:, num_query_tokens:, :]], dim=1)
        ffn_out = self.ffn(joint)
        joint   = self.norm2(joint + ffn_out)
        return joint

class DualQFormerEncoder(nn.Module):
    # ... (保持不变)
    def __init__(self, num_layers, hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            DualQFormerLayer(hidden_size, num_heads, dropout_prob, intermediate_size, encoder_hidden_size)
            for _ in range(num_layers)
        ])
    def forward(self, joint_embeds, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask):
        x = joint_embeds
        for layer in self.layers:
            x = layer(x, num_query_tokens, encoder_hidden_states, encoder_attention_mask, self_attn_mask)
        return x

# ==============================================================================
# --- 主模型 (已修正并包含 Patch 级特征处理) ---
# ==============================================================================
class MultiModal_T5_Classifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 t5_model_path: str,
                 qformer_num_layers: int,
                 num_query_token: int,
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 blip2_model_path: str, # <-- BLIP-2 权重路径
                 # --- 新增的参数 ---
                 clip_patch_dim: int = 1024, # 原始特征维度 (1024)
                 qformer_input_dim: int = 768, # Q-Former 期望的特征维度 (768)
                 # --------------------
                 audio_feat_dim: int = 512,
                 bert_tokenizer_name_or_path: str = "bert-base-uncased"):
        super().__init__()
        self.d_q = 768 # Q-Former/BERT 隐藏层维度
        self.num_video_queries = num_query_token
        self.num_audio_queries = num_query_token
        self.qformer_input_dim = qformer_input_dim # 768
        self.clip_patch_dim = clip_patch_dim # 1024

        # ... (T5 初始化部分) ...
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_path)
        t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
        self.d_t5 = t5_base_model.config.hidden_size
        self.t5_model = get_peft_model(
            t5_base_model,
            LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q", "v"], bias="none", task_type=TaskType.SEQ_2_SEQ_LM
            )
        )
        logging.info("T5 模型已应用 LoRA。")
        self.print_trainable_parameters(self.t5_model)

        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_tokenizer_name_or_path)

        # --- (开始修正：恢复丢失的代码块) ---
        logging.info(f"正在从本地路径加载 BLIP-2 预训练权重: {blip2_model_path}")
        blip2_checkpoint = torch.load(blip2_model_path, map_location="cpu")
        
        # 这一行是关键，它定义了 blip2_state_dict
        blip2_state_dict = blip2_checkpoint['model']

        # 1. 初始化我们的 BertEmbeddings 模块
        bert_config = BertConfig.from_pretrained(bert_tokenizer_name_or_path)
        self.bert_embedding = BertModel(config=bert_config).embeddings
        
        # 2. 准备一个字典来加载
        bert_embedding_weights = {}
        
        # 3. 智能地从 blip2_state_dict 提取权重
        for k, v in blip2_state_dict.items():
            if k.startswith('Qformer.bert.embeddings.'):
                new_key = k[len('Qformer.bert.embeddings.'):]
                if new_key != 'position_ids':
                    bert_embedding_weights[new_key] = v

        # 4. 手动检查备用键
        alt_keys_map = {
            'Qformer.bert.word_embeddings.weight': 'word_embeddings.weight',
            'Qformer.bert.position_embeddings.weight': 'position_embeddings.weight'
        }
        
        for blip_key, hf_key in alt_keys_map.items():
            if hf_key not in bert_embedding_weights and blip_key in blip2_state_dict:
                logging.info(f"找到备用键: {blip_key} -> {hf_key}")
                bert_embedding_weights[hf_key] = blip2_state_dict[blip_key]
        
        # 5. 加载权重
        msg = self.bert_embedding.load_state_dict(bert_embedding_weights, strict=False)
        logging.info(f"加载 BERT Embeddings: {msg}")
        
        if msg.missing_keys and not all(k == 'position_ids' for k in msg.missing_keys):
                 logging.warning(f"加载 BERT Embeddings 时丢失了非预期的权重: {msg.missing_keys}")
        
        if msg.unexpected_keys:
            logging.warning(f"加载 BERT Embeddings 时发现了非预期的权重: {msg.unexpected_keys}")
            
        del blip2_checkpoint # 释放内存
        # --- (结束修正) ---
        
        # 初始化 Q-Formers
        num_heads = 12; attn_drop = 0.1; inter_size = 4 * self.d_q
        
        # --- (这是我们新加的降维层，保持不变) ---
        if self.clip_patch_dim != self.qformer_input_dim:
            self.video_patch_downsampler = nn.Linear(self.clip_patch_dim, self.qformer_input_dim, bias=True)
            logging.info(f"已创建视频 Patch 降维层: {self.clip_patch_dim} -> {self.qformer_input_dim}")
        else:
            self.video_patch_downsampler = nn.Identity()
            logging.info("视频 Patch 维度匹配，无需降维层。")

        self.video_qformer = DualQFormerEncoder(
            num_layers=qformer_num_layers, hidden_size=self.d_q, num_heads=num_heads,
            dropout_prob=attn_drop, intermediate_size=inter_size, encoder_hidden_size=self.qformer_input_dim # 应该是 768
        )
        self.audio_qformer = DualQFormerEncoder(
            num_layers=qformer_num_layers, hidden_size=self.d_q, num_heads=num_heads,
            dropout_prob=attn_drop, intermediate_size=inter_size, encoder_hidden_size=self.d_q
        )
        
        # --- (这部分现在可以正常工作了) ---
        logging.info("正在将 BLIP-2 Q-Former 权重迁移到 Video Q-Former...")
        self._load_blip2_qformer_weights(self.video_qformer, blip2_state_dict)
        logging.info("正在将 BLIP-2 Q-Former 权重迁移到 Audio Q-Former...")
        self._load_blip2_qformer_weights(self.audio_qformer, blip2_state_dict)

        # 释放 BLIP-2 权重字典内存
        del blip2_state_dict
        
        self.video_query_tokens = nn.Parameter(torch.zeros(1, self.num_video_queries, self.d_q))
        self.audio_query_tokens = nn.Parameter(torch.zeros(1, self.num_audio_queries, self.d_q))
        nn.init.trunc_normal_(self.video_query_tokens, std=0.02)
        nn.init.trunc_normal_(self.audio_query_tokens, std=0.02)

        self.audio_upscaler = nn.Linear(audio_feat_dim, self.d_q, bias=True)

        self.proj_video = nn.Linear(self.d_q, self.d_t5, bias=True)
        self.proj_audio = nn.Linear(self.d_q, self.d_t5, bias=True)
        self.classifier_head = nn.Linear(self.d_t5, num_classes)

    def _load_blip2_qformer_weights(self, qformer_encoder, blip2_state_dict):
        """
        辅助函数，从预训练的 BLIP-2 检查点加载权重到我们的 Q-Former。
        这会同时加载自注意力和交叉注意力的权重。
        """
        qformer_keys = {}
        for k, v in blip2_state_dict.items():
            if k.startswith('Qformer.bert.'):
                # 移除 'Qformer.bert.' 前缀
                new_key = k[len('Qformer.bert.'):]
                
                # --- 关键的键名映射 ---
                new_key = new_key.replace('encoder.layer', 'layers')
                new_key = new_key.replace('attention.self', 'self_attn')
                new_key = new_key.replace('attention.output.dense', 'self_attn.out')
                new_key = new_key.replace('attention.output.LayerNorm', 'norm1')
                new_key = new_key.replace('crossattention.self.query', 'cross_attn.query')
                new_key = new_key.replace('crossattention.self.key', 'cross_attn.key')
                new_key = new_key.replace('crossattention.self.value', 'cross_attn.value')
                new_key = new_key.replace('crossattention.output.dense', 'cross_attn.out')
                new_key = new_key.replace('crossattention.output.LayerNorm', 'norm_cross')
                new_key = new_key.replace('intermediate.dense', 'ffn.fc1')
                new_key = new_key.replace('output.dense', 'ffn.fc2')
                new_key = new_key.replace('output.LayerNorm', 'norm2')
                # -------------------------

                # 确保映射后的键存在于模型中
                if new_key in qformer_encoder.state_dict():
                    qformer_keys[new_key] = v
                
        msg = qformer_encoder.load_state_dict(qformer_keys, strict=False)
        logging.info(f"从 BLIP-2 加载 Q-Former 权重: {msg}")

    def print_trainable_parameters(self, model):
        trainable_params, all_params = 0, 0
        for _, p in model.named_parameters():
            all_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print(f"可训练参数: {trainable_params} / {all_params} ({100.0 * trainable_params / max(1, all_params):.2f}%)")

    def _build_joint_and_mask(self, query_tokens, bert_tokens):
        B = query_tokens.size(0)
        guide_text_embeds = self.bert_embedding(bert_tokens.input_ids)
        joint = torch.cat([query_tokens, guide_text_embeds], dim=1)
        ones_q = torch.ones((B, query_tokens.size(1)), dtype=bert_tokens.attention_mask.dtype, device=query_tokens.device)
        valid = torch.cat([ones_q, bert_tokens.attention_mask], dim=1).float()
        self_attn_mask = (1.0 - valid).unsqueeze(1).unsqueeze(2) * -10000.0
        return joint, self_attn_mask

    def forward(self,
                video_features: torch.Tensor, # 预期形状: (B, T, P, 1024)
                audio_features: torch.Tensor,
                questions: list[str]) -> torch.Tensor:
        device = video_features.device
        B = video_features.size(0)
        
        # --- (这是我们新加的 Patch 级特征处理，保持不变) ---
        # 1. 降维: (B, T, P, 1024) -> (B, T, P, 768)
        video_features_down = self.video_patch_downsampler(video_features)
        
        # 2. 展平: (B, T, P, 768) -> (B, T*P, 768)
        video_features_flat = video_features_down.view(B, -1, self.qformer_input_dim)
        
        # ------------------------------------

        bert_tokens = self.bert_tokenizer(
            questions, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)

        video_query = self.video_query_tokens.expand(B, -1, -1)
        video_joint, video_self_mask = self._build_joint_and_mask(video_query, bert_tokens)
        
        # 3. 修正注意力掩码的计算
        video_atts = torch.ones(video_features_flat.size()[:-1], dtype=torch.long, device=device) 
        extended_video_atts = (1.0 - video_atts.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        
        video_joint_out = self.video_qformer(
            joint_embeds=video_joint,
            num_query_tokens=self.num_video_queries,
            encoder_hidden_states=video_features_flat, # 使用展平后的特征
            encoder_attention_mask=extended_video_atts,
            self_attn_mask=video_self_mask
        )
        video_query_out = video_joint_out[:, :self.num_video_queries, :]

        # ... (Audio 和 T5 部分保持不变) ...

        audio_features_up = self.audio_upscaler(audio_features)
        audio_query = self.audio_query_tokens.expand(B, -1, -1)
        audio_joint, audio_self_mask = self._build_joint_and_mask(audio_query, bert_tokens)
        audio_atts = torch.ones(audio_features_up.size()[:-1], dtype=torch.long, device=device)
        extended_audio_atts = (1.0 - audio_atts.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        audio_joint_out = self.audio_qformer(
            joint_embeds=audio_joint,
            num_query_tokens=self.num_audio_queries,
            encoder_hidden_states=audio_features_up,
            encoder_attention_mask=extended_audio_atts,
            self_attn_mask=audio_self_mask
        )
        audio_query_out = audio_joint_out[:, :self.num_audio_queries, :]

        video_q_for_t5 = self.proj_video(video_query_out)
        audio_q_for_t5 = self.proj_audio(audio_query_out)

        t5_tokens = self.t5_tokenizer(
            questions, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)
        t5_text_embeds = self.t5_model.get_input_embeddings()(t5_tokens.input_ids)

        inputs_t5 = torch.cat([video_q_for_t5, audio_q_for_t5, t5_text_embeds], dim=1)
        q_atts = torch.ones((B, self.num_video_queries + self.num_audio_queries), dtype=torch.long, device=device)
        final_t5_attention_mask = torch.cat([q_atts, t5_tokens.attention_mask], dim=1)

        encoder_outputs = self.t5_model.encoder(
            inputs_embeds=inputs_t5,
            attention_mask=final_t5_attention_mask,
            return_dict=True
        ).last_hidden_state

        cls_repr = encoder_outputs[:, 0, :] 
        logits = self.classifier_head(cls_repr)
        return logits




        import os
import json
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import argparse
import logging
from torch.optim import AdamW
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup

# --- (DDP 导入) ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# --- 从 model.py 导入模型类 ---
from model import MultiModal_T5_Classifier

# ==============================================================================
# --- 配置区域 (已更新梯度累积) ---
# ==============================================================================
class TrainConfig:
    # **1. 数据和模型路径**
    TRAIN_ANNOTATIONS_PATH = "/home/qinxilin/baseline/annots/music_avqa/music_avqa_train.json"
    TEST_ANNOTATIONS_PATH  = "/home/qinxilin/baseline/annots/music_avqa/music_avqa_test.json"
    
    # --- (确保这是您的 Patch 级特征路径) ---
    VIDEO_FEATURES_DIR   = "/home/qinxilin/baseline/frame_ViT-L14@336px_patchlevel"
    AUDIO_FEATURES_DIR   = "/home/qinxilin/baseline/clap"
    
    T5_MODEL_PATH        = "/home/qinxilin/baseline/flan-t5-base"
    BLIP2_MODEL_PATH     = "/home/qinxilin/baseline/blip2_pretrained_flant5xl.pth" 

    # **2. 保存路径**
    SAVE_DIR = "/home/qinxilin/baseline/saved_models_patchlevel"
    LOG_DIR  = "/home/qinxilin/baseline/best_accuracy_v3clap_patchlevel"

    # **3. 训练超参数**
    EPOCHS          = 40
    
    # --- (GRAD_ACCUM 修改) ---
    # BATCH_SIZE 是“物理”全局 Batch Size
    # 设为 8 (每卡 4 个)，这是一个非常安全的设置
    BATCH_SIZE      = 8  
    # 累积 4 步
    GRADIENT_ACCUMULATION_STEPS = 4 
    # 最终有效 Batch Size = 8 (全局) * 4 (累积) = 32
    # --- (结束修改) ---
    
    LEARNING_RATE   = 5e-5
    WARMUP_RATIO    = 0.05
    NUM_GPUS        = 2   # 指定 GPU 数量

    # **4. 模型架构超参数**
    NUM_QUERY_TOKEN    = 32
    QFORMER_NUM_LAYERS = 6
    LORA_R             = 16
    LORA_ALPHA         = 32
    LORA_DROPOUT       = 0.05

# ==============================================================================
# --- DDP 辅助函数 (保持不变) ---
# ==============================================================================
def setup(rank, world_size):
    """初始化 DDP 进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 任意未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"DDP setup complete for rank {rank} on device {torch.cuda.current_device()}")

def cleanup():
    """销毁 DDP 进程组"""
    dist.destroy_process_group()

# ==============================================================================
# --- Trainer 类 (已修改 'train' 方法) ---
# ==============================================================================
class Trainer:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # 日志文件
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(config.LOG_DIR, f"training_log_{timestamp}_rank{rank}.txt")
        logging.info(f"Rank {rank} 日志将记录在: {self.log_file_path}")
        
        if self.rank == 0:
            self.log_hyperparameters()

        # 数据与映射
        full_train_anns   = json.load(open(config.TRAIN_ANNOTATIONS_PATH, 'r'))
        self.full_test_anns = json.load(open(config.TEST_ANNOTATIONS_PATH, 'r'))
        self.answer_to_label = {ans: i for i, ans in enumerate(sorted(list(set(item['anser'] for item in full_train_anns))))}
        num_classes        = len(self.answer_to_label)
        
        # DDP 数据分片
        train_items_per_gpu = math.ceil(len(full_train_anns) / self.world_size)
        self.train_data = full_train_anns[self.rank * train_items_per_gpu : (self.rank + 1) * train_items_per_gpu]
        
        test_items_per_gpu = math.ceil(len(self.full_test_anns) / self.world_size)
        self.test_anns_local = self.full_test_anns[self.rank * test_items_per_gpu : (self.rank + 1) * test_items_per_gpu]
        
        logging.info(f"Rank {rank}: 加载了 {len(self.train_data)} 个训练样本, {len(self.test_anns_local)} 个测试样本。")
        
        # DDP Batch Size (计算每卡的物理 BS)
        if config.BATCH_SIZE % self.world_size != 0:
            raise ValueError(f"全局 BATCH_SIZE ({config.BATCH_SIZE}) 必须能被 world_size ({self.world_size}) 整除。")
        self.per_gpu_batch_size = config.BATCH_SIZE // self.world_size
        logging.info(f"Rank {rank}: 全局 BATCH_SIZE={config.BATCH_SIZE}, 本卡 BATCH_SIZE={self.per_gpu_batch_size}")
        logging.info(f"Rank {rank}: 梯度累积步数={config.GRADIENT_ACCUMULATION_STEPS}, 有效 BS={config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")


        # 实例化模型
        self.model = MultiModal_T5_Classifier(
            num_classes=num_classes,
            t5_model_path=config.T5_MODEL_PATH,
            qformer_num_layers=config.QFORMER_NUM_LAYERS,
            num_query_token=config.NUM_QUERY_TOKEN,
            lora_r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            blip2_model_path=config.BLIP2_MODEL_PATH
        ).to(self.device)
        
        # DDP 包裹模型
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)
        logging.info(f"Rank {rank}: 模型已包裹 DDP。")

        # 优化器 & 损失
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        # (GRAD_ACCUM 修改: 总步数需要除以累积步数)
        steps_per_epoch_physical = max(1, math.ceil(len(self.train_data) / self.per_gpu_batch_size))
        steps_per_epoch_effective = max(1, math.ceil(steps_per_epoch_physical / self.config.GRADIENT_ACCUMULATION_STEPS))
        
        self.total_training_steps = steps_per_epoch_effective * self.config.EPOCHS
        self.warmup_steps = int(self.config.WARMUP_RATIO * self.total_training_steps)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps
        )

        self.best_accuracy = 0.0
        if self.rank == 0:
            os.makedirs(config.SAVE_DIR, exist_ok=True)
        logging.info(f"Rank {rank}: Trainer 初始化完成。总优化器步数: {self.total_training_steps}")

    def log_hyperparameters(self):
        # (保持不变)
        log_message = "--- 实验配置 ---\n"
        for key in dir(self.config):
            if not key.startswith('__'):
                value = getattr(self.config, key)
                log_message += f"{key}: {value}\n"
        log_message += "-------------------\n\n"
        with open(self.log_file_path, 'w') as f:
            f.write(log_message)

    def _prepare_batch(self, annotations):
        # (保持不变, 包含 .mp4.npy 修正)
        video_feats_list, audio_feats_list, questions, labels = [], [], [], []

        if not hasattr(self, '_debug_printed'): 
            if annotations:
                ann_sample = annotations[0]
                video_id_sample = ann_sample['video_id']
                video_path_sample = os.path.join(self.config.VIDEO_FEATURES_DIR, f"{video_id_sample}.mp4.npy")
                audio_path_sample = os.path.join(self.config.AUDIO_FEATURES_DIR, f"{video_id_sample}.npy")
                logging.info(f"Rank {self.rank} DEBUG: [视频] 查找 {video_path_sample} (存在: {os.path.exists(video_path_sample)})")
                logging.info(f"Rank {self.rank} DEBUG: [音频] 查找 {audio_path_sample} (存在: {os.path.exists(audio_path_sample)})")
            self._debug_printed = True
        
        for ann in annotations:
            video_id, question, answer = ann['video_id'], ann['question_content'], ann['anser']
            video_path = os.path.join(self.config.VIDEO_FEATURES_DIR, f"{video_id}.mp4.npy")
            audio_path = os.path.join(self.config.AUDIO_FEATURES_DIR, f"{video_id}.npy")
            
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                continue
            
            try:
                video_feats_list.append(torch.from_numpy(np.load(video_path)).float())
                audio_feats_list.append(torch.from_numpy(np.load(audio_path)).float())
                questions.append(question)
                labels.append(self.answer_to_label.get(answer, -1))
            except Exception as e:
                logging.warning(f"Rank {self.rank}: 加载文件时出错 (ID: {video_id}): {e}. 跳过。")
                continue

        if not questions:
            return None
            
        try:
            video_batch  = nn.utils.rnn.pad_sequence(video_feats_list, batch_first=True).to(self.device)
            audio_batch  = nn.utils.rnn.pad_sequence(audio_feats_list, batch_first=True).to(self.device)
            labels_batch = torch.tensor(labels, dtype=torch.long).to(self.device)
            return video_batch, audio_batch, questions, labels_batch
        except Exception as e:
            logging.error(f"Rank {self.rank}: pad_sequence 出错: {e}. 特征维度可能不一致。")
            return None

    @staticmethod
    def _parse_types(annotations):
        # (保持不变)
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

    # ==========================================================================
    # --- (GRAD_ACCUM) 'train' 方法已完全重写 ---
    # ==========================================================================
    def train(self):
        global_step = 0
        accum_steps = self.config.GRADIENT_ACCUMULATION_STEPS
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            np.random.shuffle(self.train_data)
            
            progress_bar = tqdm(
                range(0, len(self.train_data), self.per_gpu_batch_size), 
                desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Rank {self.rank}]",
                disable=(self.rank != 0)
            )
            
            # 在 epoch 开始时清零
            self.optimizer.zero_grad() 
            
            steps_in_epoch = len(progress_bar)
            
            for i, step in enumerate(progress_bar):
                batch_anns = self.train_data[step : step + self.per_gpu_batch_size]
                batch = self._prepare_batch(batch_anns)
                if batch is None:
                    continue

                video_feats, audio_feats, questions, labels = batch

                is_last_step = (i == steps_in_epoch - 1)
                is_update_step = (i + 1) % accum_steps == 0 or is_last_step

                # DDP + 梯度累积时:
                # 只有在最后一步(is_update_step)才同步梯度
                if not is_update_step and self.world_size > 1:
                    with self.model.no_sync():
                        logits = self.model(video_feats, audio_feats, questions)
                        loss   = self.criterion(logits, labels)
                        # 归一化 Loss
                        loss = loss / accum_steps 
                        loss.backward()
                else:
                    # 这是最后一步累积 (或 epoch 的最后一步)
                    # 正常前向和反向传播，DDP 会在此处自动同步
                    logits = self.model(video_feats, audio_feats, questions)
                    loss   = self.criterion(logits, labels)
                    # 归一化 Loss
                    loss = loss / accum_steps 
                    loss.backward()

                # 只有在累积了足够的步数后 (或 epoch 结束时)，才更新权重
                if is_update_step:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1

                    if self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        # loss.item() 乘以 accum_steps 来显示“平均”的 batch loss
                        progress_bar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}", lr=f"{current_lr:.2e}")

            # 评估 (所有进程都参与)
            self.evaluate(epoch)
            
            # 等待所有进程完成此 epoch
            dist.barrier()
    # ==========================================================================
    # --- (结束 'train' 方法修改) ---
    # ==========================================================================


    def evaluate(self, epoch):
        # (保持不变)
        self.model.eval()
        per_type = {
            ("Audio", "Counting"):      {"correct": 0, "total": 0},
            ("Audio", "Comparative"):  {"correct": 0, "total": 0},
            ("Visual", "Counting"):    {"correct": 0, "total": 0},
            ("Visual", "Location"):    {"correct": 0, "total": 0},
            ("Audio-Visual", "Existential"): {"correct": 0, "total": 0},
            ("Audio-Visual", "Counting"):    {"correct": 0, "total": 0},
            ("Audio-Visual", "Location"):    {"correct": 0, "total": 0},
            ("Audio-Visual", "Comparative"): {"correct": 0, "total": 0},
            ("Audio-Visual", "Temporal"):    {"correct": 0, "total": 0},
        }
        total_correct, total_samples = 0, 0
        
        eval_progress_bar = tqdm(
            range(0, len(self.test_anns_local), self.per_gpu_batch_size), 
            desc=f"Evaluating [Rank {self.rank}]",
            disable=(self.rank != 0)
        )

        with torch.no_grad():
            for i in eval_progress_bar:
                batch_anns = self.test_anns_local[i : i + self.per_gpu_batch_size]
                if not batch_anns: continue
                
                full_batch_anns_indices = range(self.rank * math.ceil(len(self.full_test_anns) / self.world_size) + i, 
                                                self.rank * math.ceil(len(self.full_test_anns) / self.world_size) + i + len(batch_anns))
                full_batch_anns = [self.full_test_anns[idx] for idx in full_batch_anns_indices]
                types_this_batch = self._parse_types(full_batch_anns)

                batch = self._prepare_batch(batch_anns)
                if batch is None: continue
                video_feats, audio_feats, questions, labels = batch

                valid_mask = labels != -1
                if not valid_mask.any(): continue

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
                    key = types_this_batch[k]
                    if key in per_type:
                        per_type[key]["total"] += 1
                        if predictions[k].item() == labels[valid_mask][k].item():
                            per_type[key]["correct"] += 1

        # DDP 同步
        stats_list = [total_correct, total_samples]
        type_keys_ordered = sorted(per_type.keys())
        for key in type_keys_ordered:
            stats_list.append(per_type[key]["correct"])
            stats_list.append(per_type[key]["total"])
        stats_tensor = torch.tensor(stats_list, dtype=torch.float32).to(self.device)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        # 只有 Rank 0 打印、记录和保存
        if self.rank == 0:
            synced_stats = stats_tensor.cpu().numpy()
            total_correct = synced_stats[0]
            total_samples = synced_stats[1]
            
            idx = 2
            for key in type_keys_ordered:
                per_type[key]["correct"] = synced_stats[idx]
                per_type[key]["total"] = synced_stats[idx+1]
                idx += 2
            
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

            header = f"-------------- Epoch: {epoch+1}, Accuracy: {accuracy:.4f}%{'   <-- New Best!' if is_best else ''} --------------"
            print("\n" + header)
            print("Audio QA:")
            print(f"  Count : {_safe_acc(audio_count['correct'], audio_count['total']):.2f}%   (n={audio_count['total']})")
            print(f"  Comp  : {_safe_acc(audio_comp['correct'],  audio_comp['total']):.2f}%   (n={audio_comp['total']})")
            print(f"  Avg   : {audio_avg:.2f}%   (n={audio_avg_n})")
            print("Visual QA:")
            print(f"  Count : {_safe_acc(visual_count['correct'], visual_count['total']):.2f}%   (n={visual_count['total']})")
            print(f"  Local : {_safe_acc(visual_loc['correct'],   visual_loc['total']):.2f}%   (n={visual_loc['total']})")
            print(f"  Avg   : {visual_avg:.2f}%   (n={visual_avg_n})")
            print("Audio-Visual QA:")
            print(f"  Exist : {_safe_acc(av_exist['correct'], av_exist['total']):.2f}%   (n={av_exist['total']})")
            print(f"  Count : {_safe_acc(av_count['correct'], av_count['total']):.2f}%   (n={av_count['total']})")
            print(f"  Local : {_safe_acc(av_loc['correct'],   av_loc['total']):.2f}%   (n={av_loc['total']})")
            print(f"  Comp  : {_safe_acc(av_comp['correct'],  av_comp['total']):.2f}%   (n={av_comp['total']})")
            print(f"  Temp  : {_safe_acc(av_temp['correct'],  av_temp['total']):.2f}%   (n={av_temp['total']})")
            print(f"  Avg   : {av_avg:.2f}%   (n={av_avg_n})")
            print(f"\nOverall: {accuracy:.2f}%\n")

            log_lines = [header, "Audio:", f"  Count : {_safe_acc(audio_count['correct'], audio_count['total']):.2f}%   (n={audio_count['total']})", f"  Comp  : {_safe_acc(audio_comp['correct'],  audio_comp['total']):.2f}%   (n={audio_comp['total']})", f"  Avg   : {audio_avg:.2f}%   (n={audio_avg_n})", "Visual:", f"  Count : {_safe_acc(visual_count['correct'], visual_count['total']):.2f}%   (n={visual_count['total']})", f"  Local : {_safe_acc(visual_loc['correct'],   visual_loc['total']):.2f}%   (n={visual_loc['total']})", f"  Avg   : {visual_avg:.2f}%   (n={visual_avg_n})", "Audio-Visual:", f"  Exist : {_safe_acc(av_exist['correct'], av_exist['total']):.2f}%   (n={av_exist['total']})", f"  Count : {_safe_acc(av_count['correct'], av_count['total']):.2f}%   (n={av_count['total']})", f"  Local : {_safe_acc(av_loc['correct'],   av_loc['total']):.2f}%   (n={av_loc['total']})", f"  Comp  : {_safe_acc(av_comp['correct'],  av_comp['total']):.2f}%   (n={av_comp['total']})", f"  Temp  : {_safe_acc(av_temp['correct'],  av_temp['total']):.2f}%   (n={av_temp['total']})", f"  Avg   : {av_avg:.2f}%   (n={av_avg_n})", f"Overall: {accuracy:.2f}%", ""]
            try:
                with open(self.log_file_path, 'a') as f: f.write("\n".join(log_lines))
            except Exception as e:
                logging.error(f"Rank {self.rank}: 无法将准确率写入日志文件: {e}")

            if is_best:
                self.best_accuracy = accuracy
                logging.info(f"Rank {self.rank}: 发现新的最佳准确率: {self.best_accuracy:.2f}%。正在保存模型...")
                self.save_model()

    def save_model(self):
        # (保持不变, 仅 Rank 0 保存)
        if self.rank != 0:
            return
            
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
            'video_patch_downsampler': getattr(model_to_save, 'video_patch_downsampler', None).state_dict() if hasattr(model_to_save, 'video_patch_downsampler') else None,
        }
        torch.save(checkpoint, other_params_path)
        logging.info(f"Rank {self.rank}: 模型已保存至 {self.config.SAVE_DIR}")

# ==============================================================================
# --- DDP 启动器 (保持不变) ---
# ==============================================================================
def train_worker(rank, world_size, config):
    """
    DDP 的工作进程函数
    """
    try:
        setup(rank, world_size)
        trainer = Trainer(config, rank, world_size)
        trainer.train()
    except Exception as e:
        logging.error(f"Rank {rank} 发生致命错误: {e}", exc_info=True)
    finally:
        cleanup()

def main():
    config = TrainConfig()
    world_size = config.NUM_GPUS
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Rank %(process)d] - %(levelname)s - %(message)s')
    
    if world_size <= 0:
         raise ValueError("请在 TrainConfig 中设置 NUM_GPUS >= 1")
    elif world_size == 1:
        logging.info("GPU 数量 = 1, 在单卡模式下运行...")
        # (我们为 DDP=1 的情况保留 train_worker)
        train_worker(0, 1, config)
    else:
        logging.info(f"启动 {world_size} 个 GPU 的 DDP 训练...")
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )

if __name__ == '__main__':
    main()