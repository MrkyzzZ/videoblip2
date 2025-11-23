import torch
import torch.nn as nn
import logging
from transformers import T5TokenizerFast, T5ForConditionalGeneration, BertConfig, BertTokenizerFast
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutput

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError("请先安装 PEFT 库: pip install peft")

from .base_modules import DualQFormerEncoder


class MultiModal_T5_Classifier(nn.Module):
    """多模态 T5 Caption 模型，支持音视频特征与文本联合生成"""
    def __init__(self,
                 num_classes: int,
                 t5_model_path: str,
                 qformer_num_layers: int,
                 num_query_token: int,
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 blip2_model_path: str,
                 video_feat_dim: int = 768,
                 audio_feat_dim: int = 512,
                 bert_tokenizer_name_or_path: str = "bert-base-uncased"):
        super().__init__()

        self.d_q = 768
        self.num_video_queries = num_query_token
        self.num_audio_queries = num_query_token

        # 初始化T5模型和tokenizer
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

        # 初始化BERT tokenizer
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_tokenizer_name_or_path)

        # 从BLIP-2加载预训练权重
        self._load_blip2_embeddings(blip2_model_path, bert_tokenizer_name_or_path)

        # 初始化Q-Formers
        num_heads = 12
        attn_drop = 0.1
        inter_size = 4 * self.d_q

        self.video_qformer = DualQFormerEncoder(
            num_layers=qformer_num_layers, hidden_size=self.d_q, num_heads=num_heads,
            dropout_prob=attn_drop, intermediate_size=inter_size, encoder_hidden_size=video_feat_dim
        )
        self.audio_qformer = DualQFormerEncoder(
            num_layers=qformer_num_layers, hidden_size=self.d_q, num_heads=num_heads,
            dropout_prob=attn_drop, intermediate_size=inter_size, encoder_hidden_size=self.d_q
        )

        # 从BLIP-2加载Q-Former权重
        logging.info("正在将 BLIP-2 Q-Former 权重迁移到 Video Q-Former...")
        self._load_blip2_qformer_weights(self.video_qformer, blip2_model_path)
        logging.info("正在将 BLIP-2 Q-Former 权重迁移到 Audio Q-Former...")
        self._load_blip2_qformer_weights(self.audio_qformer, blip2_model_path)

        # 初始化可学习参数
        self.video_query_tokens = nn.Parameter(torch.zeros(1, self.num_video_queries, self.d_q))
        self.audio_query_tokens = nn.Parameter(torch.zeros(1, self.num_audio_queries, self.d_q))
        nn.init.trunc_normal_(self.video_query_tokens, std=0.02)
        nn.init.trunc_normal_(self.audio_query_tokens, std=0.02)

        # 音频特征上采样器
        self.audio_upscaler = nn.Linear(audio_feat_dim, self.d_q, bias=True)

        # 投影层
        self.proj_video = nn.Linear(self.d_q, self.d_t5, bias=True)
        self.proj_audio = nn.Linear(self.d_q, self.d_t5, bias=True)

    def _load_blip2_embeddings(self, blip2_model_path, bert_tokenizer_name_or_path):
        """从BLIP-2检查点加载BERT embeddings权重"""
        logging.info(f"正在从本地路径加载 BLIP-2 预训练权重: {blip2_model_path}")
        blip2_checkpoint = torch.load(blip2_model_path, map_location="cpu")
        blip2_state_dict = blip2_checkpoint['model']

        # 初始化BERT embeddings模块
        bert_config = BertConfig.from_pretrained(bert_tokenizer_name_or_path)
        self.bert_embedding = BertModel(config=bert_config).embeddings

        # 提取embeddings权重
        bert_embedding_weights = {}

        # 处理标准键
        for k, v in blip2_state_dict.items():
            if k.startswith('Qformer.bert.embeddings.'):
                new_key = k[len('Qformer.bert.embeddings.'):]
                # 显式忽略 'position_ids' 缓冲区
                if new_key != 'position_ids':
                    bert_embedding_weights[new_key] = v

        # 处理备用键
        alt_keys_map = {
            'Qformer.bert.word_embeddings.weight': 'word_embeddings.weight',
            'Qformer.bert.position_embeddings.weight': 'position_embeddings.weight'
        }

        for blip_key, hf_key in alt_keys_map.items():
            if hf_key not in bert_embedding_weights and blip_key in blip2_state_dict:
                logging.info(f"找到备用键: {blip_key} -> {hf_key}")
                bert_embedding_weights[hf_key] = blip2_state_dict[blip_key]

        # 加载权重
        msg = self.bert_embedding.load_state_dict(bert_embedding_weights, strict=False)
        logging.info(f"加载 BERT Embeddings: {msg}")

        # 检查是否有意外的缺失权重
        if msg.missing_keys and not all(k == 'position_ids' for k in msg.missing_keys):
            logging.warning(f"加载 BERT Embeddings 时丢失了非预期的权重: {msg.missing_keys}")

        if msg.unexpected_keys:
            logging.warning(f"加载 BERT Embeddings 时发现了非预期的权重: {msg.unexpected_keys}")

        del blip2_checkpoint

    def _load_blip2_qformer_weights(self, qformer_encoder, blip2_model_path):
        """
        从预训练的 BLIP-2 检查点加载权重到Q-Former。
        这会同时加载自注意力和交叉注意力的权重。
        """
        blip2_checkpoint = torch.load(blip2_model_path, map_location="cpu")
        blip2_state_dict = blip2_checkpoint['model']

        qformer_keys = {}
        for k, v in blip2_state_dict.items():
            if k.startswith('Qformer.bert.'):
                new_key = k[len('Qformer.bert.'):]

                # 键名映射
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

                # 确保映射后的键存在于模型中
                if new_key in qformer_encoder.state_dict():
                    qformer_keys[new_key] = v

        msg = qformer_encoder.load_state_dict(qformer_keys, strict=False)

        del blip2_checkpoint

    def print_trainable_parameters(self, model):
        """打印模型的可训练参数信息"""
        trainable_params, all_params = 0, 0
        for _, p in model.named_parameters():
            all_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print(f"可训练参数: {trainable_params} / {all_params} ({100.0 * trainable_params / max(1, all_params):.2f}%)")

    def _build_joint_and_mask(self, query_tokens, bert_tokens):
        """构建联合嵌入和注意力掩码"""
        B = query_tokens.size(0)
        guide_text_embeds = self.bert_embedding(bert_tokens.input_ids)
        joint = torch.cat([query_tokens, guide_text_embeds], dim=1)
        ones_q = torch.ones((B, query_tokens.size(1)), dtype=bert_tokens.attention_mask.dtype, device=query_tokens.device)
        valid = torch.cat([ones_q, bert_tokens.attention_mask], dim=1).float()
        self_attn_mask = (1.0 - valid).unsqueeze(1).unsqueeze(2) * -10000.0
        return joint, self_attn_mask

    def forward(self,
                video_features: torch.Tensor,
                audio_features: torch.Tensor,
                questions: list[str],
                decoder_input_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                generate: bool = True,
                generation_kwargs: dict | None = None):
        """
        Args:
            video_features: 视频特征 [B, T_v, D_v]
            audio_features: 音频特征 [B, T_a, D_a]
            questions: 文本提示/问题
            decoder_input_ids: 训练阶段可显式传入 decoder 输入
            labels: teacher forcing 标签（若提供则返回 loss/logits）
            generate: 是否执行自回归生成
            generation_kwargs: 传递给 `T5.generate` 的其他参数（如 max_length, num_beams 等）

        Returns:
            dict，包含：
                - encoder_hidden_states: T5 encoder 输出
                - lm_outputs: 如果提供 labels/decoder_input_ids，返回 T5 的模型输出（含 loss/logits）
                - generated_ids: 若 generate=True，返回生成的 token ids
                - generated_text: 若 generate=True，返回解码后的字符串列表
        """
        device = video_features.device
        B = video_features.size(0)

        # 处理问题文本
        bert_tokens = self.bert_tokenizer(
            questions, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)

        # 视频Q-Former处理
        video_query = self.video_query_tokens.expand(B, -1, -1)
        video_joint, video_self_mask = self._build_joint_and_mask(video_query, bert_tokens)
        video_atts = torch.ones(video_features.size()[:-1], dtype=torch.long, device=device)
        extended_video_atts = (1.0 - video_atts.unsqueeze(1).unsqueeze(2).float()) * -10000.0
        video_joint_out = self.video_qformer(
            joint_embeds=video_joint,
            num_query_tokens=self.num_video_queries,
            encoder_hidden_states=video_features,
            encoder_attention_mask=extended_video_atts,
            self_attn_mask=video_self_mask
        )
        video_query_out = video_joint_out[:, :self.num_video_queries, :]

        # 音频Q-Former处理
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

        # 投影到T5空间
        video_q_for_t5 = self.proj_video(video_query_out)
        audio_q_for_t5 = self.proj_audio(audio_query_out)

        # T5编码器处理
        t5_tokens = self.t5_tokenizer(
            questions, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)
        t5_text_embeds = self.t5_model.get_input_embeddings()(t5_tokens.input_ids)

        # 拼接所有输入
        inputs_t5 = torch.cat([video_q_for_t5, audio_q_for_t5, t5_text_embeds], dim=1)
        q_atts = torch.ones((B, self.num_video_queries + self.num_audio_queries), dtype=torch.long, device=device)
        final_t5_attention_mask = torch.cat([q_atts, t5_tokens.attention_mask], dim=1)

        encoder_hidden = self.t5_model.encoder(
            inputs_embeds=inputs_t5,
            attention_mask=final_t5_attention_mask,
            return_dict=True
        ).last_hidden_state

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        lm_outputs = None
        if labels is not None or decoder_input_ids is not None:
            lm_outputs = self.t5_model(
                encoder_outputs=encoder_outputs,
                attention_mask=final_t5_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )

        generated_ids = None
        generated_text = None
        if generate:
            gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
            gen_kwargs.setdefault("max_length", 32)
            gen_kwargs.setdefault("num_beams", 1)
            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=final_t5_attention_mask,
                **gen_kwargs
            )
            generated_text = self.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return {
            "encoder_hidden_states": encoder_hidden,
            "lm_outputs": lm_outputs,
            "generated_ids": generated_ids,
            "generated_text": generated_text,
        }
