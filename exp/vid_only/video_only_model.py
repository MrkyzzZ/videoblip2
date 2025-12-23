import torch
import torch.nn as nn
import logging
from transformers import T5TokenizerFast, T5ForConditionalGeneration, BertConfig, BertTokenizerFast, BertModel
from transformers.modeling_outputs import BaseModelOutput

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError("请先安装 PEFT 库: pip install peft")

from models.base_modules import DualQFormerEncoder


class VideoOnly_T5_Captioner(nn.Module):
    """与原模型等价的 T5 Captioner，但仅使用视频特征，不加载音频分支。"""

    def __init__(
        self,
        num_classes: int,
        t5_model_path: str,
        qformer_num_layers: int,
        num_query_token: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        blip2_model_path: str,
        video_feat_dim: int = 768,
        clip_patch_dim: int = 1024,
        qformer_input_dim: int = 768,
        bert_tokenizer_name_or_path: str = "bert-base-uncased",
    ):
        super().__init__()

        self.d_q = 768
        self.num_video_queries = num_query_token
        self.clip_patch_dim = clip_patch_dim
        self.qformer_input_dim = qformer_input_dim

        # 初始化 T5
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_path)
        t5_base_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
        self.d_t5 = t5_base_model.config.hidden_size
        self.t5_model = get_peft_model(
            t5_base_model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q", "v"],
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            ),
        )
        logging.info("T5 模型已应用 LoRA (视频-only)。")
        self.print_trainable_parameters(self.t5_model)

        # 初始化 BERT tokenizer 与 embedding
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_tokenizer_name_or_path)
        self._load_blip2_embeddings(blip2_model_path, bert_tokenizer_name_or_path)

        # Q-Former 设置
        num_heads = 12
        attn_drop = 0.1
        inter_size = 4 * self.d_q

        # Patch 特征降维
        if self.clip_patch_dim != self.qformer_input_dim:
            self.video_patch_downsampler = nn.Linear(self.clip_patch_dim, self.qformer_input_dim)
            nn.init.xavier_uniform_(self.video_patch_downsampler.weight)
            nn.init.zeros_(self.video_patch_downsampler.bias)
        else:
            self.video_patch_downsampler = nn.Identity()

        self.video_qformer = DualQFormerEncoder(
            num_layers=qformer_num_layers,
            hidden_size=self.d_q,
            num_heads=num_heads,
            dropout_prob=attn_drop,
            intermediate_size=inter_size,
            encoder_hidden_size=self.qformer_input_dim,
        )

        logging.info("正在将 BLIP-2 Q-Former 权重迁移到 Video Q-Former (无音频分支)...")
        self._load_blip2_qformer_weights(self.video_qformer, blip2_model_path)

        # 可学习查询
        self.video_query_tokens = nn.Parameter(torch.zeros(1, self.num_video_queries, self.d_q))
        nn.init.trunc_normal_(self.video_query_tokens, std=0.02)

        # 投影到 T5 空间
        self.proj_video = nn.Linear(self.d_q, self.d_t5, bias=True)

    def _load_blip2_embeddings(self, blip2_model_path, bert_tokenizer_name_or_path):
        logging.info(f"正在从本地路径加载 BLIP-2 预训练权重: {blip2_model_path}")
        blip2_checkpoint = torch.load(blip2_model_path, map_location="cpu")
        blip2_state_dict = blip2_checkpoint["model"]

        bert_config = BertConfig.from_pretrained(bert_tokenizer_name_or_path)
        self.bert_embedding = BertModel(config=bert_config).embeddings

        bert_embedding_weights = {}
        for k, v in blip2_state_dict.items():
            if k.startswith("Qformer.bert.embeddings."):
                new_key = k[len("Qformer.bert.embeddings.") :]
                if new_key != "position_ids":
                    bert_embedding_weights[new_key] = v

        alt_keys_map = {
            "Qformer.bert.word_embeddings.weight": "word_embeddings.weight",
            "Qformer.bert.position_embeddings.weight": "position_embeddings.weight",
        }
        for blip_key, hf_key in alt_keys_map.items():
            if hf_key not in bert_embedding_weights and blip_key in blip2_state_dict:
                logging.info(f"找到备用键: {blip_key} -> {hf_key}")
                bert_embedding_weights[hf_key] = blip2_state_dict[blip_key]

        msg = self.bert_embedding.load_state_dict(bert_embedding_weights, strict=False)
        logging.info(f"加载 BERT Embeddings: {msg}")

        if msg.missing_keys and not all(k == "position_ids" for k in msg.missing_keys):
            logging.warning(f"加载 BERT Embeddings 时丢失了非预期的权重: {msg.missing_keys}")
        if msg.unexpected_keys:
            logging.warning(f"加载 BERT Embeddings 时发现了非预期的权重: {msg.unexpected_keys}")

        del blip2_checkpoint

    def _load_blip2_qformer_weights(self, qformer_encoder, blip2_model_path):
        blip2_checkpoint = torch.load(blip2_model_path, map_location="cpu")
        blip2_state_dict = blip2_checkpoint["model"]

        qformer_keys = {}
        for k, v in blip2_state_dict.items():
            if k.startswith("Qformer.bert."):
                new_key = k[len("Qformer.bert.") :]
                new_key = new_key.replace("encoder.layer", "layers")
                new_key = new_key.replace("attention.self", "self_attn")
                new_key = new_key.replace("attention.output.dense", "self_attn.out")
                new_key = new_key.replace("attention.output.LayerNorm", "norm1")
                new_key = new_key.replace("crossattention.self.query", "cross_attn.query")
                new_key = new_key.replace("crossattention.self.key", "cross_attn.key")
                new_key = new_key.replace("crossattention.self.value", "cross_attn.value")
                new_key = new_key.replace("crossattention.output.dense", "cross_attn.out")
                new_key = new_key.replace("crossattention.output.LayerNorm", "norm_cross")
                new_key = new_key.replace("intermediate.dense", "ffn.fc1")
                new_key = new_key.replace("output.dense", "ffn.fc2")
                new_key = new_key.replace("output.LayerNorm", "norm2")
                if new_key in qformer_encoder.state_dict():
                    qformer_keys[new_key] = v

        qformer_encoder.load_state_dict(qformer_keys, strict=False)
        del blip2_checkpoint

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

    def forward(
        self,
        video_features: torch.Tensor,
        video_attention_mask: torch.Tensor | None,
        questions: list[str],
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        generate: bool = True,
        generation_kwargs: dict | None = None,
    ):
        device = video_features.device
        B = video_features.size(0)

        # 处理 Patch 级特征
        if video_features.dim() == 4:
            video_features_down = self.video_patch_downsampler(video_features)
            video_features_flat = video_features_down.view(B, -1, self.qformer_input_dim)
        else:
            if video_features.size(-1) == self.clip_patch_dim:
                video_features_flat = self.video_patch_downsampler(video_features)
            elif video_features.size(-1) == self.qformer_input_dim:
                video_features_flat = video_features
            else:
                raise ValueError(
                    f"Unexpected video feature dim: {video_features.size(-1)}, expected {self.clip_patch_dim} or {self.qformer_input_dim}"
                )

        bert_tokens = self.bert_tokenizer(questions, padding="longest", truncation=True, return_tensors="pt").to(device)

        video_query = self.video_query_tokens.expand(B, -1, -1)
        video_joint, video_self_mask = self._build_joint_and_mask(video_query, bert_tokens)

        if video_attention_mask is None:
            video_atts = torch.ones(video_features_flat.size()[:-1], dtype=torch.long, device=device)
        else:
            if video_features.dim() == 4:
                P = video_features.size(2)
                video_atts = video_attention_mask.unsqueeze(-1).expand(-1, -1, P).reshape(B, -1).to(device)
            else:
                video_atts = video_attention_mask.to(device)
        extended_video_atts = (1.0 - video_atts.unsqueeze(1).unsqueeze(2).float()) * -10000.0

        video_joint_out = self.video_qformer(
            joint_embeds=video_joint,
            num_query_tokens=self.num_video_queries,
            encoder_hidden_states=video_features_flat,
            encoder_attention_mask=extended_video_atts,
            self_attn_mask=video_self_mask,
        )
        video_query_out = video_joint_out[:, : self.num_video_queries, :]

        video_q_for_t5 = self.proj_video(video_query_out)

        t5_tokens = self.t5_tokenizer(questions, padding="longest", truncation=True, return_tensors="pt").to(device)
        t5_text_embeds = self.t5_model.get_input_embeddings()(t5_tokens.input_ids)

        inputs_t5 = torch.cat([video_q_for_t5, t5_text_embeds], dim=1)
        q_atts = torch.ones((B, self.num_video_queries), dtype=torch.long, device=device)
        final_t5_attention_mask = torch.cat([q_atts, t5_tokens.attention_mask], dim=1)

        encoder_hidden = self.t5_model.encoder(
            inputs_embeds=inputs_t5,
            attention_mask=final_t5_attention_mask,
            return_dict=True,
        ).last_hidden_state

        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        lm_outputs = None
        if labels is not None or decoder_input_ids is not None:
            lm_outputs = self.t5_model(
                encoder_outputs=encoder_outputs,
                attention_mask=final_t5_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True,
            )

        generated_ids = None
        generated_text = None
        if generate:
            gen_kwargs = generation_kwargs.copy() if generation_kwargs else {}
            gen_kwargs.setdefault("max_length", 24)
            gen_kwargs.setdefault("num_beams", 3)
            gen_kwargs.setdefault("length_penalty", 0.85)
            gen_kwargs.setdefault("repetition_penalty", 1.05)
            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=final_t5_attention_mask,
                **gen_kwargs,
            )
            generated_text = self.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return {
            "encoder_hidden_states": encoder_outputs,
            "encoder_attention_mask": final_t5_attention_mask,
            "lm_outputs": lm_outputs,
            "generated_ids": generated_ids,
            "generated_text": generated_text,
        }
