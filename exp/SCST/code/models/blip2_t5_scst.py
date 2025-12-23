"""
BLIP2-T5 SCST (Self-Critical Sequence Training) 模型实现
==========================================================

本文件从 mllm-video-captioner 项目中提取，展示了如何在BLIP2-T5模型中实现SCST训练。

核心思想：
- SCST是一种强化学习方法，直接优化CIDEr等评估指标
- 使用beam search采样多个候选caption
- 计算每个caption的CIDEr reward
- 使用self-critical baseline减小方差
- 通过REINFORCE算法计算梯度

使用前提：
- 必须有一个已经用CE（交叉熵）训练好的基础模型
- SCST是在CE预训练模型基础上进行微调

Reference: https://arxiv.org/abs/1612.00563
"""

import logging
import itertools

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
import numpy as np

from transformers import T5TokenizerFast
from peft import LoraConfig, TaskType, get_peft_model

# ============================================================================
# 必需的导入：CIDEr评估器和Tokenizer
# ============================================================================
# 这些需要从evaluation目录导入
from evaluation.cider import Cider
from evaluation.ptbtokenizer import PTBTokenizer


def tokenize(refs, cands, no_op=False):
    """
    对参考文本和候选文本进行PTB tokenization
    
    这个函数是SCST训练中计算CIDEr reward的前置步骤。
    PTB tokenization会将文本标准化，去除标点符号等，
    使得CIDEr计算更加准确。
    
    Args:
        refs: list of list of str，每个样本的多个参考caption
        cands: list of str，每个样本生成的候选caption
        no_op: bool，如果True则跳过tokenization（用于调试）
    
    Returns:
        refs: dict，tokenized后的参考文本
        cands: dict，tokenized后的候选文本
    """
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}
    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


class Blip2T5SCST(nn.Module):
    """
    BLIP2-T5 模型，支持CE训练和SCST训练两种模式。
    
    关键属性:
        scst (bool): 是否启用SCST训练模式
        beam_size (int): beam search的beam数量，也是每个样本采样的序列数
    
    SCST训练流程:
        1. 编码图像/视频特征
        2. 使用beam search生成多个候选caption
        3. 计算每个候选的log概率
        4. 计算每个候选的CIDEr reward
        5. 使用self-critical baseline计算损失
        6. 反向传播更新模型
    """

    def __init__(
        self,
        # ... 其他参数省略，重点关注SCST相关参数
        scst=False,              # 是否启用SCST训练
        beam_size=5,             # beam数量
        lora=False,              # 是否使用LoRA
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        max_txt_len=32,
        prompt="",
    ):
        """
        初始化BLIP2-T5-SCST模型
        
        关键参数说明:
            scst (bool): 
                - False: 使用标准CE训练
                - True: 使用SCST强化学习训练
            
            beam_size (int):
                - SCST模式下，每个样本会采样beam_size个候选caption
                - 这些候选用于计算reward和baseline
        """
        super().__init__()
        
        # ====================================================================
        # 保存SCST相关配置
        # ====================================================================
        self.scst = scst
        self.beam_size = beam_size
        self.max_txt_len = max_txt_len
        self.prompt = prompt
        
        # ====================================================================
        # 初始化模型组件（这里简化展示，实际需要完整初始化）
        # ====================================================================
        # self.visual_encoder = ...  # 视觉编码器（如EVA-ViT）
        # self.ln_vision = ...       # LayerNorm
        # self.Qformer = ...         # Q-Former
        # self.query_tokens = ...    # Query tokens
        # self.t5_model = ...        # T5语言模型
        # self.t5_tokenizer = ...    # T5 tokenizer
        # self.t5_proj = ...         # 投影层
        
        logging.info(f"SCST mode: {self.scst}, beam_size: {self.beam_size}")


    def forward(self, samples):
        """
        前向传播，支持CE和SCST两种模式
        
        Args:
            samples (dict): 包含以下键:
                - image: 输入图像/视频 tensor, shape: (B, C, T, H, W)
                - text_input: ground truth caption列表
        
        Returns:
            dict: 包含 'loss' 键的字典
        """
        
        # ====================================================================
        # CE训练模式 (Cross-Entropy Training)
        # ====================================================================
        if not self.scst:
            return self._forward_ce(samples)
        
        # ====================================================================
        # SCST训练模式 (Self-Critical Sequence Training)
        # ====================================================================
        else:
            return self._forward_scst(samples)


    def _forward_ce(self, samples):
        """
        标准交叉熵训练
        
        这是SCST的前置步骤，必须先用CE训练出一个基础模型。
        
        损失函数: CrossEntropyLoss(predicted_logits, ground_truth_tokens)
        """
        image = samples["image"]
        B, C, T, H, W = image.shape
        image = image.permute(0,2,1,3,4).contiguous()  # B C T H W --> B T C H W
        image = image.reshape(B*T,C,H,W).contiguous()  # B T C H W --> B*T C H W

        with self.maybe_autocast():
            # 1. 编码视觉特征
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # 2. Q-Former处理
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # 3. 投影到T5空间
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            # 4. 准备输入输出
            text = samples["text_input"]
            samples["text_input"] = [self.prompt] * B
            samples["text_output"] = text

        with self.maybe_autocast(dtype=torch.bfloat16):
            # 5. Tokenize
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            
            output_tokens = self.t5_tokenizer(
                samples["text_output"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            # 6. 创建标签（padding位置设为-100，不计入损失）
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            # 7. 组合embedding
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # 8. 计算CE损失
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

            return {"loss": loss}


    def _forward_scst(self, samples):
        """
        SCST (Self-Critical Sequence Training) 训练
        
        核心步骤:
        1. 使用beam search采样多个候选caption
        2. 计算每个候选的序列log概率
        3. 计算每个候选的CIDEr reward
        4. 使用self-critical baseline计算损失
        
        损失函数:
            L = -log_prob(seq) * (reward - baseline)
        
        其中:
            - log_prob(seq): 采样序列的log概率
            - reward: 采样序列的CIDEr分数
            - baseline: 同一batch内所有候选的平均CIDEr分数
        
        Reference: https://arxiv.org/abs/1612.00563
        """
        image = samples["image"]
        B, C, T, H, W = image.shape
        image = image.permute(0,2,1,3,4).contiguous()
        image = image.reshape(B*T,C,H,W).contiguous()

        with self.maybe_autocast():
            # ================================================================
            # Step 1: 编码视觉特征
            # ================================================================
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # ================================================================
            # Step 2: Q-Former处理
            # ================================================================
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # ================================================================
            # Step 3: 投影到T5空间
            # ================================================================
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            # 保存ground truth caption
            text = samples["text_input"]
            samples["text_input"] = [self.prompt] * B
            samples["text_output"] = text

        with self.maybe_autocast(dtype=torch.bfloat16):
            # ================================================================
            # Step 4: 准备输入embedding
            # ================================================================
            input_tokens = self.t5_tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # ================================================================
            # Step 5: Beam Search采样
            # 
            # 关键参数:
            # - num_beams: beam数量
            # - num_return_sequences: 返回的序列数量（设为与beam相同）
            # - return_dict_in_generate: 返回字典格式，包含scores
            # - output_scores: 输出每步的score，用于计算log概率
            # ================================================================
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=encoder_atts,
                do_sample=False,           # 不使用采样，使用beam search
                top_p=0.9,
                temperature=1,
                num_beams=self.beam_size,  # beam数量
                max_length=32,
                repetition_penalty=1.0,
                length_penalty=1.0,
                num_return_sequences=self.beam_size,  # 每个样本返回beam_size个序列
                return_dict_in_generate=True,
                output_scores=True,        # 关键：返回每步的score
            )

            # ================================================================
            # Step 6: 计算序列log概率
            # 
            # compute_transition_scores返回每个token的转移分数
            # 将所有token的分数求和，除以长度，得到平均log概率
            # ================================================================
            transition_scores = self.t5_model.compute_transition_scores(
                outputs.sequences, 
                outputs.scores, 
                outputs.beam_indices, 
                normalize_logits=False
            )

            # 计算有效输出长度（非padding部分）
            output_length = torch.sum(transition_scores < 0, dim=1)
            
            # 计算平均log概率（用长度归一化）
            sequences_scores = transition_scores.sum(dim=1) / (output_length ** 1.0)
            sequences_scores = sequences_scores.view(B, -1)  # [batch, num_beams]

            # ================================================================
            # Step 7: 解码生成的caption
            # ================================================================
            caps_gen = self.t5_tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            caps_gen = [text.strip() for text in caps_gen]
            
            # 扩展ground truth以匹配生成的数量
            # 每个ground truth重复beam_size次
            caps_gt = list(itertools.chain(
                *([c, ] * self.beam_size for c in samples["text_output"])
            ))
            caps_gt = [[c] for c in caps_gt]

            # ================================================================
            # Step 8: 计算CIDEr Reward
            # 
            # 使用PTB tokenization处理生成和参考caption
            # 然后计算CIDEr分数作为reward
            # ================================================================
            caps_gen_tokenized, caps_gt_tokenized = tokenize(caps_gt, caps_gen)
            
            # 计算CIDEr分数
            # compute_score返回 (平均分数, 每个样本的分数数组)
            reward = Cider().compute_score(caps_gt_tokenized, caps_gen_tokenized)[1]
            reward = reward.astype(np.float32)
            reward = torch.from_numpy(reward).to(image.device)
            reward = reward.view(B, self.beam_size)

            # ================================================================
            # Step 9: 计算Self-Critical Baseline
            # 
            # Self-Critical的核心思想：
            # 使用同一个batch内所有候选的平均reward作为baseline
            # 这样可以减小方差，使训练更稳定
            # ================================================================
            reward_baseline = torch.mean(reward, -1, keepdim=True)

            # ================================================================
            # Step 10: 计算SCST损失
            # 
            # L = -log_prob(seq) * (reward - baseline)
            # 
            # 解释：
            # - 如果reward > baseline，表示这个序列比平均好，增加其概率
            # - 如果reward < baseline，表示这个序列比平均差，减少其概率
            # ================================================================
            loss = - (sequences_scores) * (reward - reward_baseline)
            loss = loss.mean()

            return {"loss": loss}


    def maybe_autocast(self, dtype=torch.float16):
        """自动混合精度上下文管理器"""
        import contextlib
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    @property
    def device(self):
        """获取模型所在设备"""
        return list(self.parameters())[0].device


    @classmethod
    def from_config(cls, cfg):
        """
        从配置文件创建模型
        
        关键配置项:
            scst (bool): 是否启用SCST模式
            beam_size (int): beam search的beam数量
        """
        # ... 其他配置解析 ...
        
        scst = cfg.get("scst", False)  # 从配置中读取scst开关
        
        model = cls(
            scst=scst,
            # ... 其他参数 ...
        )
        
        # 加载预训练权重
        model.load_checkpoint_from_config(cfg)
        
        return model


# ============================================================================
# 使用示例
# ============================================================================
"""
# 1. CE预训练（必须先完成这一步）
cfg_ce = load_config("caption_ce_example.yaml")
cfg_ce.model.scst = False
model = Blip2T5SCST.from_config(cfg_ce)
train(model, ce_dataloader)
save_checkpoint(model, "ce_pretrained.pth")

# 2. SCST微调
cfg_scst = load_config("caption_scst_example.yaml") 
cfg_scst.model.scst = True
cfg_scst.model.pretrained = "ce_pretrained.pth"  # 加载CE预训练模型
model = Blip2T5SCST.from_config(cfg_scst)
train(model, scst_dataloader)
"""
