"""
训练配置文件
包含所有训练相关的配置参数
"""


class TrainConfig:
    """训练配置类"""

    # ==============================================================================
    # --- 数据和模型路径 ---
    # ==============================================================================
    TRAIN_ANNOTATIONS_PATH = "/root/autodl-tmp/videoblip2/data/MSRVTT_annots/train.json"
    TEST_ANNOTATIONS_PATH  = "/root/autodl-tmp/videoblip2/data/MSRVTT_annots/test.json"
    VIDEO_FEATURES_DIR   = "/root/autodl-tmp/videoblip2/data/MSRVTT_patchlevel_Vit"
    AUDIO_FEATURES_DIR   = "/root/autodl-tmp/videoblip2/data/MSRVTT_clap"
    T5_MODEL_PATH        = "/root/autodl-tmp/videoblip2/pretrained/flan-t5-large"
    BERT_TOKENIZER_PATH  = "/root/autodl-tmp/videoblip2/pretrained/bert-base-uncased"

    # --- BLIP-2 模型路径 ---
    BLIP2_MODEL_PATH     = "/root/autodl-tmp/videoblip2/pretrained/blip2/blip2_pretrained_flant5xl.pth"

    # ==============================================================================
    # --- 保存路径 ---
    # ==============================================================================
    SAVE_DIR = "/root/autodl-tmp/videoblip2/saved_models" 
    LOG_DIR  = "/root/autodl-tmp/videoblip2/logs"          
    

    # ==============================================================================
    # --- 训练超参数 ---
    # ==============================================================================
    EPOCHS           = 200
    BATCH_SIZE       = 8   # 降低 Batch Size 以适配 Patch 级特征的显存消耗
    LEARNING_RATE    = 5e-6
    WARMUP_RATIO     = 0.15
    USE_COSINE_DECAY = True  # 是否启用余弦退火学习率




    # ==============================================================================
    # --- 模型架构超参数 ---
    # ==============================================================================
    NUM_QUERY_TOKEN    = 32
    QFORMER_NUM_LAYERS   = 6
    DEFAULT_QUESTION_PROMPT ="Describe the video content in detail."
    
    # Patch 级特征配置
    CLIP_PATCH_DIM = 1024  # ViT-L/14 输出维度
    QFORMER_INPUT_DIM = 768 # Q-Former 期望维度

    # LoRA 相关参数
    LORA_R       = 32
    LORA_ALPHA   = 64
    LORA_DROPOUT = 0.05
