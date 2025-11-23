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
    VIDEO_FEATURES_DIR   = "/root/autodl-tmp/videoblip2/data/MSRVTT_frame_ViT"
    AUDIO_FEATURES_DIR   = "/root/autodl-tmp/videoblip2/data/MSRVTT_clap"
    T5_MODEL_PATH        = "/root/autodl-tmp/videoblip2/pretrained/flan-t5-base"
    BERT_TOKENIZER_PATH  = "/root/autodl-tmp/videoblip2/pretrained/bert-base-uncased"

    # --- BLIP-2 模型路径 ---
    BLIP2_MODEL_PATH     = "/root/autodl-tmp/videoblip2/pretrained/blip2/blip2_pretrained_flant5xl.pth"

    # ==============================================================================
    # --- 保存路径 ---
    # ==============================================================================
    SAVE_DIR = "/root/autodl-tmp/videoblip2/saved_models" 
    LOG_DIR  = "/root/autodl-tmp/videoblip2/logs"          
    SAMPLE_OUTPUT_DIR = "/root/autodl-tmp/videoblip2/logs/sample_generations"
    SAMPLE_OUTPUT_COUNT = 5

    # ==============================================================================
    # --- 训练超参数 ---
    # ==============================================================================
    EPOCHS        = 75
    BATCH_SIZE    = 32
    LEARNING_RATE = 5e-5
    WARMUP_RATIO  = 0.05

    # ==============================================================================
    # --- 模型架构超参数 ---
    # ==============================================================================
    NUM_QUERY_TOKEN    = 32
    QFORMER_NUM_LAYERS   = 6
    DEFAULT_QUESTION_PROMPT ="Describe the video content in detail."
    
    # LoRA 相关参数
    LORA_R       = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.05
