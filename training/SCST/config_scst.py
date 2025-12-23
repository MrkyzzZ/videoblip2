from ..config import TrainConfig


class SCSTTrainConfig(TrainConfig):
    """SCST 微调配置，继承基线 CE 配置并下调学习率/Batch Size。"""

    # 保留数据/路径配置，复用 TrainConfig

    # 指定 CE 预训练权重所在目录（包含 best_lora_adapters/ 与 best_other_params.pth）。
    # 如不填则默认使用 TrainConfig.SAVE_DIR。
    CE_CKPT_DIR = "/root/autodl-tmp/videoblip2/logs/20251219_202043/best_model"
    TRAIN_ANNOTATIONS_PATH = "/root/autodl-tmp/videoblip2/data/MSRVTT_annots/train_top5_detail.json"
    
    # 训练超参数
    EPOCHS = 30
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-6
    WARMUP_RATIO = 0.2
    USE_COSINE_DECAY = True

    # 调试开关：限制每个 epoch 的训练步数、打印生成样例
    DEBUG_MAX_STEPS_PER_EPOCH = None  # None 表示不限制，跑完整个 epoch
    DEBUG_LOG_SAMPLES = 0          # 0 表示不打印调试样例

    # SCST 生成/奖励相关
    SCST_BEAM_SIZE = 3
    SCST_MAX_NEW_TOKENS = 40
    SCST_LENGTH_PENALTY = 0.8
    SCST_REPETITION_PENALTY = 1.05
    SCST_DO_SAMPLE = False
    SCST_TEMPERATURE = 0.7

    # 评估/日志可以沿用基类设置
