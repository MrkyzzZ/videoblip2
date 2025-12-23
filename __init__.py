"""
VideoBLIP2: 多模态视听问答模型
基于BLIP-2和T5的视频-音频-文本联合建模
"""

__version__ = "1.0.0"
__author__ = "AutoDL Team"

from .models import MultiModal_T5_Captioner
from .training import Trainer, TrainConfig

__all__ = [
    'MultiModal_T5_Captioner',
    'Trainer',
    'TrainConfig'
]
