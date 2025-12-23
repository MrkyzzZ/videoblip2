"""
Evaluation Package - 用于SCST训练的评估工具
============================================

本包提供CIDEr评估器和相关工具，用于：
1. SCST训练时计算reward
2. 模型评估时计算各种指标

主要组件：
- Cider: CIDEr评估器主类
- CiderScorer: CIDEr评分核心算法
- PTBTokenizer: 文本标记化工具
- tokenize: 便捷的tokenization函数
"""

from .cider import Cider
from .cider_scorer import CiderScorer
from .ptbtokenizer import PTBTokenizer, tokenize

__all__ = ['Cider', 'CiderScorer', 'PTBTokenizer', 'tokenize']
