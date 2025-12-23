"""
CIDEr (Consensus-based Image Description Evaluation) 评估器
============================================================

本文件从 pycocoevalcap 提取，用于计算CIDEr分数。
在SCST训练中，CIDEr分数作为reward信号。

CIDEr的核心思想：
- 使用TF-IDF加权的n-gram匹配
- 考虑多个参考caption的共识
- 对常见词给予较低权重，对描述性词给予较高权重

Reference: 
- Paper: CIDEr: Consensus-based Image Description Evaluation (https://arxiv.org/abs/1411.5726)
- Original Code: https://github.com/jmhessel/pycocoevalcap
"""

from .cider_scorer import CiderScorer


class Cider:
    """
    CIDEr评估器主类
    
    使用方法:
        cider = Cider()
        score, scores = cider.compute_score(gts, res)
        
    其中:
        gts: dict，格式为 {image_id: [tokenized_reference_captions]}
        res: dict，格式为 {image_id: [tokenized_hypothesis_caption]}
        score: float，所有样本的平均CIDEr分数
        scores: numpy array，每个样本的CIDEr分数
    
    在SCST中，scores用作每个生成caption的reward。
    """
    
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        """
        初始化CIDEr评估器
        
        Args:
            n (int): 考虑的最大n-gram长度，默认4（即1-gram到4-gram）
            sigma (float): 长度惩罚的高斯标准差，默认6.0
        """
        self._n = n
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        计算CIDEr分数
        
        Args:
            gts (dict): 参考captions，{image_id: [tokenized_ref...]}
            res (dict): 生成的captions，{image_id: [tokenized_hyp]}
        """
        assert gts.keys() == res.keys()
        img_ids = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for img_id in img_ids:
            hypo = res[img_id]
            ref = gts[img_id]
            assert isinstance(hypo, list) and len(hypo) == 1
            assert isinstance(ref, list) and len(ref) > 0
            cider_scorer += (hypo[0], ref)

        score, scores = cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"
