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
        # 设置n-gram范围（1到n）
        self._n = n
        # 设置高斯长度惩罚的标准差参数
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        计算CIDEr分数
        
        这是SCST训练中最重要的函数，用于计算reward。
        
        Args:
            gts (dict): 参考captions
                格式: {image_id: [tokenized_ref1, tokenized_ref2, ...]}
            res (dict): 生成的captions（hypotheses）
                格式: {image_id: [tokenized_hypothesis]}
                注意：每个image_id只有一个hypothesis
        
        Returns:
            score (float): 所有样本的平均CIDEr分数
            scores (numpy.ndarray): 每个样本的CIDEr分数
                
        在SCST中的使用:
            reward = cider.compute_score(gts, res)[1]  # 取每个样本的分数
            reward = torch.from_numpy(reward).to(device)
        """
        # 确保gts和res有相同的key
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        # 创建CIDEr评分器
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # 健全性检查
            assert(type(hypo) is list)
            assert(len(hypo) == 1)  # 每个样本只有一个hypothesis
            assert(type(ref) is list)
            assert(len(ref) > 0)    # 至少有一个reference

            # 添加到评分器
            cider_scorer += (hypo[0], ref)

        # 计算最终分数
        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"


# ============================================================================
# SCST中使用CIDEr的示例
# ============================================================================
"""
在SCST训练的forward中:

# 1. 生成候选captions
caps_gen = model.generate(...)

# 2. 准备ground truth
caps_gt = samples["text_output"]

# 3. Tokenize（使用PTBTokenizer）
caps_gen_tokenized, caps_gt_tokenized = tokenize(caps_gt, caps_gen)

# 4. 计算CIDEr reward
cider = Cider()
_, scores = cider.compute_score(caps_gt_tokenized, caps_gen_tokenized)

# 5. 转换为tensor
reward = torch.from_numpy(scores.astype(np.float32)).to(device)

# 6. 计算baseline和损失
baseline = reward.mean()
loss = -log_probs * (reward - baseline)
"""
