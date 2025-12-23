"""
CIDEr Scorer - 核心评分算法实现
================================

本文件实现了CIDEr的核心评分算法，包括：
- N-gram提取与统计
- TF-IDF权重计算
- 余弦相似度计算
- 长度惩罚

在SCST训练中，这个scorer用于计算每个生成caption的reward。

Reference: https://arxiv.org/abs/1411.5726
"""

import copy
from collections import defaultdict
import numpy as np
import math


def precook(s, n=4, out=False):
    """
    将句子转换为n-gram频率字典
    
    这是CIDEr评分的基础：统计句子中各个n-gram的出现次数。
    
    Args:
        s (str): 输入句子（已tokenized，以空格分隔）
        n (int): 最大n-gram长度
        out (bool): 调试选项
    
    Returns:
        counts (dict): {ngram_tuple: count} 格式的字典
    
    Example:
        >>> precook("a cat sits on mat", n=2)
        {('a',): 1, ('cat',): 1, ('sits',): 1, ('on',): 1, ('mat',): 1,
         ('a', 'cat'): 1, ('cat', 'sits'): 1, ('sits', 'on'): 1, ('on', 'mat'): 1}
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """
    处理参考captions，转换为n-gram统计
    
    Args:
        refs (list of str): 一个图像的多个参考caption
        n (int): 最大n-gram长度
    
    Returns:
        list of dict: 每个参考caption的n-gram统计
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """
    处理hypothesis caption，转换为n-gram统计
    
    Args:
        test (str): 生成的caption
        n (int): 最大n-gram长度
    
    Returns:
        dict: n-gram统计
    """
    return precook(test, n, True)


class CiderScorer(object):
    """
    CIDEr评分器核心类
    
    实现了完整的CIDEr评分算法：
    1. 收集所有参考和hypothesis的n-gram
    2. 计算文档频率（document frequency）
    3. 计算TF-IDF权重
    4. 使用余弦相似度计算匹配度
    5. 应用长度惩罚
    
    在SCST中的作用：
    - 为每个生成的caption计算一个CIDEr分数作为reward
    - reward越高，表示生成的caption与参考越接近
    """

    def copy(self):
        """复制评分器"""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        """
        初始化CIDEr评分器
        
        Args:
            n (int): 考虑的最大n-gram长度（默认4，即1-4 gram）
            sigma (float): 长度惩罚的高斯标准差
        """
        self.n = n
        self.sigma = sigma
        self.crefs = []          # 存储参考captions的n-gram统计
        self.ctest = []          # 存储hypothesis captions的n-gram统计
        self.document_frequency = defaultdict(float)  # 文档频率
        self.cook_append(test, refs)
        self.ref_len = None      # 参考语料库的对数长度

    def cook_append(self, test, refs):
        """添加一对(hypothesis, references)到评分器"""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest)
        return len(self.crefs)

    def __iadd__(self, other):
        """
        添加一个(hypothesis, references)对
        
        使用方式:
            scorer += (hypothesis, [ref1, ref2, ...])
        """
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """
        计算文档频率 (Document Frequency)
        
        文档频率用于计算IDF (Inverse Document Frequency)。
        一个n-gram在越多的参考caption中出现，其IDF越低，
        说明它不够"独特"，对描述不够重要。
        
        例如："a", "the" 等常见词的DF很高，IDF很低
        而 "surfboard", "skateboard" 等具体词的DF低，IDF高
        """
        for refs in self.crefs:
            # refs是一个图像的多个参考caption的n-gram列表
            # 使用set避免重复计数
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        """
        计算CIDEr分数
        
        核心算法流程：
        1. 对每个n-gram计算TF-IDF权重
        2. 将hypothesis和reference转换为TF-IDF向量
        3. 计算向量的余弦相似度
        4. 应用长度惩罚
        5. 对所有n-gram层级取平均
        6. 乘以10作为最终分数
        """
        
        def counts2vec(cnts):
            """
            将n-gram计数转换为TF-IDF向量
            
            TF-IDF = TF * IDF
            - TF (Term Frequency): n-gram在当前文档中的出现次数
            - IDF (Inverse Document Frequency): log(总文档数 / 包含该n-gram的文档数)
            
            Returns:
                vec: TF-IDF向量（按n-gram长度分组）
                norm: 向量的L2范数
                length: 句子长度（用于长度惩罚）
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            
            for (ngram, term_freq) in cnts.items():
                # 获取文档频率，如果不存在则设为1.0
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # n-gram的长度索引（0表示1-gram，1表示2-gram，以此类推）
                n = len(ngram) - 1
                # 计算TF-IDF: TF * (log(N) - log(DF)) = TF * log(N/DF)
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # 累加范数
                norm[n] += pow(vec[n][ngram], 2)
                
                if n == 1:  # 使用2-gram的长度作为句子长度
                    length += term_freq
            
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            计算hypothesis和reference之间的相似度
            
            使用余弦相似度 + 长度惩罚
            
            余弦相似度:
                cos(hyp, ref) = (hyp · ref) / (||hyp|| * ||ref||)
            
            长度惩罚:
                penalty = exp(-(length_hyp - length_ref)^2 / (2 * sigma^2))
            
            Returns:
                val: 每个n-gram层级的相似度分数
            """
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            
            for n in range(self.n):
                # 计算点积（只计算共同的n-gram）
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                
                # 除以范数得到余弦相似度
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                
                assert(not math.isnan(val[n]))
                
                # 应用高斯长度惩罚
                # 如果hypothesis长度与reference差异越大，惩罚越重
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            
            return val

        # 计算参考语料库的对数长度（用于IDF计算）
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # 计算hypothesis的TF-IDF向量
            vec, norm, length = counts2vec(test)
            
            # 对每个reference计算相似度并取平均
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            
            # 对所有n-gram层级取平均
            score_avg = np.mean(score)
            # 除以参考数量
            score_avg /= len(refs)
            # 乘以10作为最终分数（使分数范围更易读）
            score_avg *= 10.0
            scores.append(score_avg)
        
        return scores

    def compute_score(self, option=None, verbose=0):
        """
        计算最终CIDEr分数
        
        Returns:
            mean_score (float): 所有样本的平均分数
            scores (numpy.ndarray): 每个样本的分数
        
        在SCST中:
            _, scores = scorer.compute_score()
            reward = torch.from_numpy(scores).to(device)
        """
        # 首先计算文档频率
        self.compute_doc_freq()
        
        # 验证文档频率的合理性
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        
        # 计算CIDEr分数
        score = self.compute_cider()
        
        return np.mean(np.array(score)), np.array(score)


# ============================================================================
# CIDEr评分示例
# ============================================================================
"""
# 假设有以下数据
references = {
    0: ["a cat sitting on a couch", "a cat on the sofa"],
    1: ["a dog running in the park", "a dog playing outside"]
}
hypotheses = {
    0: ["a cat is on the couch"],
    1: ["a dog is running"]
}

# 计算CIDEr分数
cider = Cider()
mean_score, scores = cider.compute_score(references, hypotheses)

print(f"平均CIDEr: {mean_score}")
print(f"每个样本的CIDEr: {scores}")
# 输出示例:
# 平均CIDEr: 0.85
# 每个样本的CIDEr: [0.92, 0.78]
"""
