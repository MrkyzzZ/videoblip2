"""
PTB Tokenizer - Penn Treebank风格的文本标记化
==============================================

本文件提供PTB风格的tokenization，用于CIDEr评分前的文本预处理。

PTB Tokenizer的作用：
- 将文本转换为小写
- 分离标点符号
- 移除不需要的标点
- 标准化文本格式

在SCST中，tokenization确保生成的caption和参考caption
使用一致的格式进行CIDEr评分。

注意：原始实现使用Stanford CoreNLP的Java版本。
这里提供一个简化的Python实现，足以用于SCST训练。
"""

import re
import string


# 需要移除的标点符号列表
PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", 
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    """
    Penn Treebank风格的Tokenizer（简化版）
    
    原始实现使用Stanford CoreNLP Java工具。
    这是一个纯Python的简化实现，适用于SCST训练。
    """
    
    def tokenize(self, captions_for_image):
        """
        对captions进行tokenization
        
        Args:
            captions_for_image (dict): 格式为 {image_id: [{'caption': text}, ...]}
        
        Returns:
            final_tokenized_captions_for_image (dict): 
                格式为 {image_id: [tokenized_text, ...]}
        
        Example:
            >>> tokenizer = PTBTokenizer()
            >>> input = {0: [{'caption': 'A cat on the mat.'}]}
            >>> output = tokenizer.tokenize(input)
            >>> print(output)
            {0: ['a cat on the mat']}
        """
        final_tokenized_captions_for_image = {}
        
        for k, v in captions_for_image.items():
            if k not in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            
            for cap_dict in v:
                caption = cap_dict['caption'] if isinstance(cap_dict, dict) else cap_dict
                
                # 简化的tokenization处理
                tokenized = self._simple_tokenize(caption)
                final_tokenized_captions_for_image[k].append(tokenized)
        
        return final_tokenized_captions_for_image
    
    def _simple_tokenize(self, text):
        """
        简化的tokenization
        
        步骤：
        1. 转小写
        2. 在标点符号周围添加空格
        3. 移除不需要的标点
        4. 规范化空格
        """
        # 转小写
        text = text.lower()
        
        # 在标点符号周围添加空格
        for char in string.punctuation:
            text = text.replace(char, f' {char} ')
        
        # 分词
        words = text.split()
        
        # 移除不需要的标点
        words = [w for w in words if w not in PUNCTUATIONS]
        
        # 重新组合
        tokenized = ' '.join(words)
        
        # 规范化空格
        tokenized = re.sub(r'\s+', ' ', tokenized).strip()
        
        return tokenized


def tokenize(refs, cands, no_op=False):
    """
    便捷函数：对参考和候选caption进行tokenization
    
    这是SCST训练中计算CIDEr reward的标准预处理步骤。
    
    Args:
        refs: list of list of str，每个样本的多个参考caption
              格式：[[ref1_sample1, ref2_sample1], [ref1_sample2, ref2_sample2], ...]
        cands: list of str，每个样本的候选caption
              格式：[cand_sample1, cand_sample2, ...]
        no_op: bool，如果True则跳过tokenization（用于调试）
    
    Returns:
        refs: dict，格式为 {idx: [tokenized_ref1, tokenized_ref2, ...]}
        cands: dict，格式为 {idx: [tokenized_cand]}
    
    Example:
        >>> refs = [["a cat on mat", "cat sitting on mat"], ["dog in park"]]
        >>> cands = ["a cat on the mat", "a dog running"]
        >>> refs_tok, cands_tok = tokenize(refs, cands)
        >>> print(cands_tok)
        {0: ['a cat on the mat'], 1: ['a dog running']}
    """
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}
    else:
        # 将refs和cands转换为PTBTokenizer期望的格式
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}

        # Tokenize
        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 示例：在SCST训练中使用tokenize
    
    # 假设模型生成了以下captions
    generated_captions = [
        "A cat is sitting on the couch.",
        "A dog running in the park."
    ]
    
    # Ground truth captions
    gt_captions = [
        ["a cat on the sofa", "a cat sitting on a couch"],
        ["a dog plays in the park", "dog running outside"]
    ]
    
    # Tokenize
    refs_tokenized, cands_tokenized = tokenize(gt_captions, generated_captions)
    
    print("Tokenized references:", refs_tokenized)
    print("Tokenized candidates:", cands_tokenized)
    
    # 现在可以用于CIDEr计算
    # from cider import Cider
    # cider = Cider()
    # score, scores = cider.compute_score(refs_tokenized, cands_tokenized)
