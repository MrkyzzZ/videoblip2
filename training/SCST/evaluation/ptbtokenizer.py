"""
PTB Tokenizer - Penn Treebank 风格标记化（简化版）。
"""

import re
import string

PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"]


class PTBTokenizer:
    def tokenize(self, captions_for_image):
        final_tokenized_captions_for_image = {}
        for k, v in captions_for_image.items():
            if k not in final_tokenized_captions_for_image:
                final_tokenized_captions_for_image[k] = []
            for cap_dict in v:
                caption = cap_dict['caption'] if isinstance(cap_dict, dict) else cap_dict
                tokenized = self._simple_tokenize(caption)
                final_tokenized_captions_for_image[k].append(tokenized)
        return final_tokenized_captions_for_image

    def _simple_tokenize(self, text):
        text = text.lower()
        for char in string.punctuation:
            text = text.replace(char, f' {char} ')
        words = text.split()
        words = [w for w in words if w not in PUNCTUATIONS]
        tokenized = ' '.join(words)
        tokenized = re.sub(r'\s+', ' ', tokenized).strip()
        return tokenized


def tokenize(refs, cands, no_op=False):
    tokenizer = PTBTokenizer()
    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}
    else:
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}
        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)
    return refs, cands
