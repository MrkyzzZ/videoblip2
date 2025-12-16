import json
import os
import argparse
import sys
import re
from collections import defaultdict
from typing import List, Tuple
from tqdm import tqdm

try:
    from pycocoevalcap.cider.cider_scorer import CiderScorer
except ImportError:
    CiderScorer = None


# Heuristic detail scoring vocabulary (borrowed from scripts/test.py)
VAGUE_WORDS = {
    "something", "someone", "thing", "things", "stuff", "person", "people",
    "man", "woman", "guy", "girl", "video", "clip", "scene", "screen",
    "doing", "showing", "being", "having", "going", "getting"
}

DETAIL_INDICATORS = {
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "grey",
    "two", "three", "four", "five", "several", "many", "few",
    "left", "right", "top", "bottom", "front", "back", "behind", "beside", "between",
    "slowly", "quickly", "carefully", "loudly", "softly"
}

CONTENT_WORD_PATTERNS = [
    r"\b\w+ing\b",
    r"\b(?!red|bed|fed|led|sled|shed|speed|seed|need|breed|feed)\w+ed\b",
    r"\b\w+ly\b"
]


def _detail_score(caption: str) -> float:
    if not caption or not caption.strip():
        return 0.0
    caption_lower = caption.lower().strip()
    words = re.findall(r"\b[a-zA-Z]+\b", caption_lower)
    if not words:
        return 0.0

    word_count = len(words)
    unique_words = set(words)

    if word_count < 5:
        length_score = word_count * 0.5
    elif word_count <= 25:
        length_score = min(word_count, 20)
    else:
        length_score = 20 - (word_count - 25) * 0.3

    diversity_score = (len(unique_words) / word_count) * 10

    specificity_score = 0.0
    for word in unique_words:
        if word in DETAIL_INDICATORS:
            specificity_score += 3.0
    for word in words:
        if word in VAGUE_WORDS:
            specificity_score -= 2.0

    content_word_count = 0
    for pattern in CONTENT_WORD_PATTERNS:
        content_word_count += len(re.findall(pattern, caption_lower))
    content_density_score = min(content_word_count * 1.5, 10)

    number_bonus = len(re.findall(r"\b\d+\b", caption)) * 2

    total_score = (
        length_score * 1.0 +
        diversity_score * 1.5 +
        specificity_score * 2.0 +
        content_density_score * 1.0 +
        number_bonus
    )
    return max(total_score, 0.0)


def _cider_score(candidate: str, references: List[str]) -> float:
    if not references:
        return 0.0
    scorer = CiderScorer()
    scorer += (candidate, references)
    score, _ = scorer.compute_score()
    return float(score)


def _length_score(caption: str) -> float:
    return len(caption.split())


def _compute_score(method: str, caption: str, references: List[str]) -> float:
    if method == "cider" and CiderScorer is not None and references:
        try:
            return _cider_score(caption, references)
        except Exception:
            return 0.0
    if method == "detail":
        return _detail_score(caption)
    return _length_score(caption)

def main():
    parser = argparse.ArgumentParser(description="预处理字幕文件，为每个视频选出细节最丰富或共识度最高的字幕 (可输出 Top-K)")
    parser.add_argument(
        "--input_path", 
        type=str, 
        default="../data/MSRVTT_annots/train_val_videodatainfo.json",
        help="原始标注文件路径 (包含所有字幕)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="../data/MSRVTT_annots/train_detailed.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--selection_method",
        choices=["cider", "length", "detail"],
        default="cider",
        help="选择字幕的策略: cider=选CIDEr分最高, length=选单词数最多, detail=启发式细节评分"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="为每个视频保留的字幕数量 (>=1)"
    )
    parser.add_argument(
        "--store_scores",
        action="store_true",
        help="在输出 JSON 中附加每条字幕的得分，便于调试"
    )
    
    args = parser.parse_args()
    
    # 处理相对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input_path):
        project_root = os.path.dirname(base_dir)
        possible_path = os.path.join(project_root, args.input_path)
        if os.path.exists(possible_path):
            args.input_path = possible_path
        elif os.path.exists(args.input_path):
            pass
        else:
            possible_path_2 = os.path.join(base_dir, args.input_path)
            if os.path.exists(possible_path_2):
                args.input_path = possible_path_2

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)

    print(f"Loading annotations from {args.input_path}...")
    with open(args.input_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'sentences' in data:
        raw_sentences = data['sentences']
    elif isinstance(data, list):
        raw_sentences = data
    else:
        print("Error: Unknown JSON format.")
        sys.exit(1)

    print(f"Found {len(raw_sentences)} total captions. Grouping by video_id...")
    
    video_groups = defaultdict(list)
    for item in raw_sentences:
        vid = item.get('video_id') or item.get('id') or item.get('video')
        caption = item.get('caption') or item.get('answer')
        if vid is None or not caption:
            continue

        vid = str(vid)

        if isinstance(caption, list):
            for idx, cap in enumerate(caption):
                if not cap:
                    continue
                new_item = dict(item)
                new_item['caption'] = str(cap)
                new_item['caption_variant_idx'] = idx
                video_groups[vid].append(new_item)
        else:
            item['caption'] = str(caption)
            video_groups[vid].append(item)

    print(f"Processing {len(video_groups)} unique videos...")
    
    best_annotations = []
    
    if args.selection_method == "cider" and CiderScorer is None:
        print("Warning: pycocoevalcap 未安装，自动退回 length 策略。")

    top_k = max(1, args.top_k)

    for vid, items in tqdm(video_groups.items(), desc="Selecting detailed captions"):
        scored_items: List[Tuple[dict, float]] = []
        for idx, cand in enumerate(items):
            references = [ref['caption'] for j, ref in enumerate(items) if j != idx]
            score = _compute_score(args.selection_method, cand['caption'], references)
            scored_items.append((cand, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        selected = scored_items[:top_k]
        selected_captions = [item['caption'] for item, _ in selected]
        selected_scores = [score for _, score in selected]

        if not selected_captions:
            continue

        caption_value = selected_captions[0] if top_k == 1 else selected_captions

        new_entry = {
            "video_id": vid,
            "caption": caption_value,
            "id": selected[0][0].get('id') if selected[0][0].get('id') is not None else None,
            "video_feature_path": f"{vid}.npy",
            "clap_feature_path": f"{vid}.npy"
        }

        if args.store_scores:
            new_entry["caption_scores"] = selected_scores if top_k > 1 else selected_scores[0]

        best_annotations.append(new_entry)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving {len(best_annotations)} detailed captions to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(best_annotations, f, indent=2)
        
    print("Done! Please update your config.TRAIN_ANNOTATIONS_PATH to point to this new file.")

if __name__ == "__main__":
    main()
