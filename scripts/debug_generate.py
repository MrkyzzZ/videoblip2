import os
import json
import torch
from tqdm import tqdm

from training.config import TrainConfig
from training.trainer import Trainer


def main(num_samples: int = 5, gpu_ids=None):
    config = TrainConfig()
    trainer = Trainer(config, gpu_ids=gpu_ids)
    model = trainer.model
    device = trainer.device

    model.eval()

    print(f"Total test videos: {len(trainer.test_anns)}")

    # 随机抽取若干视频
    import random
    indices = random.sample(range(len(trainer.test_anns)), min(num_samples, len(trainer.test_anns)))

    for idx in indices:
        ann = trainer.test_anns[idx]
        video_id = ann.get("video_id") or ann.get("id") or ann.get("video")
        print("\n==============================")
        print(f"Video ID: {video_id}")

        # 参考字幕（可能是 list，也可能是单条）
        refs = ann.get("caption") or ann.get("answer") or ann.get("anser")
        if isinstance(refs, str):
            refs = [refs]
        refs = [r for r in refs if r]
        print("[References] (up to 5 shown):")
        for r in refs[:5]:
            print("  -", r)

        # 构造单样本 batch，复用 Trainer 的 _prepare_batch 逻辑最安全
        batch = trainer._prepare_batch([ann], training=False)
        if batch is None:
            print("[Warning] Failed to prepare features for this video, skip.")
            continue
        video_feats, audio_feats, questions, captions, _ = batch

        with torch.no_grad():
            outputs = model(video_feats, audio_feats, questions, labels=None, generate=True)

        preds = outputs.get("generated_text") or []
        if not preds:
            print("[Prediction] <empty>")
        else:
            print("[Prediction]:", preds[0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    if args.gpu_ids is None or args.gpu_ids == "":
        gpu_ids = None
    else:
        gpu_ids = [int(x) for x in str(args.gpu_ids).split(",") if x.strip() != ""]

    main(num_samples=args.num_samples, gpu_ids=gpu_ids)
