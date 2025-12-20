#!/usr/bin/env python3
"""Run evaluation on a saved checkpoint (no training).

Usage example:
    python scripts/eval_checkpoint.py --checkpoint_dir /root/autodl-tmp/videoblip2/saved_models --gpu_ids 0
"""

import argparse
import logging
import os
import sys

# Ensure project and its parent are on sys.path so that `videoblip2` can be imported when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
for p in (PROJECT_ROOT, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from videoblip2.training.config import TrainConfig  # type: ignore
from videoblip2.training.trainer import Trainer     # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint without training")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing best_lora_adapters/ and best_other_params.pth; defaults to TrainConfig.SAVE_DIR")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="GPU ids, e.g., '0,1' or leave empty for auto")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Override eval batch size (defaults to TrainConfig.BATCH_SIZE)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args()


def parse_gpu_ids(raw):
    if raw is None or raw == "":
        return None
    ids = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids or None


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    config = TrainConfig()
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    trainer = Trainer(config, gpu_ids=parse_gpu_ids(args.gpu_ids), enable_logging=False)

    if not trainer.load_checkpoint(args.checkpoint_dir):
        logging.error("Failed to load checkpoint. Abort eval.")
        sys.exit(1)

    logging.info("Running evaluation on checkpoint ...")
    # epoch index 0 for logging; disable save_best to avoid writing during eval
    trainer.evaluate(epoch=0, save_best=False)


if __name__ == "__main__":
    main()
