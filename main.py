#!/usr/bin/env python3
"""
多模态视听问答模型训练脚本
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from videoblip2.training.trainer import Trainer
from videoblip2.training.config import TrainConfig


def _parse_gpu_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids or None


def main():
    parser = argparse.ArgumentParser(description="多模态视听问答模型训练脚本")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="使用的 GPU ID 列表，如 '0,1,2,3'；留空则自动检测",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化训练器并开始训练
    config = TrainConfig()
    trainer = Trainer(config, gpu_ids=_parse_gpu_ids(args.gpu_ids))
    trainer.train()


if __name__ == '__main__':
    main()
