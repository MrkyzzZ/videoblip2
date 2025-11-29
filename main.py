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

from videoblip2.training.trainer import Trainer # type: ignore
from videoblip2.training.config import TrainConfig # type: ignore


def _parse_gpu_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        ids = [int(item) for item in raw if str(item).strip() != ""]
        return ids or None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None
        ids = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            ids.append(int(part))
        return ids or None
    raise TypeError(f"无法解析 GPU ID: 类型 {type(raw)}")


def main():
    parser = argparse.ArgumentParser(description="多模态视听问答模型训练脚本")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=0,
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
