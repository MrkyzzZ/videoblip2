#!/usr/bin/env python3
"""SCST 训练入口。"""

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /.../videoblip2/training
REPO_ROOT = os.path.dirname(PROJECT_ROOT)  # /.../videoblip2
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)  # /... (父目录，用于导入 videoblip2 包)
for path in (REPO_ROOT, WORKSPACE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from videoblip2.training.SCST.trainer_scst import SCSTTrainer  # type: ignore
from videoblip2.training.SCST.config_scst import SCSTTrainConfig  # type: ignore


def _parse_gpu_ids(raw):
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
    parser = argparse.ArgumentParser(description="SCST (CIDEr) 微调入口")
    parser.add_argument("--gpu_ids", type=str, default=0, help="使用的 GPU ID 列表，如 '0,1'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = SCSTTrainConfig()
    trainer = SCSTTrainer(config, gpu_ids=_parse_gpu_ids(args.gpu_ids))
    trainer.train()


if __name__ == "__main__":
    main()
