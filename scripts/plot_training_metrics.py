#!/usr/bin/env python3
"""Parse training_log_gpu*.txt and plot Loss / BLEU / METEOR / ROUGE-L / CIDEr vs epoch.

Usage:
    python scripts/plot_training_metrics.py \
        --log /root/autodl-tmp/videoblip2/exp/logs/20251220_161311/training_log_gpu0.txt \
        --out /root/autodl-tmp/videoblip2/exp/logs/20251220_161311/metrics.png

Requires matplotlib: pip install matplotlib
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Regex to capture metrics from log lines (no Exact Match field in latest logs)
LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Loss:\s*(?P<loss>[0-9.]+).*?"
    r"BLEU-1:\s*(?P<bleu1>[0-9.]+).*?"
    r"BLEU-4:\s*(?P<bleu4>[0-9.]+).*?"
    r"METEOR:\s*(?P<meteor>[0-9.]+).*?"
    r"ROUGE-L:\s*(?P<rouge>[0-9.]+).*?"
    r"CIDEr:\s*(?P<cider>[0-9.]+)"
)


def parse_log(log_path: Path):
    data = {
        "epoch": [], "loss": [], "bleu1": [], "bleu4": [],
        "meteor": [], "rouge": [], "cider": [],
    }
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            gd = m.groupdict()
            data["epoch"].append(int(gd["epoch"]))
            data["loss"].append(float(gd["loss"]))
            data["bleu1"].append(float(gd["bleu1"]))
            data["bleu4"].append(float(gd["bleu4"]))
            data["meteor"].append(float(gd["meteor"]))
            data["rouge"].append(float(gd["rouge"]))
            data["cider"].append(float(gd["cider"]))
    return data


def plot_curves(data, out_path: Path):
    if not data["epoch"]:
        raise ValueError("No metrics found in log; check the log format or path.")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes.flatten()

    # Loss
    ax[0].plot(data["epoch"], data["loss"], label="Loss", color="tab:blue")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss")
    ax[0].grid(True, alpha=0.3)

    # BLEU-1 / BLEU-4
    ax[1].plot(data["epoch"], data["bleu1"], label="BLEU-1", color="tab:orange")
    ax[1].plot(data["epoch"], data["bleu4"], label="BLEU-4", color="tab:red")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Score")
    ax[1].set_title("BLEU")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # METEOR
    ax[2].plot(data["epoch"], data["meteor"], label="METEOR", color="tab:purple")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Score")
    ax[2].set_title("METEOR")
    ax[2].grid(True, alpha=0.3)

    # ROUGE-L
    ax[3].plot(data["epoch"], data["rouge"], label="ROUGE-L", color="tab:brown")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Score")
    ax[3].set_title("ROUGE-L")
    ax[3].grid(True, alpha=0.3)

    # CIDEr
    ax[4].plot(data["epoch"], data["cider"], label="CIDEr", color="tab:cyan")
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("Score")
    ax[4].set_title("CIDEr")
    ax[4].grid(True, alpha=0.3)

    # Remove unused subplot if present
    if len(ax) > 5:
        fig.delaxes(ax[5])

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from log.")
    parser.add_argument("--log", required=True, type=Path, help="Path to training_log_gpu*.txt")
    parser.add_argument("--out", required=True, type=Path, help="Output image path (e.g., metrics.png)")
    args = parser.parse_args()

    data = parse_log(args.log)
    plot_curves(data, args.out)


if __name__ == "__main__":
    main()
