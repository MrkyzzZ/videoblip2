#!/usr/bin/env python3
"""Parse training_log_gpu*.txt and plot Loss / ExactMatch / BLEU / METEOR / ROUGE-L / CIDEr vs epoch.

Usage:
    python scripts/plot_training_metrics.py \
        --log /root/autodl-tmp/videoblip2/logs/20251216_140448/training_log_gpu0.txt \
        --out /root/autodl-tmp/videoblip2/logs/20251216_140448/metrics.png

Requires matplotlib: pip install matplotlib
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Regex to capture metrics from log lines
LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Loss:\s*(?P<loss>[0-9.]+).*?"
    r"Exact Match:\s*(?P<em>[0-9.]+)%.*?"
    r"BLEU-1:\s*(?P<bleu1>[0-9.]+).*?"
    r"BLEU-4:\s*(?P<bleu4>[0-9.]+).*?"
    r"METEOR:\s*(?P<meteor>[0-9.]+).*?"
    r"ROUGE-L:\s*(?P<rouge>[0-9.]+).*?"
    r"CIDEr:\s*(?P<cider>[0-9.]+)"
)


def parse_log(log_path: Path):
    data = {
        "epoch": [], "loss": [], "em": [], "bleu1": [], "bleu4": [],
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
            data["em"].append(float(gd["em"]))
            data["bleu1"].append(float(gd["bleu1"]))
            data["bleu4"].append(float(gd["bleu4"]))
            data["meteor"].append(float(gd["meteor"]))
            data["rouge"].append(float(gd["rouge"]))
            data["cider"].append(float(gd["cider"]))
    return data


def plot_curves(data, out_path: Path):
    if not data["epoch"]:
        raise ValueError("No metrics found in log; check the log format or path.")

    plt.figure(figsize=(12, 10))

    # Loss
    plt.subplot(3, 2, 1)
    plt.plot(data["epoch"], data["loss"], label="Loss", color="tab:blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)

    # Exact Match
    plt.subplot(3, 2, 2)
    plt.plot(data["epoch"], data["em"], label="Exact Match (%)", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Exact Match (%)")
    plt.title("Exact Match")
    plt.grid(True, alpha=0.3)

    # BLEU-1 / BLEU-4
    plt.subplot(3, 2, 3)
    plt.plot(data["epoch"], data["bleu1"], label="BLEU-1", color="tab:orange")
    plt.plot(data["epoch"], data["bleu4"], label="BLEU-4", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("BLEU")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # METEOR
    plt.subplot(3, 2, 4)
    plt.plot(data["epoch"], data["meteor"], label="METEOR", color="tab:purple")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("METEOR")
    plt.grid(True, alpha=0.3)

    # ROUGE-L
    plt.subplot(3, 2, 5)
    plt.plot(data["epoch"], data["rouge"], label="ROUGE-L", color="tab:brown")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("ROUGE-L")
    plt.grid(True, alpha=0.3)

    # CIDEr
    plt.subplot(3, 2, 6)
    plt.plot(data["epoch"], data["cider"], label="CIDEr", color="tab:cyan")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("CIDEr")
    plt.grid(True, alpha=0.3)

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
