#!/usr/bin/env python3
"""
CLAP音频特征提取脚本
从视频文件中提取CLAP音频特征，用于Music AVQA数据集预处理
"""

# ===================== 填好的路径（可按需修改） =====================

DIR_MODEL = "/root/autodl-tmp/videoblip2/pretrained/clap-htsat-fused"      # 离线下载好的 laion/clap-htsat-fused 本地目录

DIR_VIDEO = "/root/autodl-tmp/videoblip2/data/MUSIC-AVQA-videos-Real" # 原始视频目录（文件名需与 video_id 对应，如 00001234.mp4）

DIR_OUT   = "/root/autodl-tmp/videoblip2/data/clap"                  # 输出 .npy 目录（每个 video_id.npy 是 [T, D]）

# ================================================================

import os
import argparse
import numpy as np
import subprocess
import shutil
import torch
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

# 支持的视频扩展名
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def check_ffmpeg():
    """检查ffmpeg是否可用"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "未检测到 ffmpeg，请先在系统中安装（apt/yum/brew/conda 均可），并确保命令行可用。"
        )


def load_audio_from_video_ffmpeg(video_path, target_sr=48000):
    """
    用 ffmpeg 将视频解码为单声道、target_sr 采样率的 float32 原始PCM，从 stdout 读取为 numpy。

    返回: np.ndarray, shape [N,] float32, 采样率=target_sr
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", video_path,
        "-ac", "1",                  # 单声道
        "-ar", str(target_sr),       # 重采样
        "-f", "f32le",               # float32 little-endian 原始PCM
        "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 解码失败: {video_path}\n{e.stderr.decode('utf-8', errors='ignore')}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, target_sr


def chunk_waveform(wav, sr, win_sec=1.0, hop_sec=1.0, min_len_sec=None):
    """
    将整段波形切成 [T, win] 片段；末段<0.5s丢弃，>=0.5s右侧零填充到1s
    """
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if min_len_sec is None:
        min_len_sec = win_sec / 2.0
    min_len = int(round(min_len_sec * sr))

    chunks = []
    # 常规步进
    for start in range(0, max(1, len(wav) - win + 1), hop):
        end = start + win
        seg = wav[start:end]
        if len(seg) < win:
            if len(seg) < min_len:
                break
            pad = np.zeros(win - len(seg), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        chunks.append(seg)

    # 特殊：整体少于一个win但>=半个win
    if not chunks and len(wav) >= min_len:
        pad = np.zeros(win - len(wav), dtype=wav.dtype)
        chunks = [np.concatenate([wav, pad], axis=0)]

    return chunks  # List[np.ndarray]，每段长度=win


def process_one_video(video_path, processor, model, device, batch_size=64,
                      target_sr=48000, win_sec=1.0, hop_sec=1.0, fp16=False,
                      out_path=None, verbose=True):
    """处理单个视频文件，提取CLAP音频特征"""
    if verbose:
        print(f"[INPUT ] {os.path.abspath(video_path)}")
        if out_path is not None:
            print(f"[OUTPUT] {os.path.abspath(out_path)}")

    wav, sr = load_audio_from_video_ffmpeg(video_path, target_sr=target_sr)
    chunks = chunk_waveform(wav, sr, win_sec=win_sec, hop_sec=hop_sec)

    # 空音频兜底
    if len(chunks) == 0:
        D = (getattr(model.config, "projection_dim", None)
             or getattr(model.config, "text_embed_dim", None)
             or 512)
        return np.zeros((0, int(D)), dtype=np.float32)

    feats = []
    model.eval()
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]  # List[np.ndarray]，每段长度一致

            # ★ 关键修复：padding=True + truncation=True，并取出 attention_mask / is_longer
            inputs = processor(
                audio=batch,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            input_features = inputs["input_features"].to(device=device, dtype=dtype)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            is_longer = inputs.get("is_longer", None)
            if is_longer is None:
                # 构造一个全 False 的 is_longer（形状 [B]）
                is_longer = torch.zeros(input_features.shape[0], dtype=torch.bool, device=device)
            else:
                is_longer = is_longer.to(device)

            # 传入所有需要的参数
            audio_embeds = model.get_audio_features(
                input_features=input_features,
                attention_mask=attention_mask,
                is_longer=is_longer
            )  # [B, D]

            audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)
            feats.append(audio_embeds.detach().cpu())

    feats = torch.cat(feats, dim=0)  # [T, D]
    return feats.float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract CLAP audio features from videos (OFFLINE, via ffmpeg)")
    parser.add_argument("--model_dir", type=str, default=DIR_MODEL, help="本地 CLAP 模型目录（离线）")
    parser.add_argument("--video_dir", type=str, default=DIR_VIDEO, help="原始视频目录（mp4/mkv/avi/mov/webm/m4v）")
    parser.add_argument("--out_dir",   type=str, default=DIR_OUT,   help="输出 .npy 目录（每个 video_id.npy 是 [T, D]）")
    parser.add_argument("--gpu_id",    type=int, default=0,         help="指定使用的GPU ID (默认: 0)")
    parser.add_argument("--device",    type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--win_sec",   type=float, default=1.0)
    parser.add_argument("--hop_sec",   type=float, default=1.0)
    parser.add_argument("--target_sr", type=int,   default=48000)
    parser.add_argument("--batch_size", type=int,  default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    # 设置指定的GPU
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = f"cuda:{args.gpu_id}"

    check_ffmpeg()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"[ERROR] 本地模型目录不存在：{os.path.abspath(args.model_dir)}")

    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"[ERROR] 输入视频目录不存在： {os.path.abspath(args.video_dir)}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("="*80)
    print("CLAP Audio Feature Extraction FROM VIDEOS (OFFLINE, ffmpeg)")
    print("-"*80)
    print(f"Local model dir : {os.path.abspath(args.model_dir)}")
    print(f"Input VIDEO dir : {os.path.abspath(args.video_dir)}")
    print(f"Output NPY dir  : {os.path.abspath(args.out_dir)}")
    print(f"Device          : {args.device}")
    print(f"Window/Hop (s)  : {args.win_sec} / {args.hop_sec}")
    print("="*80)

    processor = ClapProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model     = ClapModel.from_pretrained(args.model_dir,     local_files_only=True).to(args.device)

    # 收集视频文件
    video_files = [f for f in os.listdir(args.video_dir) if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    video_files.sort()

    print(f"Found {len(video_files)} video files.")

    D_printed = False
    pbar = tqdm(video_files, desc="Extracting CLAP audio features from videos (offline)")

    for fname in pbar:
        vid = os.path.splitext(fname)[0]  # 假设文件名就是 video_id

        in_path  = os.path.join(args.video_dir, fname)
        out_path = os.path.join(args.out_dir, f"{vid}.npy")

        if os.path.exists(out_path):
            if args.verbose:
                print(f"[SKIP  ] {vid} -> {os.path.abspath(out_path)} (exists)")
            continue

        feats = process_one_video(
            in_path, processor, model, device=args.device, batch_size=args.batch_size,
            target_sr=args.target_sr, win_sec=args.win_sec, hop_sec=args.hop_sec,
            fp16=args.fp16, out_path=out_path, verbose=args.verbose
        )

        np.save(out_path, feats)

        if not D_printed and feats.shape[0] > 0:
            print(f"\nExample feature shape for {vid}: {feats.shape}  (T, D)  --> D={feats.shape[-1]}")
            D_printed = True

    print("Done. Saved features to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
