#!/usr/bin/env python3
"""Extract ViT-L/14@336 video frame features using a local OpenCLIP checkpoint."""

import argparse
import contextlib
import logging
import os
import sys
import warnings
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required. Install via `pip install opencv-python`." ) from exc

try:
    import open_clip  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("open_clip_torch is required. Install via `pip install open_clip_torch`." ) from exc

try:
    from PIL import Image  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required. Install via `pip install pillow`." ) from exc


DIR_VIDEO = "/root/autodl-tmp/videoblip2/data/MUSIC-AVQA-videos-Synthetic"
DIR_OUT = "/root/autodl-tmp/videoblip2/data/frame_ViT-L14@336px"
DEFAULT_MODEL = "ViT-L-14-336"
DEFAULT_CKPT = "/root/autodl-tmp/videoblip2/pretrained/ViT-L-14-336px/ViT-L-14-336px.pt"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def sample_frames(video_path: str, frame_stride: int, max_frames: Optional[int]) -> List[np.ndarray]:
    """Decode video, sample frames by stride, optionally cap total."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    idx = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame is None:
            continue
        if frame_stride <= 1 or idx % frame_stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        return []

    if max_frames is not None and len(frames) > max_frames:
        sel_idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
        frames = [frames[i] for i in sel_idx]
    return frames


@contextlib.contextmanager
def suppress_open_clip_warning():
    """Temporarily silence OpenCLIP's "No pretrained weights" warning."""
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        root_logger.setLevel(prev_level)


def build_open_clip_encoder(model_name: str, ckpt_path: str, device: torch.device, dtype: torch.dtype):
    """Load OpenCLIP vision encoder from a local .pt checkpoint.

    Supports both state-dict checkpoints and TorchScript archives.
    Returns (model, preprocess, feature_dim).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    preprocess = open_clip.image_transform(336, is_train=False, mean=None, std=None)

    def _finalize_model(state_dict):
        with suppress_open_clip_warning():
            model = open_clip.create_model(model_name, pretrained=None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
        print(f"[INFO] Loaded checkpoint weights from {ckpt_path}")

        # Update preprocess stats based on model config
        prep = open_clip.image_transform(
            model.visual.image_size if hasattr(model.visual, "image_size") else 336,
            is_train=False,
            mean=getattr(model.visual, "image_mean", None),
            std=getattr(model.visual, "image_std", None),
        )

        model = model.to(device)
        model.eval()
        if dtype == torch.float16:
            model = model.half()
        return model, prep, getattr(model.visual, "output_dim", 768)

    state_dict = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="'torch.load' received a zip file that looks like a TorchScript archive",
                category=UserWarning,
            )
            state_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state_obj, dict):
            if "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
                state_obj = state_obj["state_dict"]
            state_dict = state_obj
    except RuntimeError as err:
        state_dict = None
        state_load_err = err
    else:
        state_load_err = None

    if isinstance(state_dict, dict):
        return _finalize_model(state_dict)

    # Fall back to TorchScript model if state dict loading failed
    try:
        jit_model = torch.jit.load(ckpt_path, map_location="cpu")
        state_dict = jit_model.state_dict()
    except Exception as err:  # pragma: no cover
        if state_load_err is not None:
            raise RuntimeError(
                f"Failed to load checkpoint '{ckpt_path}'. State-dict error: {state_load_err}. JIT error: {err}"
            ) from err
        raise

    return _finalize_model(state_dict)


def frames_to_features(
    frames: List[np.ndarray],
    preprocess,
    model,
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
    feature_dim: int,
) -> np.ndarray:
    if not frames:
        return np.zeros((0, feature_dim), dtype=np.float32)

    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_np = frames[i : i + batch_size]
            images = [Image.fromarray(frame) for frame in batch_np]
            pixel_values = torch.stack([preprocess(img) for img in images]).to(device)
            if dtype == torch.float16:
                pixel_values = pixel_values.half()
            else:
                pixel_values = pixel_values.float()

            image_embeds = model.encode_image(pixel_values)

            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            feats.append(image_embeds.cpu())

    feats = torch.cat(feats, dim=0)
    return feats.float().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViT-L/14@336 frame features with a local checkpoint")
    parser.add_argument("--video_dir", type=str, default=DIR_VIDEO, help="Directory containing source videos")
    parser.add_argument("--out_dir", type=str, default=DIR_OUT, help="Directory to store .npy feature files")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to ViT-L-14-336 checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="OpenCLIP model name matching the checkpoint")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the vision encoder")
    parser.add_argument("--frame_stride", type=int, default=8, help="Sample every Nth frame (1 = all)")
    parser.add_argument("--max_frames", type=int, default=64, help="Maximum frames per video (<=0 for no cap)")
    parser.add_argument("--fp16", action="store_true", help="Run encoder in float16 when CUDA is available")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"[ERROR] video_dir not found: {args.video_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    max_frames = None if args.max_frames is not None and args.max_frames <= 0 else args.max_frames

    model, preprocess, feature_dim = build_open_clip_encoder(args.model_name, args.ckpt, device, dtype)

    video_files = [f for f in os.listdir(args.video_dir) if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
    video_files.sort()

    print("=" * 80)
    print("ViT-L/14@336 Frame Feature Extraction (local checkpoint)")
    print("-" * 80)
    print(f"Model name : {args.model_name}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Input dir  : {os.path.abspath(args.video_dir)}")
    print(f"Output dir : {os.path.abspath(args.out_dir)}")
    print(f"Frame stride: {args.frame_stride}, Max frames: {args.max_frames}")
    print(f"Batch size : {args.batch_size}, Device: {device}, dtype: {dtype}")
    print("=" * 80)

    pbar = tqdm(video_files, desc="Extracting frame features")
    for fname in pbar:
        vid = os.path.splitext(fname)[0]
        in_path = os.path.join(args.video_dir, fname)
        out_path = os.path.join(args.out_dir, f"{vid}.npy")

        if os.path.exists(out_path) and not args.overwrite:
            pbar.set_postfix_str("skip")
            continue

        try:
            frames = sample_frames(in_path, args.frame_stride, max_frames)
        except Exception as exc:
            print(f"[WARN] Failed to read {in_path}: {exc}")
            continue

        feats = frames_to_features(
            frames,
            preprocess,
            model,
            device,
            args.batch_size,
            dtype,
            feature_dim,
        )
        np.save(out_path, feats)

    print("Done. Saved features to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
