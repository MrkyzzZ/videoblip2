"""
Video-only training configuration.
Extends the base TrainConfig but redirects logs/checkpoints to exp/ and keeps the same data/model paths.
"""

from training.config import TrainConfig


class VideoOnlyTrainConfig(TrainConfig):
    """Config for the video-only experiment (no audio features)."""

    # Redirect outputs under exp/
    LOG_DIR = "/root/autodl-tmp/videoblip2/exp/logs"
    SAVE_DIR = "/root/autodl-tmp/videoblip2/exp/saved_models"

    # Explicitly note audio is unused
    AUDIO_FEATURES_DIR = None
