# Training utilities for watermark detector

from .loss import (
    WatermarkLoss,
    bce_loss,
    focal_loss,
    bce_focal_loss,
)

from .trainer import DetectorTrainer

from .train import (
    train_unet_detector,
    train_bayesian_detector,
)

__all__ = [
    # Loss functions
    "WatermarkLoss",
    "bce_loss",
    "focal_loss",
    "bce_focal_loss",
    # Trainer
    "DetectorTrainer",
    # Training functions
    "train_unet_detector",
    "train_bayesian_detector",
]
