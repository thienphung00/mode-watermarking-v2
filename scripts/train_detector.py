#!/usr/bin/env python3
"""
Train watermark detector models.

Minimal training pipeline for UNetDetector only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.core.config import AppConfig
from src.data.loader import create_dataloaders
from src.data.transforms import get_detector_transforms
from src.engine.pipeline import create_pipeline
from src.models.detectors import UNetDetector
from scripts.utils import setup_logging, get_device


def create_model(
    config: AppConfig,
    device: str,
    logger,
) -> nn.Module:
    """
    Create UNetDetector model.
    
    Args:
        config: Application configuration
        device: Device to create model on
        logger: Logger instance
        
    Returns:
        UNetDetector model
        
    Raises:
        ValueError: If detector_type is not "unet"
    """
    detector_type = config.training.detector_type
    
    if detector_type != "unet":
        raise ValueError(
            f"Only UNetDetector is supported. Got detector_type='{detector_type}'. "
            "BayesianDetector is out of scope."
        )
    
    model_kwargs = config.training.model_kwargs or {}
    
    model = UNetDetector(
        input_channels=model_kwargs.get("input_channels", 4),
        base_channels=model_kwargs.get("base_channels", 64),
        num_classes=model_kwargs.get("num_classes", 1),
        depth=model_kwargs.get("depth", 4),
        use_batch_norm=model_kwargs.get("use_batch_norm", True),
        dropout=model_kwargs.get("dropout", 0.0),
    )
    
    logger.info("Created UNetDetector")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Detector model
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        # Move batch to device
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)
        
        # Forward pass
        # Call model with validate=False to avoid warnings during training
        logits = model(images, validate=False).squeeze(-1)
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: str,
) -> float:
    """
    Validate on validation set.
    
    Args:
        model: Detector model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)
            
            # Forward pass
            logits = model(images, validate=False).squeeze(-1)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Detector model
        optimizer: Optimizer
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: str,
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Detector model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train watermark detector (UNetDetector only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    
    # Data
    parser.add_argument(
        "--train-manifest",
        type=str,
        help="Override training manifest path",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        help="Override validation manifest path",
    )
    
    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    logger.info("=" * 80)
    logger.info("Training Watermark Detector (UNetDetector only)")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = AppConfig.from_yaml(args.config)
    
    if config.training is None:
        raise ValueError("Training configuration not found in config file")
    
    # Apply command-line overrides
    if args.train_manifest:
        config.training.train_manifest = args.train_manifest
    if args.val_manifest:
        config.training.val_manifest = args.val_manifest
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    
    logger.info("Configuration:")
    logger.info(f"  Detector type: {config.training.detector_type}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")
    
    # Get device (FP32 only, no FP16)
    device = get_device(args.device, use_fp16=False)
    logger.info(f"Device: {device}")
    
    # Create model (UNetDetector only)
    logger.info("Creating model...")
    model = create_model(config, device, logger)
    
    # Create loss function (BCEWithLogitsLoss only)
    logger.info("Creating loss function...")
    loss_fn = nn.BCEWithLogitsLoss()
    logger.info("Loss: BCEWithLogitsLoss")
    
    # Create optimizer (AdamW only)
    logger.info("Creating optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    logger.info("Optimizer: AdamW")
    
    # Create transforms (VAE latents only)
    logger.info("Setting up data pipeline...")
    pipeline = create_pipeline(config.diffusion, device=device)
    vae = pipeline.vae
    
    transform = get_detector_transforms(
        detector_type=config.training.detector_type,
        vae=vae,
        device=device,
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_manifest=config.training.train_manifest,
        val_manifest=config.training.val_manifest,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        transform=transform,
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # Setup checkpointing
    checkpoint_dir = Path(config.training.checkpoint_dir)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(model, optimizer, Path(args.resume), device)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, config.training.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        # Log
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint (once per epoch)
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
    
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    
    # Print summary
    logger.info("Training Summary:")
    logger.info(f"  Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"  Final val loss: {val_losses[-1]:.4f}")
    logger.info(f"  Total epochs: {len(train_losses)}")
    logger.info(f"  Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
