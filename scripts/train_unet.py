#!/usr/bin/env python3
"""
Train watermark detector models.

Minimal training pipeline for UNetDetector only.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.core.config import AppConfig
from src.data.latent_dataset import LatentDataset
from src.models.detectors import UNetDetector
from scripts.utils import setup_logging, get_device
from torch.utils.data import DataLoader


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


def create_latent_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int,
    num_workers: int = 4,
    latent_root: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from latent manifests.
    
    Args:
        train_manifest: Path to training latent manifest
        val_manifest: Path to validation latent manifest
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        latent_root: Root directory for resolving latent paths (optional)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LatentDataset(
        manifest_path=train_manifest,
        latent_root=latent_root,
    )
    
    val_dataset = LatentDataset(
        manifest_path=val_manifest,
        latent_root=latent_root,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


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
        train_loader: Training data loader (returns latents)
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        # Move batch to device (tensors are on CPU from DataLoader workers)
        # batch["image"] contains latents of shape [B, 4, 64, 64] (fp16)
        # Convert to fp32 for model training (model is fp32)
        latents = batch["image"].float().to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)
        
        # Forward pass
        # Call model with validate=False to avoid warnings during training
        logits = model(latents, validate=False).squeeze(-1)
        
        # Compute loss
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with running average loss
        avg_loss = total_loss / num_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
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
        val_loader: Validation data loader (returns latents)
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
            # Move batch to device (tensors are on CPU from DataLoader workers)
            # batch["image"] contains latents of shape [B, 4, 64, 64] (fp16)
            # Convert to fp32 for model validation (model is fp32)
            latents = batch["image"].float().to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)
            
            # Forward pass
            logits = model(latents, validate=False).squeeze(-1)
            
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
    
    # DataLoader workers
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: 4, 0 to disable multiprocessing)",
    )
    
    # Latent root directory
    parser.add_argument(
        "--latent-root",
        type=str,
        default=None,
        help="Root directory for resolving latent paths (optional)",
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
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    
    logger.info("Configuration:")
    logger.info(f"  Detector type: {config.training.detector_type}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")
    
    # Get device (FP32 only, no FP16)
    device = get_device(args.device, use_fp16=False)
    logger.info(f"Device: {device}")
    
    # Validate that we're using latent manifests
    logger.info("=" * 80)
    logger.info("Training on precomputed latents")
    logger.info("=" * 80)
    
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
    
    # Create dataloaders from latent manifests
    logger.info("Setting up data pipeline...")
    logger.info(f"Training manifest: {config.training.train_manifest}")
    logger.info(f"Validation manifest: {config.training.val_manifest}")
    
    train_loader, val_loader = create_latent_dataloaders(
        train_manifest=config.training.train_manifest,
        val_manifest=config.training.val_manifest,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        latent_root=args.latent_root,
    )
    
    # Validate latent shapes from first batch
    logger.info("Validating latent dataset...")
    sample_batch = next(iter(train_loader))
    sample_latent = sample_batch["image"][0]
    sample_label = sample_batch["label"][0]
    
    assert sample_latent.shape == (4, 64, 64), (
        f"Invalid latent shape: {sample_latent.shape}, expected (4, 64, 64)"
    )
    assert sample_label.item() in [0, 1], (
        f"Invalid label: {sample_label.item()}, must be 0 or 1"
    )
    logger.info(f"✓ Latent shape validated: {sample_latent.shape}")
    logger.info(f"✓ Label validated: {sample_label.item()}")
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"DataLoader workers: {config.training.num_workers}")
    
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
        # Log epoch start
        logger.info(f"Epoch [{epoch}/{config.training.epochs - 1}] starting")
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        # Calculate epoch duration
        epoch_time = time.time() - start_time
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"time={epoch_time:.1f}s"
        )
        
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
