"""
High-level training experiment orchestrator.
Coordinates data loading, model initialization, and training loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..config.config_loader import ConfigLoader
from ..data import create_dataloaders, LatentWatermarkDataset, get_detector_transforms
from ..models.unet_detector import UNetDetector
from ..models.bayesian_detector import BayesianDetectorModule, ValidationMetric
from ..utils.logger import ExperimentLogger, LoggerConfig
from ..utils.io import ReproIO, ensure_dir
from .loss import WatermarkLoss
from .trainer import DetectorTrainer


def train_unet_detector(
    config_paths: Dict[str, str],
    sd_pipeline: Optional[Any] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level function to train UNet detector.
    
    Args:
        config_paths: Dictionary with paths to config files
            - "train": train_config.yaml
        sd_pipeline: Initialized SD pipeline (for VAE encoder access)
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
    
    Returns:
        Dictionary with training results:
        - "best_checkpoint": Path to best model
        - "final_metrics": Final validation metrics
        - "training_history": Per-epoch metrics
    """
    # 1. Load configs
    config_loader = ConfigLoader()
    train_config = config_loader.load_yaml(config_paths["train"])
    unet_cfg = train_config["unet_detector"]
    common_cfg = train_config["common"]
    
    # Set random seed for reproducibility
    ReproIO.seed_all(common_cfg["seed"])
    
    # 2. Set up device
    device = common_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # 3. Set up data loaders
    # Get VAE encoder from SD pipeline
    vae_encoder = sd_pipeline.vae if sd_pipeline else None
    
    if vae_encoder is None:
        print("Warning: VAE encoder not provided. Latent encoding will be disabled.")
    
    # Create transforms
    transform = get_detector_transforms(
        detector_type="unet",
        vae_encoder=vae_encoder,
        mode="train",
        image_size=tuple(unet_cfg["data"].get("image_size", [512, 512]))
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_manifest=unet_cfg["data"]["train_split"],
        val_manifest=unet_cfg["data"]["val_split"],
        dataset_class=LatentWatermarkDataset,
        batch_size=unet_cfg["data"]["batch_size"],
        num_workers=unet_cfg["data"]["num_workers"],
        vae_encoder=vae_encoder,
        transform=transform,
        image_root=unet_cfg["data"].get("image_root")
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # 4. Initialize model
    model = UNetDetector(
        input_channels=unet_cfg["model"]["input_channels"],
        base_channels=unet_cfg["model"]["base_channels"],
        num_classes=unet_cfg["model"]["num_classes"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Initialize loss
    loss_fn = WatermarkLoss(
        loss_type=unet_cfg["loss"]["type"],
        focal_alpha=unet_cfg["loss"]["focal_alpha"],
        focal_gamma=unet_cfg["loss"]["focal_gamma"]
    )
    
    # 6. Initialize optimizer
    optimizer_name = unet_cfg["training"].get("optimizer", "adam").lower()
    lr = unet_cfg["training"]["learning_rate"]
    weight_decay = unet_cfg["training"].get("weight_decay", 0.0001)
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # 7. Initialize scheduler
    scheduler = None
    scheduler_type = unet_cfg["training"].get("scheduler", "cosine").lower()
    epochs = unet_cfg["training"]["epochs"]
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=epochs // 3,
            gamma=0.1
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )
    # None for "none" scheduler
    
    # 8. Initialize logger
    log_cfg = common_cfg.get("logging", {})
    logger_config = LoggerConfig(
        backend="tensorboard",  # Default to tensorboard
        project=log_cfg.get("wandb_project"),
        run_name=f"unet_detector_epochs_{epochs}",
        log_dir=log_cfg.get("tensorboard_dir", "outputs/experiments/detectors"),
        tags=["unet_detector", "watermarking"]
    )
    logger = ExperimentLogger(logger_config)
    
    # Store log_every in logger for trainer to access
    logger._log_every = log_cfg.get("log_every", 100)
    
    # 9. Initialize trainer
    trainer = DetectorTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=unet_cfg["training"].get("mixed_precision", True),
        gradient_accumulation_steps=unet_cfg["training"].get("gradient_accumulation_steps", 1),
        logger=logger
    )
    
    # 10. Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    training_history = []
    
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        state = trainer.load_checkpoint(resume_from_checkpoint)
        start_epoch = state["epoch"] + 1
        best_val_loss = state["metrics"].get("loss", float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")
    
    # 11. Training loop
    print("\nStarting training...")
    checkpoint_dir = unet_cfg["checkpoint"]["save_dir"]
    save_every = unet_cfg["checkpoint"].get("save_every", 10)
    keep_last_n = unet_cfg["checkpoint"].get("keep_last_n", 5)
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Update scheduler
        if scheduler:
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()
        
        # Log metrics
        logger.log_metrics({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"]
        }, step=epoch)
        
        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            print(f"✓ New best model! Val loss: {best_val_loss:.4f}")
        
        # Save checkpoint every N epochs or if best
        if (epoch + 1) % save_every == 0 or is_best:
            checkpoint_path = trainer.save_checkpoint(
                epoch=epoch,
                metrics=val_metrics,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            trainer.cleanup_old_checkpoints(checkpoint_dir, keep_last_n=keep_last_n)
        
        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics
        })
    
    # Finalize logger
    logger.finalize(status="completed")
    
    # Get best checkpoint path
    best_checkpoint_path = Path(checkpoint_dir) / "best_model.ckpt"
    best_checkpoint = str(best_checkpoint_path) if best_checkpoint_path.exists() else None
    
    print(f"\n✓ Training complete!")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Best val loss: {best_val_loss:.4f}")
    
    return {
        "best_checkpoint": best_checkpoint,
        "final_metrics": val_metrics,
        "training_history": training_history
    }


def train_bayesian_detector(
    config_paths: Dict[str, str],
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level function to train Bayesian detector.
    
    NOTE: Bayesian detector training requires g-values recovered from images.
    This assumes g-values have been pre-computed and saved as numpy arrays
    (from Stage 5: Detection & Evaluation).
    
    Args:
        config_paths: Dictionary with paths to config files
            - "train": train_config.yaml
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
    
    Returns:
        Dictionary with training results:
        - "best_checkpoint": Path to best model
        - "final_metrics": Final validation metrics
        - "training_history": Per-epoch metrics
    """
    # 1. Load configs
    config_loader = ConfigLoader()
    train_config = config_loader.load_yaml(config_paths["train"])
    bayesian_cfg = train_config["bayesian_detector"]
    common_cfg = train_config["common"]
    
    # Set random seed for reproducibility
    ReproIO.seed_all(common_cfg["seed"])
    
    # 2. Set up device
    device = common_cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # 3. Load pre-computed g-values, masks, and labels
    print("Loading pre-computed g-values, masks, and labels...")
    
    # Load training data
    train_g_values = np.load(bayesian_cfg["data"]["train_g_values"])
    train_masks = np.load(bayesian_cfg["data"]["train_masks"])
    train_labels = np.load(bayesian_cfg["data"]["train_labels"])
    
    # Load validation data (if available)
    val_g_values = None
    val_masks = None
    val_labels = None
    
    if "val_g_values" in bayesian_cfg["data"] and bayesian_cfg["data"]["val_g_values"]:
        val_g_values = np.load(bayesian_cfg["data"]["val_g_values"])
        val_masks = np.load(bayesian_cfg["data"]["val_masks"])
        val_labels = np.load(bayesian_cfg["data"]["val_labels"])
    
    print(f"Training samples: {len(train_g_values)}")
    if val_g_values is not None:
        print(f"Validation samples: {len(val_g_values)}")
    
    # Convert to tensors
    train_g_values = torch.from_numpy(train_g_values).float()
    train_masks = torch.from_numpy(train_masks).float()
    train_labels = torch.from_numpy(train_labels).long()
    
    if val_g_values is not None:
        val_g_values = torch.from_numpy(val_g_values).float()
        val_masks = torch.from_numpy(val_masks).float()
        val_labels = torch.from_numpy(val_labels).long()
    
    # Ensure g-values are in correct format
    # Expected: [N, spatial_pos, depth] or [N, H, W, depth]
    # If g-values are in {-1, 1}, convert to {0, 1} for Bayesian detector
    if train_g_values.min() < 0:
        print("Converting g-values from {-1, 1} to {0, 1}")
        train_g_values = (train_g_values + 1) / 2
        if val_g_values is not None:
            val_g_values = (val_g_values + 1) / 2
    
    # Determine watermarking_depth from data
    if train_g_values.dim() == 4:
        # [N, H, W, depth] -> flatten to [N, H*W, depth]
        _, H, W, depth = train_g_values.shape
        train_g_values = train_g_values.view(len(train_g_values), H * W, depth)
        train_masks = train_masks.view(len(train_masks), H * W)
        
        if val_g_values is not None:
            _, H, W, depth = val_g_values.shape
            val_g_values = val_g_values.view(len(val_g_values), H * W, depth)
            val_masks = val_masks.view(len(val_masks), H * W)
    elif train_g_values.dim() == 3:
        # [N, spatial_pos, depth]
        depth = train_g_values.shape[2]
    else:
        raise ValueError(f"Unexpected g_values shape: {train_g_values.shape}")
    
    watermarking_depth = depth if bayesian_cfg["model"]["watermarking_depth"] is None else bayesian_cfg["model"]["watermarking_depth"]
    print(f"Watermarking depth: {watermarking_depth}")
    
    # 4. Initialize model
    model = BayesianDetectorModule(
        watermarking_depth=watermarking_depth,
        baserate=bayesian_cfg["model"]["baserate"]
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Initialize logger
    log_cfg = common_cfg.get("logging", {})
    logger_config = LoggerConfig(
        backend="tensorboard",
        project=log_cfg.get("wandb_project"),
        run_name=f"bayesian_detector_epochs_{bayesian_cfg['training']['epochs']}",
        log_dir=log_cfg.get("tensorboard_dir", "outputs/experiments/detectors"),
        tags=["bayesian_detector", "watermarking"]
    )
    logger = ExperimentLogger(logger_config)
    logger._log_every = log_cfg.get("log_every", 100)
    
    # 6. Determine validation metric
    validation_metric_str = bayesian_cfg["validation"].get("metric", "tpr_at_fpr")
    if validation_metric_str == "tpr_at_fpr":
        validation_metric = ValidationMetric.TPR_AT_FPR
    else:
        validation_metric = ValidationMetric.CROSS_ENTROPY
    
    # 7. Train model
    print("\nStarting training...")
    
    from ..models.bayesian_detector import train_bayesian_detector as train_detector
    
    training_result = train_detector(
        model=model,
        g_values_train=train_g_values,
        mask_train=train_masks,
        labels_train=train_labels,
        g_values_val=val_g_values,
        mask_val=val_masks,
        labels_val=val_labels,
        epochs=bayesian_cfg["training"]["epochs"],
        learning_rate=bayesian_cfg["training"]["learning_rate"],
        minibatch_size=bayesian_cfg["training"]["minibatch_size"],
        l2_weight=bayesian_cfg["regularization"]["l2_weight"],
        shuffle=bayesian_cfg["training"].get("shuffle", True),
        validation_metric=validation_metric,
        device=device,
        verbose=common_cfg["logging"].get("verbose", False)
    )
    
    # 8. Log training history
    history = training_result["history"]
    for epoch, metrics in history.items():
        logger.log_metrics({
            "epoch": epoch,
            **metrics
        }, step=epoch)
    
    # 9. Save checkpoint
    checkpoint_dir = bayesian_cfg["checkpoint"]["save_dir"]
    ensure_dir(checkpoint_dir)
    
    checkpoint_path = Path(checkpoint_dir) / "best_model.ckpt"
    torch.save({
        "epoch": training_result["best_epoch"],
        "model_state_dict": model.state_dict(),
        "metrics": {
            "val_loss": training_result["best_val_loss"]
        },
        "watermarking_depth": watermarking_depth,
        "baserate": bayesian_cfg["model"]["baserate"]
    }, checkpoint_path)
    
    print(f"\n✓ Training complete!")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best epoch: {training_result['best_epoch']}")
    print(f"Best val loss: {training_result['best_val_loss']:.4f}")
    
    logger.finalize(status="completed")
    
    return {
        "best_checkpoint": str(checkpoint_path),
        "final_metrics": {"val_loss": training_result["best_val_loss"]},
        "training_history": history
    }
