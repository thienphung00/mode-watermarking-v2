"""
Training loop for watermark detector.
Handles forward/backward passes, optimization, checkpointing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..utils.io import ensure_dir, AssetStore
from ..utils.logger import ExperimentLogger


class DetectorTrainer:
    """
    Training loop for watermark detector.
    Handles forward/backward passes, optimization, checkpointing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Detector model (UNetDetector)
            loss_fn: Loss function (WatermarkLoss)
            optimizer: Optimizer (Adam, AdamW)
            scheduler: Learning rate scheduler (optional)
            device: Training device (default: "cuda")
            mixed_precision: Use FP16 mixed precision (default: True)
            gradient_accumulation_steps: Steps for gradient accumulation (default: 1)
            logger: Experiment logger (optional)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logger = logger
        
        # Move model to device
        self.model.to(device)
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Checkpoint management
        self.checkpoint_history = []
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            epoch: Current epoch number
        
        Returns:
            Dictionary with metrics (loss, accuracy, etc.)
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        # Reset gradients at start of epoch
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: only update every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Compute metrics
            with torch.no_grad():
                # Convert logits to predictions
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).long()
                
                # Squeeze if needed
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
            
            # Log every N steps
            log_every = getattr(self.logger, "_log_every", 100) if self.logger else 100
            if self.logger and self.global_step % log_every == 0:
                step_loss = total_loss / num_batches if num_batches > 0 else 0.0
                step_acc = correct / total if total > 0 else 0.0
                self.logger.log_metrics({
                    "train_loss": step_loss,
                    "train_accuracy": step_acc,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                }, step=self.global_step)
        
        # Average metrics over epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation DataLoader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Forward pass
                with autocast(enabled=self.mixed_precision):
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, labels)
                
                # Compute metrics
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).long()
                
                # Squeeze if needed
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: str,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            checkpoint_dir: Directory to save checkpoint
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        ensure_dir(checkpoint_dir)
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "is_best": is_best
        }
        
        # Add scheduler state if available
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = Path(checkpoint_dir) / "best_model.ckpt"
        else:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.ckpt"
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "metrics": metrics,
                "is_best": is_best
            }, f, indent=2)
        
        # Track checkpoint history
        self.checkpoint_history.append(str(checkpoint_path))
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dictionary with loaded state (epoch, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Update training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        return {
            "epoch": self.current_epoch,
            "metrics": checkpoint.get("metrics", {}),
            "is_best": checkpoint.get("is_best", False)
        }
    
    def cleanup_old_checkpoints(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5
    ) -> None:
        """
        Remove old checkpoints, keeping only the last N.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_last_n: Number of checkpoints to keep (default: 5)
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return
        
        # Get all checkpoint files (excluding best_model.ckpt)
        checkpoint_files = sorted(
            checkpoint_path.glob("checkpoint_epoch_*.ckpt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[keep_last_n:]:
            checkpoint_file.unlink(missing_ok=True)
            # Also remove metadata JSON
            checkpoint_file.with_suffix(".json").unlink(missing_ok=True)
