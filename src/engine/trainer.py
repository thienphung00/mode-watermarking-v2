"""
Unified training loop for watermark detectors.

Model-agnostic trainer that accepts any detector implementing the interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class DetectorTrainer:
    """
    Unified trainer for watermark detectors.

    Handles training loop, validation, checkpointing, and logging
    in a model-agnostic way.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Any,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        max_checkpoints: int = 5,
        input_representation: str = "z_0",
        extraction_method: str = "none",
        vae_scale_factor: float = 0.18215,
    ):
        """
        Initialize detector trainer.

        Args:
            model: Detector model implementing training_step/validation_step interface
            optimizer: Optimizer instance
            loss_fn: Loss function
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            input_representation: Input representation type ("z_0", "z_T", "g_binary")
            extraction_method: Extraction method applied ("none", "whitened", "normalized", "sign")
            vae_scale_factor: VAE scaling factor (default: 0.18215)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.max_checkpoints = max_checkpoints
        
        # Input representation metadata (for checkpoint self-description)
        self.input_representation = input_representation
        self.extraction_method = extraction_method
        self.vae_scale_factor = vae_scale_factor

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_every_n_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            log_every_n_steps: Logging frequency

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass
            if hasattr(self.model, "training_step"):
                # Use model's training_step if available
                result = self.model.training_step(batch, batch_idx)
                loss = result["loss"]
            else:
                # Standard training step
                outputs = self.model(batch["image"])
                loss = self.loss_fn(outputs, batch["label"])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            if (batch_idx + 1) % log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Compute average metrics
        avg_loss = total_loss / max(num_batches, 1)

        return {"train_loss": avg_loss}

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Move batch to device
                batch = self._to_device(batch)

                # Forward pass
                if hasattr(self.model, "validation_step"):
                    # Use model's validation_step if available
                    result = self.model.validation_step(batch, batch_idx)
                    loss = result["loss"]
                else:
                    # Standard validation step
                    outputs = self.model(batch["image"])
                    loss = self.loss_fn(outputs, batch["label"])

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

        # Compute average metrics
        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        log_every_n_steps: int = 10,
        save_every_n_epochs: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            log_every_n_steps: Logging frequency
            save_every_n_epochs: Checkpoint saving frequency

        Returns:
            Dictionary containing training history
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, log_every_n_steps)
            history["train_loss"].append(train_metrics["train_loss"])

            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                print(f"Epoch {epoch}: val_loss={val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best.pt")

            # Save periodic checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

        return history

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint with self-describing metadata.

        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            return

        checkpoint_path = self.checkpoint_dir / filename

        # Determine expected shape from model
        if hasattr(self.model, "inc"):
            # UNetDetector
            expected_shape = [4, 64, 64]
        elif hasattr(self.model, "embedding"):
            # BayesianDetector - shape depends on input, use None
            expected_shape = None
        else:
            expected_shape = None

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            # Minimal checkpoint metadata (self-describing)
            "metadata": {
                "input_representation": self.input_representation,
                "extraction_method": self.extraction_method,
                "expected_shape": expected_shape,
                "vae_scale_factor": self.vae_scale_factor,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        validate_metadata: bool = True,
        expected_input_representation: Optional[str] = None,
        expected_extraction_method: Optional[str] = None,
    ) -> None:
        """
        Load model checkpoint with optional metadata validation.

        Args:
            checkpoint_path: Path to checkpoint file
            validate_metadata: If True, validate checkpoint metadata against current config
            expected_input_representation: Expected input representation (for validation)
            expected_extraction_method: Expected extraction method (for validation)
        
        Raises:
            ValueError: If metadata validation fails and validate_metadata=True
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Validate metadata if requested
        if validate_metadata and "metadata" in checkpoint:
            metadata = checkpoint["metadata"]
            
            if expected_input_representation is not None:
                if metadata.get("input_representation") != expected_input_representation:
                    raise ValueError(
                        f"Checkpoint expects input_representation='{metadata.get('input_representation')}', "
                        f"but current pipeline produces '{expected_input_representation}'. "
                        f"Mismatch will cause detection failures."
                    )
            
            if expected_extraction_method is not None:
                if metadata.get("extraction_method") != expected_extraction_method:
                    raise ValueError(
                        f"Checkpoint expects extraction_method='{metadata.get('extraction_method')}', "
                        f"but current pipeline uses '{expected_extraction_method}'. "
                        f"Mismatch will cause detection failures."
                    )
            
            # Update trainer metadata from checkpoint
            self.input_representation = metadata.get("input_representation", self.input_representation)
            self.extraction_method = metadata.get("extraction_method", self.extraction_method)
            self.vae_scale_factor = metadata.get("vae_scale_factor", self.vae_scale_factor)
            
            print(f"Checkpoint metadata validated: {metadata}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        if self.checkpoint_dir is None:
            return

        # Get all checkpoint files (excluding best.pt)
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("epoch_*.pt")], key=lambda x: x.stat().st_mtime
        )

        # Remove old checkpoints
        while len(checkpoints) > self.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            print(f"Removed old checkpoint: {old_checkpoint}")

