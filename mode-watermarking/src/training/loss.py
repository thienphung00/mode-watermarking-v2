"""
Loss functions for watermark detector training.
Supports BCE, Focal, and combined BCE+Focal losses.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Binary Cross-Entropy loss.
    
    Args:
        predictions: Logits [B, 1] or probabilities [B, 1]
        targets: Labels [B] or [B, 1], dtype=long, values in {0, 1}
        reduction: "mean", "sum", or "none"
    
    Returns:
        Loss scalar or tensor
    """
    # Ensure targets are the right shape and dtype
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    
    # Convert to float for BCE
    targets = targets.float()
    
    # Apply sigmoid if predictions are logits
    if predictions.dim() == 2 and predictions.size(1) == 1:
        # Assume logits, apply sigmoid
        probs = torch.sigmoid(predictions)
    else:
        probs = predictions
    
    # Compute BCE loss
    bce = F.binary_cross_entropy(probs, targets, reduction=reduction)
    
    return bce


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Focal loss for handling class imbalance.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        predictions: Logits [B, 1]
        targets: Labels [B] or [B, 1], dtype=long
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: "mean", "sum", or "none"
    
    Returns:
        Loss scalar or tensor
    """
    # Ensure targets are the right shape and dtype
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    
    targets = targets.float()
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(predictions)
    
    # Compute p_t (probability of true class)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Compute BCE
    bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    
    # Apply focal weight and alpha
    focal = alpha * focal_weight * bce
    
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    else:
        return focal


def bce_focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    bce_weight: float = 0.5,
    focal_weight: float = 0.5,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Combined BCE and Focal loss.
    
    Args:
        predictions: Logits [B, 1]
        targets: Labels [B] or [B, 1], dtype=long
        alpha: Focal loss alpha (default: 0.25)
        gamma: Focal loss gamma (default: 2.0)
        bce_weight: Weight for BCE component (default: 0.5)
        focal_weight: Weight for Focal component (default: 0.5)
        reduction: "mean", "sum", or "none"
    
    Returns:
        Combined loss scalar or tensor
    """
    bce = bce_loss(predictions, targets, reduction=reduction)
    focal = focal_loss(predictions, targets, alpha, gamma, reduction=reduction)
    
    return bce_weight * bce + focal_weight * focal


class WatermarkLoss:
    """
    Unified loss interface for detector training.
    """
    
    def __init__(
        self,
        loss_type: str = "bce_focal",  # "bce", "focal", "bce_focal"
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5
    ):
        """
        Initialize loss function.
        
        Args:
            loss_type: Type of loss to use ("bce", "focal", "bce_focal")
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            bce_weight: Weight for BCE in combined loss
            focal_weight: Weight for Focal in combined loss
        """
        self.loss_type = loss_type.lower()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        if self.loss_type not in ["bce", "focal", "bce_focal"]:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. "
                f"Must be 'bce', 'focal', or 'bce_focal'"
            )
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions (logits) [B, 1]
            targets: Ground truth labels [B] or [B, 1]
        
        Returns:
            Loss scalar
        """
        if self.loss_type == "bce":
            return bce_loss(predictions, targets)
        elif self.loss_type == "focal":
            return focal_loss(
                predictions,
                targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
        elif self.loss_type == "bce_focal":
            return bce_focal_loss(
                predictions,
                targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                bce_weight=self.bce_weight,
                focal_weight=self.focal_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
