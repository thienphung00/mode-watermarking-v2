"""
Bayesian detector for image watermark detection.
Adapted from SynthID Text Bayesian detector for image domain.

Uses Bayes' rule to compute P(watermarked|g_values) from recovered g-values.
Works with binary g-values (±1 mapped to {0, 1}) recovered from images.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreType(Enum):
    """Type of score returned by a WatermarkDetector.
    
    In all cases, larger score corresponds to watermarked image.
    """
    # Negative p-value where the p-value is the probability of observing
    # equal or stronger watermarking in unwatermarked images.
    NEGATIVE_P_VALUE = "negative_p_value"
    
    # Prob(watermarked | g-values).
    POSTERIOR = "posterior"


class LikelihoodModelWatermarked(nn.Module):
    """
    Watermarked likelihood model for binary-valued g-values.
    
    Takes in g-values and returns P(g_values|watermarked).
    Uses logistic regression model to learn the probability distribution
    of g-values under the watermarked hypothesis.
    """
    
    def __init__(
        self,
        watermarking_depth: int,
        init_beta_mean: float = -2.5,
        init_beta_std: float = 0.001,
        init_delta_std: float = 0.001
    ):
        """
        Initialize likelihood model.
        
        Args:
            watermarking_depth: Number of watermarking layers/depth
            init_beta_mean: Mean for beta initialization (default: -2.5)
            init_beta_std: Std for beta initialization (default: 0.001)
            init_delta_std: Std for delta initialization (default: 0.001)
        """
        super().__init__()
        self.watermarking_depth = watermarking_depth
        
        # Beta parameter: [1, 1, depth]
        self.beta = nn.Parameter(
            torch.normal(
                mean=torch.full((1, 1, watermarking_depth), init_beta_mean),
                std=torch.full((1, 1, watermarking_depth), init_beta_std)
            )
        )
        
        # Delta parameter: [1, 1, depth, depth]
        self.delta = nn.Parameter(
            torch.normal(
                mean=torch.zeros(1, 1, watermarking_depth, watermarking_depth),
                std=torch.full((1, 1, watermarking_depth, watermarking_depth), init_delta_std)
            )
        )
    
    def l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss on delta parameters."""
        return torch.sum(self.delta ** 2)
    
    def _compute_latents(self, g_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the probability distribution given g-values.
        
        Args:
            g_values: g-values (0 or 1) of shape [batch_size, spatial_positions, watermarking_depth]
        
        Returns:
            Tuple of (p_one_unique, p_two_unique), both of shape
            [batch_size, spatial_positions, watermarking_depth]
        """
        batch_size, spatial_pos, depth = g_values.shape
        
        # Tile g-values to produce feature vectors for predicting latents
        # Shape: [batch, spatial_pos, depth, depth]
        x = g_values.unsqueeze(-1).repeat(1, 1, 1, depth)
        
        # Apply autoregressive masking (lower triangular)
        # Mask all elements above -1 diagonal
        tril_mask = torch.tril(torch.ones(depth, depth, device=g_values.device), diagonal=-1)
        x = x * tril_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute logits: einsum('bijkl,bijkl->bij', delta, x) + beta
        # delta: [1, 1, depth, depth]
        # x: [batch, spatial_pos, depth, depth]
        logits = torch.einsum('abcd,abcd->abc', self.delta, x) + self.beta
        # logits: [batch, spatial_pos, depth]
        
        # Convert to probabilities
        p_two_unique = torch.sigmoid(logits)
        p_one_unique = 1 - p_two_unique
        
        return p_one_unique, p_two_unique
    
    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihoods P(g_values|watermarked).
        
        Args:
            g_values: g-values (0 or 1) of shape [batch_size, spatial_positions, watermarking_depth]
        
        Returns:
            P(g_values|watermarked) of shape [batch_size, spatial_positions, watermarking_depth]
        """
        p_one_unique, p_two_unique = self._compute_latents(g_values)
        
        # P(g_tl | watermarked) = 0.5 * [(g_tl + 0.5) * p_two_unique + p_one_unique]
        # Convert g_values from {0, 1} to match the formula
        # Note: If g_values are in {0, 1}, we use them directly
        # If they're in {-1, 1}, we convert: (g_values + 1) / 2 -> {0, 1}
        if g_values.min() < 0:
            # Convert from {-1, 1} to {0, 1}
            g_normalized = (g_values + 1) / 2
        else:
            g_normalized = g_values
        
        likelihood = 0.5 * ((g_normalized + 0.5) * p_two_unique + p_one_unique)
        return likelihood


class LikelihoodModelUnwatermarked(nn.Module):
    """
    Unwatermarked likelihood model for binary-valued g-values.
    
    Returns P(g_values|unwatermarked) = 0.5 for all g-values.
    """
    
    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihoods P(g-values|unwatermarked).
        
        Args:
            g_values: g-values (0 or 1) of shape [batch_size, spatial_positions, watermarking_depth, ...]
        
        Returns:
            Likelihoods P(g_values|unwatermarked) = 0.5 for all positions
            Shape: same as g_values
        """
        return 0.5 * torch.ones_like(g_values)


def compute_posterior(
    likelihoods_watermarked: torch.Tensor,
    likelihoods_unwatermarked: torch.Tensor,
    mask: torch.Tensor,
    prior: float
) -> torch.Tensor:
    """
    Compute posterior P(watermarked|g_values) given likelihoods, mask and prior.
    
    Args:
        likelihoods_watermarked: Shape [batch, spatial_pos, depth].
                                P(g_values|watermarked)
        likelihoods_unwatermarked: Shape [batch, spatial_pos, depth].
                               P(g_values|unwatermarked)
        mask: Binary array shape [batch, spatial_pos] indicating which g-values
              should be used. g-values with mask value 0 are discarded.
        prior: Prior probability P(w) that the image is watermarked.
    
    Returns:
        Posterior probability P(watermarked|g_values), shape [batch]
    """
    # Expand mask to match likelihood dimensions
    mask_expanded = mask.unsqueeze(-1)  # [batch, spatial_pos, 1]
    
    # Clip prior to avoid numerical issues
    prior = torch.clamp(torch.tensor(prior, device=likelihoods_watermarked.device), 
                       min=1e-5, max=1.0 - 1e-5)
    
    # Compute log-likelihoods
    log_likelihoods_watermarked = torch.log(
        torch.clamp(likelihoods_watermarked, min=1e-30)
    )
    log_likelihoods_unwatermarked = torch.log(
        torch.clamp(likelihoods_unwatermarked, min=1e-30)
    )
    
    # Compute log odds (relative surprisal)
    log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked
    
    # Sum relative surprisals across spatial positions and layers
    relative_surprisal_likelihood = torch.sum(log_odds * mask_expanded, dim=(1, 2))  # [batch]
    
    # Add prior
    relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)
    relative_surprisal = relative_surprisal_prior + relative_surprisal_likelihood
    
    # Compute posterior: P(w|g) = sigmoid(relative_surprisal)
    posterior = torch.sigmoid(relative_surprisal)
    
    return posterior


class BayesianDetectorModule(nn.Module):
    """
    Bayesian classifier for watermark detection.
    
    Uses Bayes' rule to compute P(watermarked|g_values) from g-values and masks.
    """
    
    def __init__(
        self,
        watermarking_depth: int,
        baserate: float = 0.5,
        init_beta_mean: float = -2.5,
        init_beta_std: float = 0.001,
        init_delta_std: float = 0.001
    ):
        """
        Initialize Bayesian detector module.
        
        Args:
            watermarking_depth: Number of watermarking layers/depth
            baserate: Prior probability P(w) that an image is watermarked (default: 0.5)
            init_beta_mean: Mean for beta initialization (default: -2.5)
            init_beta_std: Std for beta initialization (default: 0.001)
            init_delta_std: Std for delta initialization (default: 0.001)
        """
        super().__init__()
        self.watermarking_depth = watermarking_depth
        self.baserate = baserate
        
        self.likelihood_model_watermarked = LikelihoodModelWatermarked(
            watermarking_depth=watermarking_depth,
            init_beta_mean=init_beta_mean,
            init_beta_std=init_beta_std,
            init_delta_std=init_delta_std
        )
        self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
        
        # Prior as a learnable parameter (clamped during forward pass)
        self.prior_logit = nn.Parameter(torch.tensor(np.log(baserate / (1 - baserate))))
    
    @property
    def score_type(self) -> ScoreType:
        return ScoreType.POSTERIOR
    
    def l2_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss on delta parameters."""
        return self.likelihood_model_watermarked.l2_loss()
    
    def forward(
        self,
        g_values: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute posterior P(watermarked|g_values).
        
        Args:
            g_values: g-values (0 or 1, or -1/1) of shape 
                     [batch_size, spatial_positions, watermarking_depth]
            mask: Binary array shape [batch_size, spatial_positions] indicating
                  which g-values should be used. g-values with mask value 0 are discarded.
        
        Returns:
            P(watermarked|g_values) of shape [batch_size]
        """
        # Ensure g_values are in correct shape
        if g_values.dim() == 4:
            # Flatten spatial dimensions: [batch, H, W, depth] -> [batch, H*W, depth]
            batch, H, W, depth = g_values.shape
            g_values = g_values.view(batch, H * W, depth)
            mask = mask.view(batch, H * W) if mask.dim() == 3 else mask
        
        # Compute likelihoods
        likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
        likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values)
        
        # Get prior from logit parameter
        prior = torch.sigmoid(self.prior_logit).item()
        
        # Compute posterior
        posterior = compute_posterior(
            likelihoods_watermarked,
            likelihoods_unwatermarked,
            mask,
            prior
        )
        
        return posterior


def cross_entropy_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels [batch] (0 or 1)
        y_pred: Predicted probabilities [batch] (0 to 1)
    
    Returns:
        Scalar loss value
    """
    y_pred = torch.clamp(y_pred, min=1e-5, max=1.0 - 1e-5)
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean()


def tpr_at_fpr(
    model: BayesianDetectorModule,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    target_fpr: float = 0.01,
    batch_size: int = 64
) -> float:
    """
    Calculate True Positive Rate at a given False Positive Rate.
    
    Args:
        model: Trained Bayesian detector module
        g_values: g-values tensor [N, spatial_pos, depth]
        mask: Mask tensor [N, spatial_pos]
        labels: Binary labels [N] (0 or 1)
        target_fpr: Target false positive rate (default: 0.01 = 1%)
        batch_size: Batch size for evaluation
    
    Returns:
        TPR at target FPR
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for all samples
        predictions = []
        for i in range(0, len(g_values), batch_size):
            end = min(i + batch_size, len(g_values))
            batch_g = g_values[i:end]
            batch_m = mask[i:end]
            pred = model(batch_g, batch_m)
            predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions)
        
        # Separate positive and negative scores
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        positive_scores = predictions[positive_mask]
        negative_scores = predictions[negative_mask]
        
        # Find threshold at target FPR
        if len(negative_scores) == 0:
            return 0.0
        
        threshold_idx = int((1 - target_fpr) * len(negative_scores))
        threshold = torch.sort(negative_scores)[0][threshold_idx]
        
        # Compute TPR
        if len(positive_scores) == 0:
            return 0.0
        
        tpr = (positive_scores >= threshold).float().mean().item()
        return tpr


class ValidationMetric(Enum):
    """Validation metric options."""
    TPR_AT_FPR = "tpr_at_fpr"
    CROSS_ENTROPY = "cross_entropy"


def train_bayesian_detector(
    model: BayesianDetectorModule,
    g_values_train: torch.Tensor,
    mask_train: torch.Tensor,
    labels_train: torch.Tensor,
    g_values_val: Optional[torch.Tensor] = None,
    mask_val: Optional[torch.Tensor] = None,
    labels_val: Optional[torch.Tensor] = None,
    epochs: int = 50,
    learning_rate: float = 2.1e-2,
    minibatch_size: int = 64,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
    target_fpr: float = 0.01,
    device: str = "cuda",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train Bayesian detector model.
    
    Args:
        model: BayesianDetectorModule to train
        g_values_train: Training g-values [N_train, spatial_pos, depth]
        mask_train: Training masks [N_train, spatial_pos]
        labels_train: Training labels [N_train] (0 or 1)
        g_values_val: Validation g-values [N_val, spatial_pos, depth] (optional)
        mask_val: Validation masks [N_val, spatial_pos] (optional)
        labels_val: Validation labels [N_val] (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        minibatch_size: Minibatch size for training
        l2_weight: Weight for L2 regularization on delta parameters
        shuffle: Whether to shuffle training data
        validation_metric: Metric to use for validation ("tpr_at_fpr" or "cross_entropy")
        target_fpr: Target FPR for TPR@FPR metric (default: 0.01)
        device: Device to train on
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history:
        - "history": Dict mapping epoch -> {"loss": ..., "val_loss": ...}
        - "best_val_loss": Best validation loss
        - "best_epoch": Best epoch number
    """
    model = model.to(device)
    g_values_train = g_values_train.to(device)
    mask_train = mask_train.to(device)
    labels_train = labels_train.to(device)
    
    if g_values_val is not None:
        g_values_val = g_values_val.to(device)
        mask_val = mask_val.to(device)
        labels_val = labels_val.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Shuffle training data
    if shuffle:
        indices = torch.randperm(len(g_values_train), device=device)
        g_values_train = g_values_train[indices]
        mask_train = mask_train[indices]
        labels_train = labels_train[indices]
    
    # Training history
    history = {}
    best_val_loss = float('inf')
    best_epoch = 0
    
    n_minibatches = len(g_values_train) // minibatch_size
    l2_batch_weight = l2_weight / max(n_minibatches, 1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Train on minibatches
        for i in range(0, len(g_values_train), minibatch_size):
            end = min(i + minibatch_size, len(g_values_train))
            
            batch_g = g_values_train[i:end]
            batch_m = mask_train[i:end]
            batch_labels = labels_train[i:end].float()
            
            # Forward pass
            predictions = model(batch_g, batch_m)
            
            # Compute loss
            ce_loss = cross_entropy_loss(batch_labels, predictions)
            l2_loss = model.l2_loss()
            loss = ce_loss + l2_batch_weight * l2_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        # Validation
        val_loss = None
        if g_values_val is not None:
            model.eval()
            with torch.no_grad():
                if validation_metric == ValidationMetric.TPR_AT_FPR:
                    # Negative TPR@FPR as loss (higher TPR = lower loss)
                    tpr = tpr_at_fpr(model, g_values_val, mask_val, labels_val, 
                                   target_fpr=target_fpr, batch_size=minibatch_size)
                    val_loss = -tpr  # Negative because we want to maximize TPR
                else:  # CROSS_ENTROPY
                    # Compute cross-entropy loss on validation set
                    val_predictions = []
                    for i in range(0, len(g_values_val), minibatch_size):
                        end = min(i + minibatch_size, len(g_values_val))
                        batch_g = g_values_val[i:end]
                        batch_m = mask_val[i:end]
                        pred = model(batch_g, batch_m)
                        val_predictions.append(pred)
                    
                    val_predictions = torch.cat(val_predictions)
                    val_loss = cross_entropy_loss(labels_val.float(), val_predictions).item()
        
        # Track best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model state
            best_state = model.state_dict().copy()
        
        history[epoch] = {
            "loss": avg_loss,
            "val_loss": val_loss
        }
        
        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch}: loss={avg_loss:.4f} (train), val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: loss={avg_loss:.4f} (train)")
    
    # Load best model state
    if val_loss is not None and best_epoch > 0:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    }
