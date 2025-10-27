"""
Watermark detection module for image watermarking.

Adapted from synthid-text detector training approaches for image watermarking.
Implements Bayesian and mean-based detection with proper training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings

from ..utils.utils import (
    WatermarkConfig, keyed_hash, pool_latent_features, get_timestep_bucket,
    compute_patch_coordinates, validate_watermark_key
)


@dataclass
class DetectionResult:
    """Result of watermark detection."""
    is_watermarked: bool
    probability: float
    confidence: float
    model_id: Optional[str] = None
    detection_scores: Optional[Dict[str, float]] = None


@dataclass
class TrainingData:
    """Training data for detector."""
    g_values: torch.Tensor  # Shape: [batch_size, num_scales, num_patches]
    masks: torch.Tensor     # Shape: [batch_size, num_scales, num_patches]
    labels: torch.Tensor    # Shape: [batch_size] (0: unwatermarked, 1: watermarked)
    model_ids: Optional[torch.Tensor] = None  # Shape: [batch_size]


class ImageMeanDetector:
    """
    Mean-based detector adapted from synthid-text for image watermarking.
    
    Uses statistical scoring across spatial patches and scales.
    """
    
    def __init__(self, config: Optional[WatermarkConfig] = None):
        """
        Initialize mean detector.
        
        Args:
            config: Watermark configuration
        """
        self.config = config or WatermarkConfig()
        self.scale_weights = self._compute_scale_weights()
    
    def _compute_scale_weights(self) -> Dict[int, float]:
        """Compute weights for different scales."""
        scales = list(self.config.scales)
        # Linear decreasing weights (larger scales get higher weights)
        weights = np.linspace(len(scales), 1, len(scales))
        weights = weights / np.sum(weights) * len(scales)  # Normalize
        
        return {scale: weight for scale, weight in zip(scales, weights)}
    
    def mean_score(
        self,
        g_values: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mean score across scales and patches.
        
        Args:
            g_values: G-values of shape [batch_size, num_scales, num_patches]
            masks: Binary masks of shape [batch_size, num_scales, num_patches]
            
        Returns:
            Mean scores of shape [batch_size]
        """
        # Apply masks
        masked_g_values = g_values * masks
        
        # Compute mean per scale
        scale_scores = []
        for i, scale in enumerate(self.config.scales):
            scale_g_values = masked_g_values[:, i, :]  # [batch_size, num_patches]
            scale_masks = masks[:, i, :]  # [batch_size, num_patches]
            
            # Count valid patches per sample
            num_valid = torch.sum(scale_masks, dim=1)  # [batch_size]
            num_valid = torch.clamp(num_valid, min=1)  # Avoid division by zero
            
            # Compute mean for this scale
            scale_mean = torch.sum(scale_g_values, dim=1) / num_valid
            scale_scores.append(scale_mean)
        
        # Stack and apply scale weights
        scale_scores = torch.stack(scale_scores, dim=1)  # [batch_size, num_scales]
        weights = torch.tensor([self.scale_weights[scale] for scale in self.config.scales])
        weights = weights.to(scale_scores.device)
        
        # Weighted average across scales
        final_scores = torch.sum(scale_scores * weights, dim=1) / torch.sum(weights)
        
        return final_scores
    
    def weighted_mean_score(
        self,
        g_values: torch.Tensor,
        masks: torch.Tensor,
        patch_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted mean score with patch-level weighting.
        
        Args:
            g_values: G-values of shape [batch_size, num_scales, num_patches]
            masks: Binary masks of shape [batch_size, num_scales, num_patches]
            patch_weights: Optional patch weights of shape [num_patches]
            
        Returns:
            Weighted mean scores of shape [batch_size]
        """
        if patch_weights is None:
            # Default: linearly decreasing weights
            num_patches = g_values.shape[2]
            patch_weights = torch.linspace(10, 1, num_patches)
            patch_weights = patch_weights / torch.sum(patch_weights) * num_patches
        
        # Apply patch weights
        weighted_g_values = g_values * patch_weights.unsqueeze(0).unsqueeze(0)
        
        # Apply masks
        masked_g_values = weighted_g_values * masks
        
        # Compute weighted mean per scale
        scale_scores = []
        for i, scale in enumerate(self.config.scales):
            scale_g_values = masked_g_values[:, i, :]  # [batch_size, num_patches]
            scale_masks = masks[:, i, :]  # [batch_size, num_patches]
            
            # Count valid patches per sample
            num_valid = torch.sum(scale_masks, dim=1)  # [batch_size]
            num_valid = torch.clamp(num_valid, min=1)  # Avoid division by zero
            
            # Compute weighted mean for this scale
            scale_mean = torch.sum(scale_g_values, dim=1) / num_valid
            scale_scores.append(scale_mean)
        
        # Stack and apply scale weights
        scale_scores = torch.stack(scale_scores, dim=1)  # [batch_size, num_scales]
        weights = torch.tensor([self.scale_weights[scale] for scale in self.config.scales])
        weights = weights.to(scale_scores.device)
        
        # Weighted average across scales
        final_scores = torch.sum(scale_scores * weights, dim=1) / torch.sum(weights)
        
        return final_scores


class ImageBayesianDetector(nn.Module):
    """
    Bayesian detector adapted from synthid-text for image watermarking.
    
    Uses likelihood models for watermarked/unwatermarked distributions.
    """
    
    def __init__(
        self,
        config: Optional[WatermarkConfig] = None,
        baserate: float = 0.5
    ):
        """
        Initialize Bayesian detector.
        
        Args:
            config: Watermark configuration
            baserate: Prior probability that an image is watermarked
        """
        super().__init__()
        self.config = config or WatermarkConfig()
        self.baserate = baserate
        
        # Initialize likelihood models
        self.watermarked_model = WatermarkedLikelihoodModel(self.config)
        self.unwatermarked_model = UnwatermarkedLikelihoodModel(self.config)
        
        # Prior parameter (learnable)
        self.prior = nn.Parameter(torch.tensor(baserate))
    
    def forward(
        self,
        g_values: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute posterior probability P(watermarked | g_values).
        
        Args:
            g_values: G-values of shape [batch_size, num_scales, num_patches]
            masks: Binary masks of shape [batch_size, num_scales, num_patches]
            
        Returns:
            Posterior probabilities of shape [batch_size]
        """
        # Compute likelihoods
        likelihood_watermarked = self.watermarked_model(g_values, masks)
        likelihood_unwatermarked = self.unwatermarked_model(g_values, masks)
        
        # Compute posterior using Bayes' rule
        posterior = self._compute_posterior(
            likelihood_watermarked, likelihood_unwatermarked, masks
        )
        
        return posterior
    
    def _compute_posterior(
        self,
        likelihood_watermarked: torch.Tensor,
        likelihood_unwatermarked: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute posterior probability using Bayes' rule."""
        # Apply masks to likelihoods
        masked_likelihood_w = likelihood_watermarked * masks
        masked_likelihood_u = likelihood_unwatermarked * masks
        
        # Sum over scales and patches
        log_likelihood_w = torch.sum(torch.log(masked_likelihood_w + 1e-8), dim=(1, 2))
        log_likelihood_u = torch.sum(torch.log(masked_likelihood_u + 1e-8), dim=(1, 2))
        
        # Compute posterior
        log_prior = torch.log(self.prior + 1e-8)
        log_prior_complement = torch.log(1 - self.prior + 1e-8)
        
        log_posterior_w = log_likelihood_w + log_prior
        log_posterior_u = log_likelihood_u + log_prior_complement
        
        # Normalize
        log_sum = torch.logsumexp(torch.stack([log_posterior_w, log_posterior_u]), dim=0)
        posterior = torch.exp(log_posterior_w - log_sum)
        
        return posterior


class WatermarkedLikelihoodModel(nn.Module):
    """Likelihood model for watermarked images."""
    
    def __init__(self, config: WatermarkConfig):
        super().__init__()
        self.config = config
        
        # Learnable parameters for each scale
        self.scale_params = nn.ParameterDict({
            str(scale): nn.Parameter(torch.randn(2))  # [mean, std]
            for scale in config.scales
        })
    
    def forward(
        self,
        g_values: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood for watermarked images.
        
        Args:
            g_values: G-values of shape [batch_size, num_scales, num_patches]
            masks: Binary masks of shape [batch_size, num_scales, num_patches]
            
        Returns:
            Likelihoods of shape [batch_size, num_scales, num_patches]
        """
        likelihoods = []
        
        for i, scale in enumerate(self.config.scales):
            scale_g_values = g_values[:, i, :]  # [batch_size, num_patches]
            scale_masks = masks[:, i, :]  # [batch_size, num_patches]
            
            # Get parameters for this scale
            params = self.scale_params[str(scale)]
            mean, log_std = params[0], params[1]
            std = torch.exp(log_std) + 1e-8
            
            # Compute Gaussian likelihood
            likelihood = torch.exp(-0.5 * ((scale_g_values - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
            
            # Apply masks
            likelihood = likelihood * scale_masks + (1 - scale_masks)
            
            likelihoods.append(likelihood)
        
        return torch.stack(likelihoods, dim=1)


class UnwatermarkedLikelihoodModel(nn.Module):
    """Likelihood model for unwatermarked images."""
    
    def __init__(self, config: WatermarkConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        g_values: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood for unwatermarked images.
        
        Assumes uniform distribution over [-1, 1] for unwatermarked images.
        
        Args:
            g_values: G-values of shape [batch_size, num_scales, num_patches]
            masks: Binary masks of shape [batch_size, num_scales, num_patches]
            
        Returns:
            Likelihoods of shape [batch_size, num_scales, num_patches]
        """
        # Uniform distribution over [-1, 1]
        uniform_likelihood = torch.ones_like(g_values) * 0.5
        
        # Apply masks
        likelihood = uniform_likelihood * masks + (1 - masks)
        
        return likelihood


class DetectorTrainer:
    """
    Trainer for image watermark detectors.
    
    Adapted from synthid-text training pipeline.
    """
    
    def __init__(
        self,
        config: Optional[WatermarkConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize detector trainer.
        
        Args:
            config: Watermark configuration
            device: Device for training
        """
        self.config = config or WatermarkConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize detectors
        self.mean_detector = ImageMeanDetector(self.config)
        self.bayesian_detector = ImageBayesianDetector(self.config).to(self.device)
    
    def train_bayesian_detector(
        self,
        training_data: TrainingData,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train Bayesian detector.
        
        Args:
            training_data: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Move data to device
        g_values = training_data.g_values.to(self.device)
        masks = training_data.masks.to(self.device)
        labels = training_data.labels.to(self.device)
        
        # Split data
        indices = torch.randperm(len(g_values))
        split_idx = int(len(g_values) * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_g_values = g_values[train_indices]
        train_masks = masks[train_indices]
        train_labels = labels[train_indices]
        
        val_g_values = g_values[val_indices]
        val_masks = masks[val_indices]
        val_labels = labels[val_indices]
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.bayesian_detector.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.bayesian_detector.train()
            train_loss = 0.0
            
            for i in range(0, len(train_g_values), batch_size):
                batch_g_values = train_g_values[i:i+batch_size]
                batch_masks = train_masks[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.bayesian_detector(batch_g_values, batch_masks)
                loss = criterion(predictions, batch_labels.float())
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.bayesian_detector.eval()
            val_loss = 0.0
            val_predictions = []
            val_labels_list = []
            
            with torch.no_grad():
                for i in range(0, len(val_g_values), batch_size):
                    batch_g_values = val_g_values[i:i+batch_size]
                    batch_masks = val_masks[i:i+batch_size]
                    batch_labels = val_labels[i:i+batch_size]
                    
                    predictions = self.bayesian_detector(batch_g_values, batch_masks)
                    loss = criterion(predictions, batch_labels.float())
                    
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels_list.extend(batch_labels.cpu().numpy())
            
            # Compute metrics
            avg_train_loss = train_loss / (len(train_g_values) // batch_size)
            avg_val_loss = val_loss / (len(val_g_values) // batch_size)
            
            try:
                val_auc = roc_auc_score(val_labels_list, val_predictions)
            except ValueError:
                val_auc = 0.5  # Random performance
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_auc'].append(val_auc)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        return history
    
    def evaluate_detector(
        self,
        test_data: TrainingData,
        detector_type: str = "bayesian",
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate detector performance.
        
        Args:
            test_data: Test data
            detector_type: Type of detector ("bayesian" or "mean")
            threshold: Detection threshold
            
        Returns:
            Evaluation metrics
        """
        g_values = test_data.g_values.to(self.device)
        masks = test_data.masks.to(self.device)
        labels = test_data.labels.to(self.device)
        
        if detector_type == "bayesian":
            self.bayesian_detector.eval()
            with torch.no_grad():
                predictions = self.bayesian_detector(g_values, masks)
            predictions = predictions.cpu().numpy()
        else:  # mean detector
            predictions = self.mean_detector.mean_score(g_values, masks).cpu().numpy()
            # Normalize to [0, 1] range
            predictions = (predictions + 1) / 2
        
        labels_np = labels.cpu().numpy()
        
        # Compute metrics
        predictions_binary = (predictions > threshold).astype(int)
        
        tp = np.sum((predictions_binary == 1) & (labels_np == 1))
        fp = np.sum((predictions_binary == 1) & (labels_np == 0))
        tn = np.sum((predictions_binary == 0) & (labels_np == 0))
        fn = np.sum((predictions_binary == 0) & (labels_np == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc = roc_auc_score(labels_np, predictions)
        except ValueError:
            auc = 0.5
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }


class WatermarkDetector:
    """
    Main watermark detector for diffusion models.
    
    Combines statistical analysis with learned detection for robust watermark detection.
    """
    
    def __init__(
        self,
        watermark_key: Union[str, bytes],
        config: Optional[WatermarkConfig] = None,
        device: Optional[torch.device] = None,
        use_learned_detector: bool = True
    ):
        """
        Initialize watermark detector.
        
        Args:
            watermark_key: Watermark key for detection
            config: Watermark configuration
            device: Device for computations
            use_learned_detector: Whether to use learned detector
        """
        self.watermark_key = validate_watermark_key(watermark_key)
        self.config = config or WatermarkConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_learned_detector = use_learned_detector
        
        # Initialize detectors
        self.mean_detector = ImageMeanDetector(self.config)
        self.bayesian_detector = None
        
        if use_learned_detector:
            self.bayesian_detector = ImageBayesianDetector(self.config).to(self.device)
        
        # Detection statistics
        self.stats = {
            'total_detections': 0,
            'positive_detections': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
    
    def extract_g_values(
        self,
        latent: torch.Tensor,
        model_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract g-values and masks from latent tensor.
        
        Args:
            latent: Latent tensor to analyze
            model_id: Model ID to detect
            
        Returns:
            Tuple of (g_values, masks) where:
            - g_values: shape [num_scales, max_patches]
            - masks: shape [num_scales, max_patches]
        """
        # Get latent shape (handle batch dimension)
        if len(latent.shape) == 4:
            latent_shape = latent.shape[1:]  # Remove batch dimension
            latent_single = latent[0]  # Use first sample
        else:
            latent_shape = latent.shape
            latent_single = latent
        
        # Find maximum number of patches across all scales
        max_patches = 0
        all_patch_coords = {}
        
        for scale in self.config.scales:
            patch_coords_list = compute_patch_coordinates(latent_shape, scale)
            all_patch_coords[scale] = patch_coords_list
            max_patches = max(max_patches, len(patch_coords_list))
        
        g_values_list = []
        masks_list = []
        
        # Extract g-values for each scale
        for scale in self.config.scales:
            patch_coords_list = all_patch_coords[scale]
            
            scale_g_values = []
            scale_masks = []
            
            for patch_coords in patch_coords_list:
                # Pool features
                pooled_features = pool_latent_features(latent_single, patch_coords, scale)
                
                # Compute g-value
                bucket = get_timestep_bucket(50, self.config.temporal_windows)  # Use mid-timestep
                g_value = keyed_hash(
                    self.watermark_key,
                    model_id,
                    scale,
                    patch_coords,
                    bucket,
                    pooled_features
                )
                
                scale_g_values.append(g_value)
                scale_masks.append(1.0)  # All patches are valid
            
            # Pad to max_patches length
            while len(scale_g_values) < max_patches:
                scale_g_values.append(0.0)  # Padding value
                scale_masks.append(0.0)  # Mask out padding
            
            g_values_list.append(torch.tensor(scale_g_values))
            masks_list.append(torch.tensor(scale_masks))
        
        # Stack results
        g_values = torch.stack(g_values_list)  # [num_scales, max_patches]
        masks = torch.stack(masks_list)  # [num_scales, max_patches]
        
        return g_values, masks
    
    def detect_statistical(
        self,
        latent: torch.Tensor,
        model_id: str,
        threshold: float = 0.5
    ) -> DetectionResult:
        """
        Detect watermark using statistical analysis.
        
        Args:
            latent: Latent tensor to analyze
            model_id: Model ID to detect
            threshold: Detection threshold
            
        Returns:
            Detection result
        """
        # Extract g-values
        g_values, masks = self.extract_g_values(latent, model_id)
        
        # Compute mean score
        score = self.mean_detector.mean_score(
            g_values.unsqueeze(0), masks.unsqueeze(0)
        ).item()
        
        # Normalize score to [0, 1]
        probability = (score + 1) / 2
        
        # Determine if watermarked
        is_watermarked = probability > threshold
        confidence = abs(probability - 0.5) * 2
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            probability=probability,
            confidence=confidence,
            model_id=model_id,
            detection_scores={'mean_score': score}
        )
    
    def detect_bayesian(
        self,
        latent: torch.Tensor,
        model_id: str,
        threshold: float = 0.5
    ) -> DetectionResult:
        """
        Detect watermark using Bayesian analysis.
        
        Args:
            latent: Latent tensor to analyze
            model_id: Model ID to detect
            threshold: Detection threshold
            
        Returns:
            Detection result
        """
        if self.bayesian_detector is None:
            raise RuntimeError("Bayesian detector not initialized")
        
        # Extract g-values
        g_values, masks = self.extract_g_values(latent, model_id)
        
        # Move to device
        g_values = g_values.to(self.device)
        masks = masks.to(self.device)
        
        # Compute posterior probability
        self.bayesian_detector.eval()
        with torch.no_grad():
            probability = self.bayesian_detector(
                g_values.unsqueeze(0), masks.unsqueeze(0)
            ).item()
        
        # Determine if watermarked
        is_watermarked = probability > threshold
        confidence = abs(probability - 0.5) * 2
        
        return DetectionResult(
            is_watermarked=is_watermarked,
            probability=probability,
            confidence=confidence,
            model_id=model_id,
            detection_scores={'bayesian_score': probability}
        )
    
    def detect(
        self,
        latent: torch.Tensor,
        model_id: Optional[str] = None,
        threshold: float = 0.5,
        method: str = "combined"
    ) -> DetectionResult:
        """
        Detect watermark using specified method.
        
        Args:
            latent: Latent tensor to analyze
            model_id: Model ID to detect (required for statistical method)
            threshold: Detection threshold
            method: Detection method ("statistical", "bayesian", "combined")
            
        Returns:
            Detection result
        """
        if method == "statistical":
            if model_id is None:
                raise ValueError("model_id required for statistical detection")
            return self.detect_statistical(latent, model_id, threshold)
        
        elif method == "bayesian":
            return self.detect_bayesian(latent, model_id or "unknown", threshold)
        
        elif method == "combined":
            # Use both methods and combine results
            results = []
            
            if model_id is not None:
                stat_result = self.detect_statistical(latent, model_id, threshold)
                results.append(stat_result)
            
            if self.bayesian_detector is not None:
                bayesian_result = self.detect_bayesian(latent, model_id or "unknown", threshold)
                results.append(bayesian_result)
            
            if not results:
                raise ValueError("No detection methods available")
            
            # Combine probabilities (simple average)
            combined_prob = np.mean([r.probability for r in results])
            combined_confidence = np.mean([r.confidence for r in results])
            
            return DetectionResult(
                is_watermarked=combined_prob > threshold,
                probability=combined_prob,
                confidence=combined_confidence,
                model_id=model_id,
                detection_scores={'combined_score': combined_prob}
            )
        
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return self.stats.copy()