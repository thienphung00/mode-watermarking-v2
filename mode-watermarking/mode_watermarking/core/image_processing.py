"""
Image processing module for watermarking in diffusion models.

This module provides functionality similar to logits_processing.py from synthid-text,
but adapted for image generation in diffusion models.
"""

import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from dataclasses import dataclass
import numpy as np

from ..utils.utils import (
    WatermarkConfig, keyed_hash, create_spatial_mask, pool_latent_features,
    get_timestep_bucket, compute_patch_coordinates, compute_watermark_strength,
    validate_watermark_key
)


@dataclass
class ImageWatermarkState:
    """State for image watermarking during diffusion sampling."""
    
    # Current state
    timestep: int
    latent_shape: Tuple[int, int, int]
    
    # Watermarking parameters
    watermark_key: bytes
    model_id: str
    config: WatermarkConfig
    
    # Processing state
    processed_patches: Dict[Tuple[int, int], bool] = None
    noise_scores: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize state."""
        if self.processed_patches is None:
            self.processed_patches = {}
        if self.noise_scores is None:
            self.noise_scores = {}


def update_noise_scores(
    state: ImageWatermarkState,
    noise: torch.Tensor,
    latent: torch.Tensor,
    scale: int,
    patch_coords: Tuple[int, int]
) -> None:
    """
    Update noise scores for a specific patch.
    
    Args:
        state: Current watermarking state
        noise: Noise tensor
        latent: Latent tensor
        scale: Spatial scale
        patch_coords: Patch coordinates
    """
    # Get temporal bucket
    bucket = get_timestep_bucket(state.timestep, state.config.temporal_windows)
    
    # Pool features
    pooled_features = pool_latent_features(latent, patch_coords, scale)
    
    # Compute g-value
    g_value = keyed_hash(
        state.watermark_key,
        state.model_id,
        scale,
        patch_coords,
        bucket,
        pooled_features
    )
    
    # Store score
    score_key = f"scale_{scale}_patch_{patch_coords[0]}_{patch_coords[1]}"
    state.noise_scores[score_key] = g_value
    
    # Mark patch as processed
    state.processed_patches[patch_coords] = True


def update_noise_scores_adaptive(
    state: ImageWatermarkState,
    noise: torch.Tensor,
    latent: torch.Tensor,
    adaptive_threshold: float = 0.1
) -> None:
    """
    Update noise scores adaptively based on content complexity.
    
    Args:
        state: Current watermarking state
        noise: Noise tensor
        latent: Latent tensor
        adaptive_threshold: Threshold for adaptive processing
    """
    # Compute content complexity (variance)
    content_variance = torch.var(latent).item()
    
    # Only process patches in high-complexity regions
    if content_variance > adaptive_threshold:
        for scale in state.config.scales:
            patch_coords_list = compute_patch_coordinates(state.latent_shape, scale)
            
            for patch_coords in patch_coords_list:
                # Check if patch has high complexity
                patch_data = latent[:, 
                    patch_coords[0] * scale:(patch_coords[0] + 1) * scale,
                    patch_coords[1] * scale:(patch_coords[1] + 1) * scale
                ]
                
                patch_variance = torch.var(patch_data).item()
                
                if patch_variance > adaptive_threshold:
                    update_noise_scores(state, noise, latent, scale, patch_coords)


class ImageWatermarkProcessor:
    """
    Processor for image watermarking in diffusion models.
    
    Handles the core watermarking logic for image generation.
    """
    
    def __init__(
        self,
        watermark_key: Union[str, bytes],
        model_id: str,
        config: Optional[WatermarkConfig] = None,
        device: Optional[torch.device] = None,
        embedding_technique: str = "multi_temporal"
    ):
        """
        Initialize image watermark processor.
        
        Args:
            watermark_key: Secret watermark key
            model_id: Model identifier
            config: Watermark configuration
            device: Device for computations
            embedding_technique: Embedding technique
        """
        self.watermark_key = validate_watermark_key(watermark_key)
        self.model_id = model_id
        self.config = config or WatermarkConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_technique = embedding_technique
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'patches_processed': 0,
            'scales_used': {scale: 0 for scale in self.config.scales}
        }
    
    def embed_watermark(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Embed watermark into noise tensor.
        
        Args:
            noise: Predicted noise tensor
            latent: Current latent tensor
            timestep: Current timestep
            
        Returns:
            Watermarked noise tensor
        """
        # Create watermarking state
        latent_shape = latent.shape[1:] if len(latent.shape) == 4 else latent.shape
        state = ImageWatermarkState(
            timestep=timestep,
            latent_shape=latent_shape,
            watermark_key=self.watermark_key,
            model_id=self.model_id,
            config=self.config
        )
        
        # Apply watermarking based on technique
        if self.embedding_technique == "multi_temporal":
            return self._embed_multi_temporal(noise, latent, state)
        elif self.embedding_technique == "late_stage":
            return self._embed_late_stage(noise, latent, state)
        elif self.embedding_technique == "random_step":
            return self._embed_random_step(noise, latent, state)
        else:
            raise ValueError(f"Unknown embedding technique: {self.embedding_technique}")
    
    def _embed_multi_temporal(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        state: ImageWatermarkState
    ) -> torch.Tensor:
        """Embed watermark using multi-temporal technique."""
        modified_noise = noise.clone()
        
        for scale in self.config.scales:
            patch_coords_list = compute_patch_coordinates(state.latent_shape, scale)
            
            for patch_coords in patch_coords_list:
                # Update scores
                update_noise_scores(state, modified_noise, latent, scale, patch_coords)
                
                # Apply watermark bias
                modified_noise = self._apply_patch_bias(
                    modified_noise, latent, state, scale, patch_coords
                )
                
                # Update statistics
                self.stats['patches_processed'] += 1
                self.stats['scales_used'][scale] += 1
        
        self.stats['total_processed'] += 1
        return modified_noise
    
    def _embed_late_stage(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        state: ImageWatermarkState
    ) -> torch.Tensor:
        """Embed watermark using late-stage technique."""
        # Only embed in late timesteps
        if state.timestep > 20:
            return noise
        
        # Use increased strength for late-stage embedding
        original_strengths = self.config.spatial_strengths.copy()
        for scale in self.config.spatial_strengths:
            self.config.spatial_strengths[scale] *= 2.0
        
        result = self._embed_multi_temporal(noise, latent, state)
        
        # Restore original strengths
        self.config.spatial_strengths = original_strengths
        
        return result
    
    def _embed_random_step(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        state: ImageWatermarkState
    ) -> torch.Tensor:
        """Embed watermark using random-step technique."""
        import random
        
        # Randomly decide whether to embed
        if random.random() > 0.3:  # 30% probability
            return noise
        
        return self._embed_multi_temporal(noise, latent, state)
    
    def _apply_patch_bias(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        state: ImageWatermarkState,
        scale: int,
        patch_coords: Tuple[int, int]
    ) -> torch.Tensor:
        """Apply watermark bias to a specific patch."""
        # Get g-value from state
        score_key = f"scale_{scale}_patch_{patch_coords[0]}_{patch_coords[1]}"
        g_value = state.noise_scores.get(score_key, 0.0)
        
        # Compute strength
        strength = compute_watermark_strength(scale, state.timestep, self.config)
        
        # Create mask
        mask = create_spatial_mask(noise.shape, scale, patch_coords, noise.device)
        
        # Apply bias
        bias = g_value * strength
        return noise + bias * mask
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


class ImageWatermarkMixin:
    """
    Mixin class for adding image watermarking capabilities.
    
    Provides a simple interface for integrating watermarking into image generation.
    """
    
    def __init__(
        self,
        watermark_key: Union[str, bytes],
        model_id: str,
        config: Optional[WatermarkConfig] = None,
        embedding_technique: str = "multi_temporal"
    ):
        """
        Initialize image watermark mixin.
        
        Args:
            watermark_key: Secret watermark key
            model_id: Model identifier
            config: Watermark configuration
            embedding_technique: Embedding technique
        """
        self.watermark_processor = ImageWatermarkProcessor(
            watermark_key=watermark_key,
            model_id=model_id,
            config=config,
            embedding_technique=embedding_technique
        )
        self.is_watermarking_enabled = True
    
    def enable_watermarking(self) -> None:
        """Enable watermarking."""
        self.is_watermarking_enabled = True
    
    def disable_watermarking(self) -> None:
        """Disable watermarking."""
        self.is_watermarking_enabled = False
    
    def process_noise(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Process noise tensor with watermarking.
        
        Args:
            noise: Noise tensor
            latent: Latent tensor
            timestep: Current timestep
            
        Returns:
            Processed noise tensor
        """
        if not self.is_watermarking_enabled:
            return noise
        
        return self.watermark_processor.embed_watermark(noise, latent, timestep)
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """Get watermarking statistics."""
        return self.watermark_processor.get_stats()


# Default configuration
DEFAULT_IMAGE_WATERMARKING_CONFIG = WatermarkConfig()