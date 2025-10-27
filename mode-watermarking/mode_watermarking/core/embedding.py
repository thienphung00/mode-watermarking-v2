"""
Watermark embedding module for diffusion models.

This module implements watermark embedding during the diffusion sampling process
using multi-scale spatial tiling and temporal windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass

from ..utils.utils import (
    WatermarkConfig, keyed_hash, create_spatial_mask, pool_latent_features,
    get_timestep_bucket, compute_patch_coordinates, compute_watermark_strength,
    validate_watermark_key
)
import random


class BaseNoiseModifier(nn.Module):
    """
    Base class for noise modification during diffusion sampling.
    
    Provides common functionality for all watermarking techniques.
    """
    
    def __init__(
        self,
        watermark_key: Union[str, bytes],
        model_id: str,
        config: Optional[WatermarkConfig] = None,
        device: Optional[torch.device] = None,
        latent_channels: int = 4,
        default_latent_size: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize base noise modifier.
        
        Args:
            watermark_key: Secret watermark key
            model_id: Unique model identifier
            config: Watermark configuration
            device: Device for computations
            latent_channels: Number of latent channels (typically 4 for SD)
            default_latent_size: Default latent spatial dimensions (H/8, W/8)
        """
        super().__init__()
        
        self.watermark_key = validate_watermark_key(watermark_key)
        self.model_id = model_id
        self.config = config or WatermarkConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_channels = latent_channels
        self.default_latent_size = default_latent_size
        
        # Dynamic patch coordinates cache - will be populated as needed
        self.patch_coords_cache = {}
        
        # Cache for computed g-values
        self.g_value_cache = {}
        
        # Track seen latent dimensions for optimization
        self.seen_latent_shapes = set()
        
        # Statistics for monitoring
        self.stats = {
            'total_modifications': 0,
            'scale_modifications': {scale: 0 for scale in self.config.scales},
            'temporal_modifications': {window: 0 for window in self.config.temporal_windows},
            'latent_shapes_seen': []
        }
    
    def _get_patch_coordinates(self, latent_shape: Tuple[int, int, int], scale: int) -> List[Tuple[int, int]]:
        """Get patch coordinates for a given scale, with caching."""
        cache_key = (latent_shape, scale)
        if cache_key not in self.patch_coords_cache:
            self.patch_coords_cache[cache_key] = compute_patch_coordinates(latent_shape, scale)
        return self.patch_coords_cache[cache_key]
    
    def _validate_latent_dimensions(self, latent: torch.Tensor) -> None:
        """Validate and track latent dimensions."""
        shape = latent.shape
        if shape not in self.seen_latent_shapes:
            self.seen_latent_shapes.add(shape)
            self.stats['latent_shapes_seen'].append(shape)
            
            # Validate channel count
            if len(shape) == 4:  # Batch dimension present
                channels = shape[1]
            else:  # No batch dimension
                channels = shape[0]
            
        if channels != self.latent_channels:
                print(f"Warning: Expected {self.latent_channels} channels, got {channels}")
    
    def _apply_watermark_bias(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int,
        scale: int,
        patch_coords: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Apply watermark bias to noise tensor.
        
        Args:
            noise: Predicted noise tensor
            latent: Current latent tensor
            timestep: Current timestep
            scale: Spatial scale
            patch_coords: Patch coordinates
            
        Returns:
            Modified noise tensor
        """
        # Get temporal bucket
        bucket = get_timestep_bucket(timestep, self.config.temporal_windows)
        
        # Pool features for deterministic hashing
        pooled_features = pool_latent_features(latent, patch_coords, scale)
        
        # Compute g-value
        g_value = keyed_hash(
            self.watermark_key,
            self.model_id,
            scale,
            patch_coords,
            bucket,
            pooled_features
        )
        
        # Compute watermark strength
        strength = compute_watermark_strength(scale, timestep, self.config)
        
        # Create spatial mask
        mask = create_spatial_mask(noise.shape, scale, patch_coords, noise.device)
        
        # Apply bias
        bias = g_value * strength
        modified_noise = noise + bias * mask
        
        # Update statistics
        self.stats['total_modifications'] += 1
        self.stats['scale_modifications'][scale] += 1
        
        return modified_noise
    
    def modify_noise(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Modify noise tensor with watermark.
        
        Args:
            noise: Predicted noise tensor
            latent: Current latent tensor
            timestep: Current timestep
            
        Returns:
            Modified noise tensor
        """
        # Validate dimensions
        self._validate_latent_dimensions(latent)
        
        # Get latent shape (handle batch dimension)
        if len(latent.shape) == 4:
            latent_shape = latent.shape[1:]  # Remove batch dimension
        else:
            latent_shape = latent.shape
        
        modified_noise = noise.clone()
        
        # Apply watermarking for each scale
        for scale in self.config.scales:
            # Get patch coordinates
            patch_coords_list = self._get_patch_coordinates(latent_shape, scale)
            
            # Apply watermarking to each patch
            for patch_coords in patch_coords_list:
                modified_noise = self._apply_watermark_bias(
                    modified_noise, latent, timestep, scale, patch_coords
                )
        
        return modified_noise


class MultiTemporalNoiseModifier(BaseNoiseModifier):
    """
    Multi-temporal watermarking technique.
    
    Embeds watermarks throughout the sampling process with varying strength
    based on temporal windows.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize multi-temporal noise modifier."""
        super().__init__(*args, **kwargs)
        self.technique_name = "multi_temporal"


class LateStageNoiseModifier(BaseNoiseModifier):
    """
    Late-stage watermarking technique.
    
    Embeds watermarks only in the final timesteps when the image is nearly complete.
    """
    
    def __init__(
        self,
        *args,
        late_stage_threshold: int = 20,
        late_stage_strength_multiplier: float = 2.0,
        **kwargs
    ):
        """
        Initialize late-stage noise modifier.
        
        Args:
            late_stage_threshold: Timestep threshold for late-stage embedding
            late_stage_strength_multiplier: Multiplier for watermark strength
        """
        super().__init__(*args, **kwargs)
        self.late_stage_threshold = late_stage_threshold
        self.late_stage_strength_multiplier = late_stage_strength_multiplier
        self.technique_name = "late_stage"
    
    def modify_noise(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Modify noise tensor with late-stage watermarking.
        
        Only applies watermarking in late timesteps.
        """
        if timestep > self.late_stage_threshold:
            return noise  # No watermarking in early timesteps
        
        # Apply watermarking with increased strength
        original_strengths = self.config.spatial_strengths.copy()
        for scale in self.config.spatial_strengths:
            self.config.spatial_strengths[scale] *= self.late_stage_strength_multiplier
        
        modified_noise = super().modify_noise(noise, latent, timestep)
        
        # Restore original strengths
        self.config.spatial_strengths = original_strengths
        
        return modified_noise


class RandomStepNoiseModifier(BaseNoiseModifier):
    """
    Random-step watermarking technique.
    
    Embeds watermarks at random timesteps and patches for additional security.
    """
    
    def __init__(
        self,
        *args,
        embedding_probability: float = 0.3,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize random-step noise modifier.
        
        Args:
            embedding_probability: Probability of embedding at each timestep
            random_seed: Random seed for reproducibility
        """
        super().__init__(*args, **kwargs)
        self.embedding_probability = embedding_probability
        self.random_seed = random_seed
        self.technique_name = "random_step"
        
        if random_seed is not None:
            random.seed(random_seed)
        
    def modify_noise(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Modify noise tensor with random-step watermarking.
        
        Randomly decides whether to apply watermarking at each timestep.
        """
        if random.random() > self.embedding_probability:
            return noise  # Skip watermarking based on probability
        
        # Apply watermarking with random patch selection
        modified_noise = super().modify_noise(noise, latent, timestep)
        
        return modified_noise


class WatermarkEmbedder:
    """
    High-level watermark embedder for diffusion models.
    
    This class provides a simple interface for embedding watermarks during
    diffusion sampling by wrapping different NoiseModifier techniques.
    """
    
    def __init__(
        self,
        watermark_key: Union[str, bytes],
        model_id: str,
        config: Optional[WatermarkConfig] = None,
        device: Optional[torch.device] = None,
        latent_channels: int = 4,
        default_latent_size: Tuple[int, int] = (64, 64),
        embedding_technique: str = "multi_temporal",
        **technique_kwargs
    ):
        """
        Initialize watermark embedder.
        
        Args:
            watermark_key: Secret watermark key
            model_id: Unique model identifier
            config: Watermark configuration
            device: Device for computations
            latent_channels: Number of latent channels (typically 4 for SD)
            default_latent_size: Default latent spatial dimensions (H/8, W/8)
            embedding_technique: Embedding technique ("multi_temporal", "late_stage", "random_step")
            **technique_kwargs: Additional arguments for specific techniques
        """
        # Create appropriate noise modifier based on technique
        if embedding_technique == "multi_temporal":
            self.noise_modifier = MultiTemporalNoiseModifier(
                watermark_key=watermark_key,
                model_id=model_id,
                config=config,
                device=device,
                latent_channels=latent_channels,
                default_latent_size=default_latent_size
            )
        elif embedding_technique == "late_stage":
            self.noise_modifier = LateStageNoiseModifier(
                watermark_key=watermark_key,
                model_id=model_id,
                config=config,
                device=device,
                latent_channels=latent_channels,
                default_latent_size=default_latent_size,
                **technique_kwargs
            )
        elif embedding_technique == "random_step":
            self.noise_modifier = RandomStepNoiseModifier(
                watermark_key=watermark_key,
                model_id=model_id,
                config=config,
                device=device,
                latent_channels=latent_channels,
                default_latent_size=default_latent_size,
                **technique_kwargs
            )
        else:
            raise ValueError(f"Unknown embedding technique: {embedding_technique}. "
                            f"Supported techniques: multi_temporal, late_stage, random_step")
        
        self.watermark_key = self.noise_modifier.watermark_key
        self.model_id = self.noise_modifier.model_id
        self.config = self.noise_modifier.config
        self.embedding_technique = embedding_technique
        
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
        return self.noise_modifier.modify_noise(noise, latent, timestep)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return self.noise_modifier.stats.copy()