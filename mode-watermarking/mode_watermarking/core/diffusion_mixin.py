"""
Diffusion model mixin for watermarking integration.

This module provides mixin classes for integrating watermarking into diffusion models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Mapping
from dataclasses import dataclass

from .image_processing import ImageWatermarkProcessor, ImageWatermarkMixin
from ..utils.utils import WatermarkConfig, validate_watermark_key


@dataclass
class DiffusionWatermarkConfig:
    """Configuration for diffusion model watermarking."""
    
    # Watermarking parameters
    watermark_key: Union[str, bytes]
    model_id: str
    embedding_technique: str = "multi_temporal"
    
    # Model-specific parameters
    latent_channels: int = 4
    default_latent_size: Tuple[int, int] = (64, 64)
    
    # Watermarking configuration
    watermark_config: Optional[WatermarkConfig] = None
    
    # Integration parameters
    apply_to_noise: bool = True
    apply_to_latent: bool = False
    hook_timesteps: Optional[List[int]] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.watermark_key = validate_watermark_key(self.watermark_key)
        if self.watermark_config is None:
            self.watermark_config = WatermarkConfig()


class DiffusionWatermarkMixin:
    """
    Mixin class for adding watermarking capabilities to diffusion models.
    
    This mixin provides methods for integrating watermarking into the diffusion
    sampling process without modifying the core model architecture.
    """
    
    def __init__(self, watermark_config: DiffusionWatermarkConfig):
        """
        Initialize diffusion watermark mixin.
        
        Args:
            watermark_config: Watermarking configuration
        """
        self.watermark_config = watermark_config
        self.watermark_processor = ImageWatermarkProcessor(
            watermark_key=watermark_config.watermark_key,
            model_id=watermark_config.model_id,
            config=watermark_config.watermark_config,
            embedding_technique=watermark_config.embedding_technique
        )
        self.is_watermarking_enabled = True
    
    def enable_watermarking(self) -> None:
        """Enable watermarking during sampling."""
        self.is_watermarking_enabled = True
    
    def disable_watermarking(self) -> None:
        """Disable watermarking during sampling."""
        self.is_watermarking_enabled = False
    
    def apply_watermark_to_noise(
        self,
        noise: torch.Tensor,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Apply watermark to predicted noise.
        
        Args:
            noise: Predicted noise tensor
            latent: Current latent tensor
            timestep: Current timestep
            
        Returns:
            Watermarked noise tensor
        """
        if not self.is_watermarking_enabled:
            return noise
        
        # Check if we should apply watermarking at this timestep
        if self.watermark_config.hook_timesteps is not None:
            if timestep not in self.watermark_config.hook_timesteps:
                return noise
        
        return self.watermark_processor.embed_watermark(noise, latent, timestep)
    
    def apply_watermark_to_latent(
        self,
        latent: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Apply watermark directly to latent tensor.
        
        Args:
            latent: Current latent tensor
            timestep: Current timestep
            
        Returns:
            Watermarked latent tensor
        """
        if not self.is_watermarking_enabled:
            return latent
        
        # Check if we should apply watermarking at this timestep
        if self.watermark_config.hook_timesteps is not None:
            if timestep not in self.watermark_config.hook_timesteps:
                return latent
        
        # Create dummy noise for watermarking
        dummy_noise = torch.zeros_like(latent)
        watermarked_noise = self.watermark_processor.embed_watermark(
            dummy_noise, latent, timestep
        )
        
        # Apply watermark as additive bias to latent
        return latent + watermarked_noise - dummy_noise
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """Get watermarking statistics."""
        return self.watermark_processor.get_stats()


class StableDiffusionWatermarkMixin(DiffusionWatermarkMixin):
    """
    Specialized mixin for Stable Diffusion models.
    
    Provides Stable Diffusion-specific watermarking integration.
    """
    
    def __init__(self, watermark_config: DiffusionWatermarkConfig):
        """
        Initialize Stable Diffusion watermark mixin.
        
        Args:
            watermark_config: Watermarking configuration
        """
        super().__init__(watermark_config)
        
        # Stable Diffusion specific parameters
        self.latent_channels = 4
        self.default_latent_size = (64, 64)  # 512x512 -> 64x64 latent
    
    def hook_unet_forward(
        self,
        unet: nn.Module,
        timestep: int,
        sample: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Hook into UNet forward pass for watermarking.
        
        Args:
            unet: UNet model
            timestep: Current timestep
            sample: Input sample tensor
            encoder_hidden_states: Encoder hidden states
            **kwargs: Additional arguments
            
        Returns:
            Modified UNet output
        """
        # Get original UNet output
        original_output = unet(sample, timestep, encoder_hidden_states, **kwargs)
        
        # Apply watermarking if enabled
        if self.is_watermarking_enabled and self.watermark_config.apply_to_noise:
            if isinstance(original_output, tuple):
                # Handle tuple output (some UNet variants return tuples)
                noise = original_output[0]
                watermarked_noise = self.apply_watermark_to_noise(
                    noise, sample, timestep
                )
                return (watermarked_noise,) + original_output[1:]
            else:
                # Single tensor output
                return self.apply_watermark_to_noise(
                    original_output, sample, timestep
                )
        
        return original_output


def create_watermarked_diffusion_model(
    base_model: nn.Module,
    watermark_config: DiffusionWatermarkConfig
) -> nn.Module:
    """
    Create a watermarked diffusion model by wrapping the base model.
    
    Args:
        base_model: Base diffusion model
        watermark_config: Watermarking configuration
        
    Returns:
        Watermarked diffusion model
    """
    class WatermarkedModel(base_model.__class__, DiffusionWatermarkMixin):
        def __init__(self, base_model, watermark_config):
            # Initialize base model
            super().__init__()
            self.__dict__.update(base_model.__dict__)
            
            # Initialize watermark mixin
            DiffusionWatermarkMixin.__init__(self, watermark_config)
    
    return WatermarkedModel(base_model, watermark_config)


def create_watermarked_stable_diffusion_model(
    base_model: nn.Module,
    watermark_config: DiffusionWatermarkConfig
) -> nn.Module:
    """
    Create a watermarked Stable Diffusion model.
    
    Args:
        base_model: Base Stable Diffusion model
        watermark_config: Watermarking configuration
        
    Returns:
        Watermarked Stable Diffusion model
    """
    class WatermarkedStableDiffusionModel(base_model.__class__, StableDiffusionWatermarkMixin):
        def __init__(self, base_model, watermark_config):
            # Initialize base model
            super().__init__()
            self.__dict__.update(base_model.__dict__)
            
            # Initialize Stable Diffusion watermark mixin
            StableDiffusionWatermarkMixin.__init__(self, watermark_config)
    
    return WatermarkedStableDiffusionModel(base_model, watermark_config)


# Default configuration
DEFAULT_DIFFUSION_WATERMARKING_CONFIG = DiffusionWatermarkConfig(
    watermark_key="default_watermark_key_change_me",
    model_id="default_model",
    embedding_technique="multi_temporal",
    latent_channels=4,
    default_latent_size=(64, 64),
    apply_to_noise=True,
    apply_to_latent=False
)