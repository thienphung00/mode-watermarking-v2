"""
Latent caching for ablation experiments.

Provides deterministic caching of inverted latents to avoid recomputation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ..detection.inversion import DDIMInverter

logger = logging.getLogger(__name__)


class LatentCache:
    """
    Deterministic cache for inverted latents.
    
    Cache keys are based on:
    - Image ID (from filename)
    - Number of inversion steps
    - Model version (hash of model_id)
    """
    
    def __init__(
        self,
        cache_dir: Path,
        model_id: str,
        num_inversion_steps: int = 25,
    ):
        """
        Initialize latent cache.
        
        Args:
            cache_dir: Directory to store cached latents
            model_id: Model identifier (for cache key)
            num_inversion_steps: Number of inversion steps (for cache key)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.num_inversion_steps = num_inversion_steps
        
        # Compute model hash for cache key
        model_hash = hashlib.md5(model_id.encode()).hexdigest()[:8]
        self.model_hash = model_hash
    
    def get_cache_key(self, image_path: Path) -> str:
        """
        Compute deterministic cache key for image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Cache key string
        """
        image_id = image_path.stem
        cache_key = f"{image_id}_steps{self.num_inversion_steps}_model{self.model_hash}"
        return cache_key
    
    def get_cache_path(self, image_path: Path) -> Path:
        """
        Get cache file path for image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Path to cache file
        """
        cache_key = self.get_cache_key(image_path)
        return self.cache_dir / f"{cache_key}.pt"
    
    def get(
        self,
        image_path: Path,
        inverter: DDIMInverter,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Get inverted latent (from cache or compute).
        
        Args:
            image_path: Path to image
            inverter: DDIM inverter instance
            device: Device to run on
            
        Returns:
            Inverted latent tensor [1, 4, 64, 64]
        """
        cache_path = self.get_cache_path(image_path)
        
        # Try cache
        if cache_path.exists():
            try:
                latent_T = torch.load(cache_path, map_location=device)
                # Validate cache integrity
                if latent_T.shape[0] == 1 and latent_T.shape[1] == 4 and len(latent_T.shape) == 4:
                    logger.debug(f"Cache hit: {cache_path}")
                    return latent_T
                else:
                    logger.warning(f"Invalid cache shape {latent_T.shape}, recomputing")
            except Exception as e:
                logger.warning(f"Failed to load cached latent {cache_path}: {e}")
        
        # Cache miss: compute
        logger.debug(f"Cache miss: {cache_path}")
        image = Image.open(image_path).convert("RGB")
        latent_T = inverter.invert(
            image,
            num_inference_steps=self.num_inversion_steps,
            prompt="",
            guidance_scale=1.0,
        )  # [1, 4, 64, 64]
        
        # Save to cache
        try:
            torch.save(latent_T, cache_path)
            logger.debug(f"Cached latent: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache latent {cache_path}: {e}")
        
        return latent_T

