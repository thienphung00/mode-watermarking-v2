"""
Utility functions for image watermarking in diffusion models.

This module provides core utilities for watermarking operations including
hashing, encoding, spatial operations, and common helper functions.

The hashing implementation uses a Linear Congruential Generator (LCG) approach
adapted from synthid-text for efficient and deterministic watermark generation.
"""

import hashlib
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class WatermarkConfig:
    """Configuration for watermarking parameters."""
    # Spatial scales (latent coordinate units)
    scales: Tuple[int, ...] = (64, 32, 16)
    
    # Temporal windows (timestep ranges)
    temporal_windows: Tuple[Tuple[int, int], ...] = ((90, 70), (69, 40), (39, 10))
    
    # Spatial strengths by scale
    spatial_strengths: Dict[int, float] = None
    
    # Temporal weights by window
    temporal_weights: Dict[Tuple[int, int], float] = None
    
    # Hash parameters
    hash_algorithm: str = "sha256"
    
    # Smoothing parameters
    smoothing_kernel_size: int = 3
    smoothing_sigma: float = 1.0
    
    def __post_init__(self):
        if self.spatial_strengths is None:
            self.spatial_strengths = {64: 0.06, 32: 0.04, 16: 0.02}
        
        if self.temporal_weights is None:
            self.temporal_weights = {
                (90, 70): 1.0,  # early window
                (69, 40): 0.6,  # mid window
                (39, 10): 0.2   # late window
            }


def accumulate_hash(
    current_hash: torch.LongTensor,
    data: torch.LongTensor,
    multiplier: int = 6364136223846793005,
    increment: int = 1,
) -> torch.LongTensor:
    """
    Accumulate hash of data on current hash using LCG approach.
    
    Adapted from synthid-text for image watermarking. Uses linear congruential 
    generator (LCG) with newlib/musl parameters for efficient iterative hashing.
    
    This function has the property:
    f(x, data[T]) = f(f(x, data[:T - 1]), data[T])
    
    Args:
        current_hash: Current hash tensor (shape,)
        data: Data tensor to accumulate (shape, tensor_len)
        multiplier: LCG multiplier (default: newlib/musl parameter)
        increment: LCG increment
        
    Returns:
        Updated hash tensor (shape,)
    """
    for i in range(data.shape[-1]):
        current_hash = torch.add(current_hash, data[..., i])
        current_hash = torch.mul(current_hash, multiplier)
        current_hash = torch.add(current_hash, increment)
    return current_hash


def keyed_hash(
    watermark_key: Union[str, bytes],
    model_id: str,
    scale_id: int,
    patch_coords: Tuple[int, int],
    timestep_bucket: int,
    pooled_features: Optional[torch.Tensor] = None
) -> float:
    """
    Compute deterministic g-value using SHA256 keyed hashing with LCG accumulation.
    
    Args:
        watermark_key: Secret watermark key
        model_id: Model identifier
        scale_id: Spatial scale identifier
        patch_coords: Patch coordinates (row, col)
        timestep_bucket: Temporal bucket identifier
        pooled_features: Optional pooled latent features
        
    Returns:
        Deterministic g-value in [-1, 1]
    """
    # Normalize key to bytes
    if isinstance(watermark_key, str):
        key_bytes = watermark_key.encode('utf-8')
    else:
        key_bytes = watermark_key
    
    # Create input data for LCG accumulation
    input_data = []
    
    # Add model_id as bytes
    model_bytes = model_id.encode('utf-8')
    input_data.extend([int(b) for b in model_bytes])
    
    # Add scale_id
    input_data.append(scale_id)
    
    # Add patch coordinates
    input_data.extend(patch_coords)
    
    # Add timestep bucket
    input_data.append(timestep_bucket)
    
    # Add pooled features if available
    if pooled_features is not None:
        # Convert tensor to integers (scale to avoid overflow)
        features = pooled_features.flatten()[:8]  # Use first 8 values
        # Scale features to integer range [0, 1000000] to avoid overflow
        features_scaled = (features * 1000000).long().clamp(0, 1000000)
        input_data.extend(features_scaled.tolist())
    
    # Convert to tensor for LCG processing
    data_tensor = torch.tensor(input_data, dtype=torch.long)
    
    # Initialize hash with key-derived seed
    # Use first 4 bytes of key hash as initial seed to avoid overflow
    key_hash = hashlib.sha256(key_bytes).digest()
    initial_seed = int.from_bytes(key_hash[:4], byteorder='big')
    current_hash = torch.tensor(initial_seed, dtype=torch.long)
    
    # Accumulate hash using LCG
    # Reshape data for accumulate_hash (expects shape with last dim as sequence)
    data_reshaped = data_tensor.unsqueeze(0)  # Add batch dimension
    current_hash = accumulate_hash(current_hash.unsqueeze(0), data_reshaped)
    current_hash = current_hash.squeeze(0)  # Remove batch dimension
    
    # Convert to float in [-1, 1] range
    # Use modulo to ensure positive range, then normalize
    hash_value = current_hash.item() % (2**32)
    g_value = (hash_value / (2**32 - 1)) * 2 - 1
    
    return g_value


def validate_watermark_key(key: Union[str, bytes]) -> bytes:
    """
    Validate and normalize a watermark key.
    
    Args:
        key: Key as string or bytes
        
    Returns:
        Normalized key bytes
        
    Raises:
        ValueError: If key is invalid
    """
    if isinstance(key, str):
        # Try to decode as hex first, then as UTF-8
        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            key_bytes = key.encode('utf-8')
    elif isinstance(key, bytes):
        key_bytes = key
    else:
        raise ValueError("Key must be string or bytes")
    
    if len(key_bytes) < 16:
        raise ValueError("Key must be at least 16 bytes")
    
    return key_bytes


def create_spatial_mask(
    latent_shape: Tuple[int, int, int],
    scale: int,
    patch_coords: Tuple[int, int],
    device: torch.device
) -> torch.Tensor:
    """
    Create spatial mask for watermark application.
    
    Args:
        latent_shape: Latent tensor shape (C, H, W)
        scale: Spatial scale
        patch_coords: Patch coordinates (row, col)
        device: Device for tensor creation
        
    Returns:
        Spatial mask tensor
    """
    _, height, width = latent_shape
    mask = torch.zeros(latent_shape, device=device)
    
    # Calculate patch boundaries
    patch_size = scale
    row_start = patch_coords[0] * patch_size
    col_start = patch_coords[1] * patch_size
    
    # Ensure boundaries are within latent dimensions
    row_end = min(row_start + patch_size, height)
    col_end = min(col_start + patch_size, width)
    
    # Create mask
    mask[:, row_start:row_end, col_start:col_end] = 1.0
    
    return mask


def pool_latent_features(
    latent: torch.Tensor,
    patch_coords: Tuple[int, int],
    scale: int,
    pool_type: str = "mean"
) -> torch.Tensor:
    """
    Pool features from a latent patch.
    
    Args:
        latent: Latent tensor (C, H, W)
        patch_coords: Patch coordinates (row, col)
        scale: Spatial scale
        pool_type: Pooling type ("mean", "max", "adaptive")
        
    Returns:
        Pooled features tensor
    """
    _, height, width = latent.shape
    
    # Calculate patch boundaries
    patch_size = scale
    row_start = patch_coords[0] * patch_size
    col_start = patch_coords[1] * patch_size
    
    # Ensure boundaries are within latent dimensions
    row_end = min(row_start + patch_size, height)
    col_end = min(col_start + patch_size, width)
    
    # Extract patch
    patch = latent[:, row_start:row_end, col_start:col_end]
    
    # Pool features
    if pool_type == "mean":
        pooled = torch.mean(patch, dim=(1, 2))
    elif pool_type == "max":
        pooled = torch.max(patch.view(patch.size(0), -1), dim=1)[0]
    elif pool_type == "adaptive":
        # Adaptive pooling to fixed size
        pooled = F.adaptive_avg_pool2d(patch.unsqueeze(0), (1, 1)).squeeze(0).squeeze(-1).squeeze(-1)
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")
    
    return pooled


def get_timestep_bucket(timestep: int, temporal_windows: Tuple[Tuple[int, int], ...]) -> int:
    """
    Get temporal bucket for a timestep.
    
    Args:
        timestep: Current timestep
        temporal_windows: Temporal window definitions
        
    Returns:
        Bucket index
    """
    for i, (start, end) in enumerate(temporal_windows):
        if start >= timestep >= end:
            return i
    
    # Default to last bucket if not found
    return len(temporal_windows) - 1


def compute_patch_coordinates(
    latent_shape: Tuple[int, int, int],
    scale: int
) -> List[Tuple[int, int]]:
    """
    Compute patch coordinates for a given scale.
    
    Args:
        latent_shape: Latent tensor shape (C, H, W)
        scale: Spatial scale
        device: Device for tensor creation
        
    Returns:
        List of patch coordinates
    """
    _, height, width = latent_shape
    
    # Calculate number of patches
    patches_h = math.ceil(height / scale)
    patches_w = math.ceil(width / scale)
    
    # Generate patch coordinates
    coordinates = []
    for row in range(patches_h):
        for col in range(patches_w):
            coordinates.append((row, col))
    
    return coordinates


def compute_watermark_strength(
    scale: int,
    timestep: int,
    config: WatermarkConfig
) -> float:
    """
    Compute watermark strength for a given scale and timestep.
    
    Args:
        scale: Spatial scale
        timestep: Current timestep
        config: Watermark configuration
        
    Returns:
        Watermark strength
    """
    # Get spatial strength
    spatial_strength = config.spatial_strengths.get(scale, 0.01)
    
    # Get temporal weight
    bucket = get_timestep_bucket(timestep, config.temporal_windows)
    temporal_window = config.temporal_windows[bucket]
    temporal_weight = config.temporal_weights.get(temporal_window, 0.5)
    
    return spatial_strength * temporal_weight


def apply_spatial_smoothing(
    tensor: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to tensor.
    
    Args:
        tensor: Input tensor
        kernel_size: Kernel size
        sigma: Gaussian sigma
        
    Returns:
        Smoothed tensor
    """
    if kernel_size <= 1:
        return tensor
    
    # Create Gaussian kernel
    kernel = torch.zeros(kernel_size, kernel_size, device=tensor.device)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = math.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
    
    kernel = kernel / kernel.sum()
    
    # Apply convolution
    smoothed = F.conv2d(
        tensor.unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size // 2
    ).squeeze(0)
    
    return smoothed