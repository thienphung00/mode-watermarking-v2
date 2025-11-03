"""
Mask utilities for spatial and frequency-domain masking of g-fields.
Implements zero-mean masks for non-distortionary watermark embedding.
"""
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np


def generate_zero_mean_mask(
    mask_id: str,
    shape: Tuple[int, int, int],
    mode: str = "frequency",
    **kwargs
) -> np.ndarray:
    """
    Generate mask with E[M ⊙ G_t] = 0 property for non-distortionary embedding.
    
    Args:
        mask_id: Identifier for mask type
        shape: (C, H, W) tensor shape
        mode: "frequency" or "spatial"
        **kwargs: Additional mask parameters
    
    Returns:
        Zero-mean mask tensor [C, H, W]
    """
    C, H, W = shape
    
    if mode == "frequency":
        # Create high-frequency mask via FFT
        mask = np.ones((H, W), dtype=np.float32)
        
        # Zero out low frequencies (preserve_low_freq)
        cutoff_freq = kwargs.get("cutoff_freq", 0.3)
        center_h, center_w = H // 2, W // 2
        cutoff_h = int(H * cutoff_freq)
        cutoff_w = int(W * cutoff_freq)
        
        # Create circular low-freq mask
        y, x = np.ogrid[:H, :W]
        dist_from_center = np.sqrt(
            ((y - center_h) / max(cutoff_h, 1)) ** 2 + 
            ((x - center_w) / max(cutoff_w, 1)) ** 2
        )
        low_freq_mask = dist_from_center <= 1.0
        
        # High-freq mask = 1 - low_freq (but we want zero-mean)
        mask[low_freq_mask] = 0.0
        
        # Normalize to zero mean for non-distortionary property
        mask = mask - np.mean(mask)
        
        # Replicate across channels
        M = np.stack([mask] * C, axis=0)
        
    elif mode == "spatial":
        # Spatial mask (center/edges), normalized to zero mean
        mask_type = kwargs.get("type", "center")
        
        if mask_type == "center":
            y, x = np.ogrid[:H, :W]
            cy, cx = H / 2.0, W / 2.0
            r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            R = np.sqrt((cy) ** 2 + (cx) ** 2)
            mask = 1.0 - (r / max(R, 1e-10))
            mask = np.clip(mask, 0.0, 1.0)
        elif mask_type == "edges":
            y, x = np.ogrid[:H, :W]
            cy, cx = H / 2.0, W / 2.0
            r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            R = np.sqrt((cy) ** 2 + (cx) ** 2)
            mask = r / max(R, 1e-10)
            mask = np.clip(mask, 0.0, 1.0)
        else:
            # Default: uniform
            mask = np.ones((H, W), dtype=np.float32)
        
        # Normalize to zero mean for non-distortionary property
        mask = mask - np.mean(mask)
        
        # Replicate across channels
        M = np.stack([mask] * C, axis=0)
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    
    # Apply strength
    strength = kwargs.get("strength", 0.8)
    M = M * strength
    
    return M.astype(np.float32)


def verify_mask_property(
    M: np.ndarray,
    G_samples: List[np.ndarray],
    tolerance: float = 0.01
) -> bool:
    """
    Verify E[M ⊙ G_t] ≈ 0 over multiple G_t samples.
    
    Args:
        M: Mask tensor [C, H, W]
        G_samples: List of G_t samples
        tolerance: Acceptable deviation from zero
    
    Returns:
        True if property holds
    """
    expectations = []
    for G_t in G_samples:
        expectation = np.mean(M * G_t)
        expectations.append(expectation)
    
    mean_expectation = np.mean(expectations)
    return abs(mean_expectation) < tolerance


def load_mask_from_config(
    mask_cfg: Dict,
    shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Parse nested mask config and generate zero-mean mask.
    
    Args:
        mask_cfg: Mask configuration dictionary
        shape: (C, H, W) tensor shape
    
    Returns:
        Zero-mean mask tensor [C, H, W]
    """
    if not mask_cfg.get("enabled", True):
        # Return zero mask if disabled
        return np.zeros(shape, dtype=np.float32)
    
    mask_type = mask_cfg.get("mask_type", "frequency")
    mask_id = mask_cfg.get("mask_id", "default")
    
    if mask_type == "frequency":
        freq_cfg = mask_cfg.get("frequency_mask", {})
        return generate_zero_mean_mask(
            mask_id=mask_id,
            shape=shape,
            mode="frequency",
            strength=freq_cfg.get("strength", 0.8),
            preserve_low_freq=freq_cfg.get("preserve_low_freq", True),
            cutoff_freq=freq_cfg.get("cutoff_freq", 0.3)
        )
    elif mask_type == "spatial":
        spatial_cfg = mask_cfg.get("spatial_mask", {})
        return generate_zero_mean_mask(
            mask_id=mask_id,
            shape=shape,
            mode="spatial",
            type=spatial_cfg.get("type", "center"),
            strength=spatial_cfg.get("strength", 0.8),
            center_radius=spatial_cfg.get("center_radius", 0.5)
        )
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


# Legacy function for backward compatibility
def generate_mask(mask_id: str, shape: Tuple[int, int, int], mode: str = "spatial") -> np.ndarray:
    """
    Generate a simple mask tensor M[c,h,w] (legacy function).
    Use generate_zero_mean_mask for non-distortionary embedding.
    """
    return generate_zero_mean_mask(mask_id, shape, mode)
