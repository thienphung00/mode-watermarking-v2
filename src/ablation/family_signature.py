"""
Detector family signature computation for grouping configs.

This module extracts the "detector geometry" fields from watermark configs
that determine which detector family a config belongs to. Configs with the
same detector geometry share the same g-field structure and can share
trained likelihood models.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Any

from ..core.config import AppConfig, WatermarkedConfig


def compute_family_signature(config: AppConfig) -> Dict[str, Any]:
    """
    Extract detector family signature from config.
    
    The signature includes only fields that affect detector geometry:
    - mapping_mode: binary vs continuous
    - g_field.geometry: domain, frequency_mode, cutoffs
    - g_field.grid_shape: shape of g-field
    - g_field.cutoff: frequency cutoffs
    - mask.strategy: mask mode and parameters
    - mask.grid_shape: mask geometry
    - latent_type: z0 vs zT (affects inversion)
    
    Args:
        config: AppConfig instance
        
    Returns:
        Dictionary with signature fields (deterministic, JSON-serializable)
        
    Raises:
        ValueError: If config is not watermarked
    """
    if not isinstance(config.watermark, WatermarkedConfig):
        raise ValueError("Config must be watermarked to compute family signature")
    
    wm_config = config.watermark
    g_field = wm_config.algorithm_params.g_field
    mask = wm_config.algorithm_params.mask
    
    # Extract signature fields
    signature = {
        "mapping_mode": g_field.mapping_mode,
        "g_field.geometry": {
            "domain": g_field.domain,
            "frequency_mode": g_field.frequency_mode,
            "low_freq_cutoff": g_field.low_freq_cutoff,
            "high_freq_cutoff": g_field.high_freq_cutoff,
        },
        "g_field.grid_shape": list(g_field.shape),
        "g_field.cutoff": {
            "low": g_field.low_freq_cutoff,
            "high": g_field.high_freq_cutoff,
        },
        "mask.strategy": {
            "mode": mask.mode,
            "strength": mask.strength,
            "band": mask.band,
            "cutoff_freq": mask.cutoff_freq,
            "bandwidth_fraction": mask.bandwidth_fraction,
        },
        "mask.grid_shape": None,  # Mask uses same shape as g-field
        "latent_type": "zT",  # Detection uses zT (inverted latents)
    }
    
    # Add normalization settings (affect g-field generation)
    normalize_dict = g_field.normalize if isinstance(g_field.normalize, dict) else {}
    normalize_zero_mean = (
        g_field.normalize_zero_mean
        if g_field.normalize_zero_mean is not None
        else normalize_dict.get("zero_mean_per_timestep", True)
        or normalize_dict.get("zero_mean_per_channel", True)
    )
    normalize_unit_variance = (
        g_field.normalize_unit_variance
        if g_field.normalize_unit_variance is not None
        else normalize_dict.get("unit_variance", False)
    )
    
    signature["g_field.normalize"] = {
        "zero_mean": normalize_zero_mean,
        "unit_variance": normalize_unit_variance,
    }
    
    return signature


def compute_family_id(signature: Dict[str, Any]) -> str:
    """
    Compute deterministic family ID from signature.
    
    Args:
        signature: Family signature dictionary
        
    Returns:
        Family ID (12-character hex string)
    """
    # Sort keys for deterministic hashing
    signature_str = json.dumps(signature, sort_keys=True)
    family_hash = hashlib.sha256(signature_str.encode()).hexdigest()
    return family_hash[:12]

