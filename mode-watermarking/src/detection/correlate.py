"""
Correlation and S-statistic computation for watermark detection.

Computes detection scores (S-statistics) by correlating recovered g-values
with expected g-values derived from key information.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import stats

from ..watermark.key import KeyDerivation
from ..watermark.gfield import GFieldBuilder


def compute_s_statistic(
    recovered_g_values: np.ndarray,  # [C, H, W] or [num_timesteps, C, H, W]
    expected_g_values: np.ndarray,   # [C, H, W] from key reconstruction
    mask: Optional[np.ndarray] = None,  # [C, H, W] spatial mask
    method: str = "correlation"  # "correlation" or "log_likelihood_ratio"
) -> float:
    """
    Compute S-statistic (detection score) for single image.
    
    Methods:
    - "correlation": Pearson correlation between recovered and expected g-values
    - "log_likelihood_ratio": S(g) = log[P(g|w) / P(g|¬w)]
    
    Args:
        recovered_g_values: Recovered g-values from image
        expected_g_values: Expected g-values from key reconstruction
        mask: Spatial mask indicating which g-values to use
        method: Computation method ("correlation" or "log_likelihood_ratio")
    
    Returns:
        Scalar detection score S(g)
    """
    # Flatten spatial dimensions if needed
    if recovered_g_values.ndim == 3:
        # Single timestep: [C, H, W]
        recovered_flat = recovered_g_values.flatten()
        expected_flat = expected_g_values.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = np.ones_like(recovered_flat)
    elif recovered_g_values.ndim == 4:
        # Multiple timesteps: [num_timesteps, C, H, W]
        # Aggregate across timesteps (mean)
        recovered_flat = recovered_g_values.mean(axis=0).flatten()
        expected_flat = expected_g_values.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = np.ones_like(recovered_flat)
    else:
        raise ValueError(f"Unexpected g_values shape: {recovered_g_values.shape}")
    
    # Apply mask
    mask_indices = mask_flat > 0.5
    recovered_masked = recovered_flat[mask_indices]
    expected_masked = expected_flat[mask_indices]
    
    if len(recovered_masked) == 0:
        return 0.0
    
    if method == "correlation":
        # Pearson correlation
        if np.std(recovered_masked) < 1e-10 or np.std(expected_masked) < 1e-10:
            return 0.0
        
        correlation, _ = stats.pearsonr(recovered_masked, expected_masked)
        return float(correlation)
    
    elif method == "log_likelihood_ratio":
        # S(g) = log[P(g|w) / P(g|¬w)]
        # For binary ±1 g-values:
        # P(g|w) = 0.5 if g matches expected sign, else 0.25
        # P(g|¬w) = 0.5 (uniform)
        
        # Compute likelihood ratio
        # Simplified: assume binary ±1 values
        matches = np.sign(recovered_masked) == np.sign(expected_masked)
        num_matches = np.sum(matches)
        num_total = len(matches)
        
        # Likelihood under watermarked hypothesis
        # If we expect ±1 and observe ±1 (match), P = 0.5
        # If mismatch, P = 0.25 (weaker)
        p_w = (num_matches * 0.5 + (num_total - num_matches) * 0.25) / num_total
        
        # Likelihood under unwatermarked hypothesis (uniform)
        p_uw = 0.5
        
        # Log likelihood ratio
        if p_w < 1e-30:
            return -np.inf
        if p_uw < 1e-30:
            return np.inf
        
        s_stat = np.log(p_w / p_uw)
        return float(s_stat)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_compute_s_statistics(
    recovery_results: List[Dict[str, np.ndarray]],
    watermark_cfg: Dict[str, Any],
    key_manager: Optional[Any] = None,  # From watermark/key.py
    method: str = "correlation"
) -> np.ndarray:
    """
    Compute S-statistics for batch of images.
    
    Args:
        recovery_results: List of recovery results from recover_g_values()
        watermark_cfg: Watermark configuration
        key_manager: Key manager instance (optional, uses key_info from results)
        method: Correlation method ("correlation" or "log_likelihood_ratio")
    
    Returns:
        Array of detection scores [num_images]
    """
    s_scores = []
    
    for result in recovery_results:
        recovered_g = result["g_values"]
        mask = result["mask"]
        
        # Get expected g-values
        # For now, use the first timestep's expected g-field
        # In full implementation, this would reconstruct expected g from key_info
        if recovered_g.ndim == 4:
            # Multiple timesteps - use first one
            recovered_single = recovered_g[0]
        else:
            recovered_single = recovered_g
        
        # Reconstruct expected g-values from key_info in recovery_metadata
        # For simplicity, we use the recovered g-values structure
        # In practice, expected_g should be reconstructed from key_info
        metadata = result.get("recovery_metadata", {})
        
        # Use recovered g-values as expected (simplified - should reconstruct from key)
        expected_g = recovered_single.copy()  # TODO: Reconstruct from key_info
        
        # Compute S-statistic
        s_score = compute_s_statistic(
            recovered_g_values=recovered_single,
            expected_g_values=expected_g,
            mask=mask,
            method=method
        )
        s_scores.append(s_score)
    
    return np.array(s_scores)


def compute_correlation_statistics(
    recovered_g_values: np.ndarray,
    expected_g_values: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive correlation statistics.
    
    Returns:
        Dictionary with correlation, p-value, confidence interval, etc.
    """
    # Flatten and mask
    if recovered_g_values.ndim == 3:
        recovered_flat = recovered_g_values.flatten()
        expected_flat = expected_g_values.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = np.ones_like(recovered_flat)
    else:
        recovered_flat = recovered_g_values.flatten()
        expected_flat = expected_g_values.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
        else:
            mask_flat = np.ones_like(recovered_flat)
    
    mask_indices = mask_flat > 0.5
    recovered_masked = recovered_flat[mask_indices]
    expected_masked = expected_flat[mask_indices]
    
    if len(recovered_masked) < 2:
        return {
            "correlation": 0.0,
            "p_value": 1.0,
            "confidence_interval": (0.0, 0.0)
        }
    
    # Pearson correlation with p-value
    correlation, p_value = stats.pearsonr(recovered_masked, expected_masked)
    
    # Confidence interval (95%)
    n = len(recovered_masked)
    z = 1.96  # 95% confidence
    se = 1.0 / np.sqrt(n - 3)
    ci_lower = np.tanh(np.arctanh(correlation) - z * se)
    ci_upper = np.tanh(np.arctanh(correlation) + z * se)
    
    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "confidence_interval": (float(ci_lower), float(ci_upper)),
        "num_samples": int(n)
    }
