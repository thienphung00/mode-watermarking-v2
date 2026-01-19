"""
G_observed extraction from latent tensors.

DEPRECATED FOR DETECTION: This module is deprecated for watermark detection.
All detection must use compute_g_values() from detection/g_values.py which
provides key-dependent g-value computation.

This module may remain for:
    - Debugging and analysis
    - Legacy code compatibility
    - Non-detection use cases

For detection, use:
    from src.detection.g_values import compute_g_values
    g, mask = compute_g_values(x0, key, master_key)

Original documentation:
    Extracts the observed watermark signal from a latent representation.
    This module provides multiple extraction strategies to handle different
    watermark embedding approaches.

    Pipeline position:
        Image → Inversion → Latent z_T → observe.py → G_observed

    The extracted G_observed is then correlated with G_expected to compute
    the detection S-statistic.

    Design Principles:
        - Modular extraction functions (plug-in different strategies)
        - Pure functions: no side effects
        - Supports both spatial and frequency domain extraction
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Extraction Functions
# ============================================================================


def extract_raw(latent: torch.Tensor) -> torch.Tensor:
    """
    Raw extraction: use latent values directly as G_observed.
    
    Simplest extraction - assumes watermark is directly present
    in the latent values. Best for testing and simple embeddings.
    
    Args:
        latent: Latent tensor [B, C, H, W] or [C, H, W]
        
    Returns:
        G_observed tensor (same shape as input)
    """
    return latent.clone()


def extract_normalized(
    latent: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalized extraction: zero-mean, unit-variance latent.
    
    DEPRECATED FOR DETECTION: Do not use for watermark detection.
    Use compute_g_values() from detection/g_values.py instead.
    
    This function may remain for debugging/analysis only.
    
    Original documentation:
        Normalizes the latent to have zero mean and unit variance,
        making it directly comparable to Rademacher G_expected.
    
    Args:
        latent: Latent tensor
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized G_observed
    """
    # Flatten for statistics
    flat = latent.flatten()
    
    # Normalize
    mean = flat.mean()
    std = flat.std()
    
    normalized = (latent - mean) / (std + eps)
    
    return normalized


def extract_sign(latent: torch.Tensor) -> torch.Tensor:
    """
    Sign extraction: convert latent to ±1 values.
    
    DEPRECATED FOR DETECTION: Do not use for watermark detection.
    Use compute_g_values() from detection/g_values.py instead.
    
    This function may remain for debugging/analysis only.
    
    Original documentation:
        Extracts only the sign of each latent value, producing
        a Rademacher-distributed G_observed. Best correlation
        with Rademacher G_expected.
    
    Args:
        latent: Latent tensor
        
    Returns:
        G_observed with ±1 values
    """
    return torch.sign(latent)


def extract_whitened(
    latent: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Whitened extraction: high-pass filtered latent (residual).
    
    DEPRECATED FOR DETECTION: Do not use for watermark detection.
    Use compute_g_values() from detection/g_values.py instead.
    
    This function may remain for debugging/analysis only.
    
    Original documentation:
        Applies a Gaussian blur and subtracts from original to
        isolate high-frequency content where watermark signal lives.
        
        This is the "Whitened Matched Filter" approach.
    
    Args:
        latent: Latent tensor [B, C, H, W] or [C, H, W]
        kernel_size: Gaussian kernel size
        sigma: Gaussian sigma
        
    Returns:
        High-frequency residual as G_observed
    """
    # Handle 3D input
    squeeze_batch = False
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
        squeeze_batch = True
    
    B, C, H, W = latent.shape
    
    # Create Gaussian kernel
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    coords = torch.arange(kernel_size, dtype=latent.dtype, device=latent.device)
    coords = coords - (kernel_size - 1) / 2.0
    gaussian_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    
    kernel = gaussian_2d.expand(C, 1, kernel_size, kernel_size)
    
    # Apply blur
    padding = kernel_size // 2
    blurred = F.conv2d(latent, kernel, padding=padding, groups=C)
    
    # High-pass: original - blur
    residual = latent - blurred
    
    if squeeze_batch:
        residual = residual.squeeze(0)
    
    return residual


def extract_frequency_phase(
    latent: torch.Tensor,
    low_freq_cutoff: float = 0.12,
) -> torch.Tensor:
    """
    Frequency-domain extraction: extract phase information.
    
    For watermarks embedded in frequency domain, extract the
    phase component (normalized by magnitude).
    
    Args:
        latent: Latent tensor [B, C, H, W] or [C, H, W]
        low_freq_cutoff: Low-frequency cutoff (0-1)
        
    Returns:
        Phase-based G_observed
    """
    squeeze_batch = False
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
        squeeze_batch = True
    
    B, C, H, W = latent.shape
    
    # FFT
    latent_fft = torch.fft.fft2(latent, norm="ortho")
    
    # Extract phase (normalize by magnitude)
    eps = 1e-8
    magnitude = torch.abs(latent_fft)
    phase_normalized = latent_fft / (magnitude + eps)
    
    # Create low-frequency mask
    cutoff_h = max(1, int(H * low_freq_cutoff))
    cutoff_w = max(1, int(W * low_freq_cutoff))
    
    mask = torch.zeros(H, W, device=latent.device, dtype=latent.dtype)
    mask[:cutoff_h, :cutoff_w] = 1.0
    mask[:cutoff_h, -cutoff_w:] = 1.0
    mask[-cutoff_h:, :cutoff_w] = 1.0
    mask[-cutoff_h:, -cutoff_w:] = 1.0
    
    # Apply mask and inverse FFT
    masked_fft = phase_normalized * mask.unsqueeze(0).unsqueeze(0)
    result = torch.fft.ifft2(masked_fft, norm="ortho").real
    
    if squeeze_batch:
        result = result.squeeze(0)
    
    return result


def extract_gradient_magnitude(latent: torch.Tensor) -> torch.Tensor:
    """
    Gradient extraction: compute spatial gradients.
    
    Extracts edge/gradient information from the latent,
    useful if watermark modulates gradient patterns.
    
    Args:
        latent: Latent tensor [B, C, H, W] or [C, H, W]
        
    Returns:
        Gradient magnitude as G_observed
    """
    squeeze_batch = False
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
        squeeze_batch = True
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=latent.dtype, device=latent.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=latent.dtype, device=latent.device)
    
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
    
    B, C, H, W = latent.shape
    
    # Apply per channel
    grad_x = F.conv2d(latent, sobel_x.expand(C, 1, 3, 3), padding=1, groups=C)
    grad_y = F.conv2d(latent, sobel_y.expand(C, 1, 3, 3), padding=1, groups=C)
    
    # Magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    if squeeze_batch:
        magnitude = magnitude.squeeze(0)
    
    return magnitude


# ============================================================================
# Observer Class
# ============================================================================


class LatentObserver:
    """
    Extracts G_observed from latent tensors.
    
    DEPRECATED FOR DETECTION: Do not use for watermark detection.
    Use compute_g_values() from detection/g_values.py instead.
    
    This class may remain for debugging/analysis only.
    
    Original documentation:
        Provides a unified interface for different extraction strategies.
        The choice of extraction method should match how the watermark
        was embedded.
        
        Example:
            >>> observer = LatentObserver(method="whitened")
            >>> G_observed = observer.extract(latent)
    """
    
    # Registry of extraction methods
    METHODS: Dict[str, Callable] = {
        "raw": extract_raw,
        "normalized": extract_normalized,
        "sign": extract_sign,
        "whitened": extract_whitened,
        "frequency_phase": extract_frequency_phase,
        "gradient": extract_gradient_magnitude,
    }
    
    def __init__(
        self,
        method: str = "whitened",
        **kwargs,
    ):
        """
        Initialize observer.
        
        Args:
            method: Extraction method name
            **kwargs: Additional arguments for extraction function
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available: {list(self.METHODS.keys())}"
            )
        
        self.method = method
        self.kwargs = kwargs
    
    def extract(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Extract G_observed from latent.
        
        Args:
            latent: Latent tensor [B, C, H, W] or [C, H, W]
            
        Returns:
            G_observed tensor
        """
        extract_fn = self.METHODS[self.method]
        return extract_fn(latent, **self.kwargs)
    
    def extract_numpy(self, latent: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Extract G_observed and return as numpy array.
        
        Args:
            latent: Latent tensor or array
            
        Returns:
            G_observed as numpy array
        """
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)
        
        result = self.extract(latent)
        
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        return result
    
    @classmethod
    def extract_whitened(cls, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract whitened signal from latent (high-pass filter).
        
        DEPRECATED FOR DETECTION: Do not use for watermark detection.
        Use compute_g_values() from detection/g_values.py instead.
        
        Original documentation:
            Convenience class method for Stage 1 detection where z_0 is
            dominated by image content. The whitening filter suppresses
            the image structure to isolate the watermark signal.
        
        Args:
            latent: Latent tensor [B, C, H, W] or [C, H, W]
            **kwargs: Additional arguments for extract_whitened function
            
        Returns:
            Whitened G_observed tensor
        """
        return extract_whitened(latent, **kwargs)
    
    @classmethod
    def extract_normalized(cls, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract normalized signal from latent (zero-mean, unit-variance).
        
        DEPRECATED FOR DETECTION: Do not use for watermark detection.
        Use compute_g_values() from detection/g_values.py instead.
        
        Original documentation:
            Convenience class method for Stage 2 detection where z_T is
            noise-like (image structure removed). Simple normalization
            preserves the most signal energy.
        
        Args:
            latent: Latent tensor [B, C, H, W] or [C, H, W]
            **kwargs: Additional arguments for extract_normalized function
            
        Returns:
            Normalized G_observed tensor
        """
        return extract_normalized(latent, **kwargs)


# ============================================================================
# Convenience Functions  
# ============================================================================


def observe_latent(
    latent: Union[torch.Tensor, np.ndarray],
    method: str = "whitened",
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to extract G_observed.
    
    Args:
        latent: Latent tensor
        method: Extraction method
        **kwargs: Additional arguments
        
    Returns:
        G_observed tensor
    """
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)
    
    observer = LatentObserver(method=method, **kwargs)
    return observer.extract(latent)


def observe_latent_numpy(
    latent: Union[torch.Tensor, np.ndarray],
    method: str = "whitened",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function returning numpy array.
    
    Args:
        latent: Latent tensor or array
        method: Extraction method
        **kwargs: Additional arguments
        
    Returns:
        G_observed as numpy array
    """
    result = observe_latent(latent, method=method, **kwargs)
    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    return result


def list_extraction_methods() -> list:
    """Return list of available extraction methods."""
    return list(LatentObserver.METHODS.keys())

