"""
Mask generation for spatial and frequency-domain masking.

Implements zero-mean masks for non-distortionary watermark embedding.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import fft


class MaskGenerator:
    """
    Generates spatial or frequency-domain masks for watermark embedding.

    Ensures masks satisfy zero-mean property: E[M ⊙ G_t] ≈ 0
    """

    def __init__(
        self,
        mode: str = "frequency",
        strength: float = 0.8,
        # Frequency parameters
        band: str = "low",
        cutoff_freq: float = 0.30,
        bandwidth_fraction: float = 0.15,
        # Spatial parameters
        shape_type: str = "radial",
        radius_fraction: float = 0.50,
    ):
        """
        Initialize mask generator.

        Args:
            mode: "frequency" or "spatial"
            strength: Mask strength multiplier (0-1)
            band: Frequency band "low" or "high"
            cutoff_freq: Cutoff frequency fraction (0-1)
            bandwidth_fraction: Transition bandwidth fraction
            shape_type: Spatial shape "radial" or "rect"
            radius_fraction: Radius fraction for radial masks
        """
        self.mode = mode.lower()
        self.strength = strength
        self.band = band.lower()
        self.cutoff_freq = cutoff_freq
        self.bandwidth_fraction = bandwidth_fraction
        self.shape_type = shape_type.lower()
        self.radius_fraction = radius_fraction

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.mode not in {"frequency", "spatial"}:
            raise ValueError(f"mode must be 'frequency' or 'spatial', got '{self.mode}'")

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")

        if self.mode == "frequency":
            if self.band not in {"low", "high"}:
                raise ValueError(f"band must be 'low' or 'high', got '{self.band}'")
            if not 0.0 < self.cutoff_freq <= 1.0:
                raise ValueError(f"cutoff_freq must be in (0, 1], got {self.cutoff_freq}")
            if not 0.0 <= self.bandwidth_fraction <= 1.0:
                raise ValueError(
                    f"bandwidth_fraction must be in [0, 1], got {self.bandwidth_fraction}"
                )

        if self.mode == "spatial":
            if self.shape_type not in {"radial", "rect"}:
                raise ValueError(f"shape_type must be 'radial' or 'rect', got '{self.shape_type}'")
            if not 0.0 <= self.radius_fraction <= 1.0:
                raise ValueError(
                    f"radius_fraction must be in [0, 1], got {self.radius_fraction}"
                )

    def generate(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate a zero-mean mask.

        Args:
            shape: (C, H, W) shape for mask

        Returns:
            Zero-mean mask tensor [C, H, W]
        """
        C, H, W = shape

        if self.mode == "frequency":
            mask = self._generate_frequency_mask(H, W)
        else:  # spatial
            mask = self._generate_spatial_mask(H, W)

        # Normalize to zero mean
        mask = mask - np.mean(mask)

        # Replicate across channels
        M = np.stack([mask] * C, axis=0)

        # Apply strength multiplier
        M = M * self.strength

        return M.astype(np.float32)

    def _generate_frequency_mask(self, H: int, W: int) -> np.ndarray:
        """
        Generate frequency-domain mask.

        Args:
            H: Height
            W: Width

        Returns:
            Spatial mask [H, W] after inverse DCT
        """
        # Create mask in frequency domain (DCT)
        mask_freq = np.zeros((H, W), dtype=np.float32)

        cutoff_h = int(H * self.cutoff_freq)
        cutoff_w = int(W * self.cutoff_freq)

        if self.band == "low":
            # Low-frequency mask
            mask_freq[:cutoff_h, :cutoff_w] = 1.0

            # Apply smooth transition if bandwidth > 0
            if self.bandwidth_fraction > 0:
                for i in range(cutoff_h):
                    for j in range(cutoff_w):
                        dist = np.sqrt((i / max(cutoff_h, 1)) ** 2 + (j / max(cutoff_w, 1)) ** 2)
                        if dist <= 1.0:
                            transition = 1.0 - max(
                                0, (dist - (1.0 - self.bandwidth_fraction)) / self.bandwidth_fraction
                            )
                            mask_freq[i, j] = max(0, transition)
        else:
            # High-frequency mask
            mask_freq = np.ones((H, W), dtype=np.float32)
            mask_freq[:cutoff_h, :cutoff_w] = 0.0

        # Convert to spatial domain via inverse DCT
        mask_spatial = fft.idctn(mask_freq, norm="ortho")

        return mask_spatial.astype(np.float32)

    def _generate_spatial_mask(self, H: int, W: int) -> np.ndarray:
        """
        Generate spatial-domain mask.

        Args:
            H: Height
            W: Width

        Returns:
            Spatial mask [H, W]
        """
        if self.shape_type == "radial":
            # Radial mask
            y, x = np.ogrid[:H, :W]
            cy, cx = H / 2.0, W / 2.0
            r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            R = np.sqrt(cy**2 + cx**2) * self.radius_fraction
            mask = 1.0 - (r / max(R, 1e-10))
            mask = np.clip(mask, 0.0, 1.0)
        else:  # rect
            # Rectangular mask
            mask = np.ones((H, W), dtype=np.float32)
            border_h = int(H * (1.0 - self.radius_fraction) / 2)
            border_w = int(W * (1.0 - self.radius_fraction) / 2)
            if border_h > 0:
                mask[:border_h, :] = 0.0
                mask[-border_h:, :] = 0.0
            if border_w > 0:
                mask[:, :border_w] = 0.0
                mask[:, -border_w:] = 0.0

        return mask.astype(np.float32)

    def verify_zero_mean(self, mask: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Verify that mask has zero mean.

        Args:
            mask: Mask tensor
            tolerance: Absolute tolerance

        Returns:
            True if mask is zero-mean within tolerance
        """
        mean = np.mean(mask)
        return abs(mean) < tolerance

