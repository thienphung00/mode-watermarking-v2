"""
Unified G-field generation for watermark embedding and detection.

This module provides a single canonical implementation for G-field generation
that is used by both generation (embedding) and detection pipelines.

Key Features:
- PRF-seed → G-field generation
- Binary and continuous mapping modes
- Zero-mean and unit-variance normalization
- Frequency-domain processing (lowpass, highpass, bandpass)
- Support for single G-fields and scheduled/per-timestep G-fields

Design Principles:
- Pure and stateless: no side effects
- Deterministic: same inputs → same outputs
- Domain-agnostic: usable by both embedding and detection
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy import fft


def create_frequency_mask(
    H: int,
    W: int,
    mode: str = "bandpass",
    low_cut: Optional[float] = None,
    high_cut: Optional[float] = None,
) -> np.ndarray:
    """
    Create frequency-domain mask for DCT filtering.
    
    This function creates the exact mask geometry used in apply_frequency_bandpass().
    It identifies which frequency bins carry watermark energy based on the filtering mode.
    
    The mask is deterministic given (H, W, mode, low_cut, high_cut) and represents
    the frequency support of the watermark signal. Positions with mask=1 are valid
    watermark positions; positions with mask=0 are filtered out.
    
    Args:
        H: Height of the frequency domain
        W: Width of the frequency domain
        mode: Filtering mode - "lowpass", "highpass", or "bandpass"
        low_cut: Low frequency cutoff (fraction, 0.0 to 1.0)
        high_cut: High frequency cutoff (fraction, 0.0 to 1.0)
    
    Returns:
        Binary frequency mask [H, W] with values in {0, 1}
        1 indicates valid watermark position, 0 indicates filtered position
    
    Raises:
        ValueError: If mode is invalid or cutoffs are out of range
    
    Examples:
        >>> # Lowpass mask (keep frequencies below 0.12)
        >>> mask = create_frequency_mask(64, 64, mode="lowpass", low_cut=0.12)
        >>> print(f"Mask support: {mask.sum()} positions")
        
        >>> # Bandpass mask (keep frequencies between 0.05 and 0.4)
        >>> mask = create_frequency_mask(64, 64, mode="bandpass", low_cut=0.05, high_cut=0.4)
    """
    if mode not in {"lowpass", "highpass", "bandpass"}:
        raise ValueError(f"mode must be 'lowpass', 'highpass', or 'bandpass', got '{mode}'")
    
    mask = np.zeros((H, W), dtype=np.float32)
    
    if mode == "lowpass":
        if low_cut is None:
            raise ValueError("low_cut required for lowpass mode")
        if not 0.0 < low_cut <= 1.0:
            raise ValueError(f"low_cut must be in (0, 1], got {low_cut}")
        cutoff_h = max(1, int(H * low_cut))
        cutoff_w = max(1, int(W * low_cut))
        mask[:cutoff_h, :cutoff_w] = 1.0
    
    elif mode == "highpass":
        if high_cut is None:
            raise ValueError("high_cut required for highpass mode")
        if not 0.0 < high_cut <= 1.0:
            raise ValueError(f"high_cut must be in (0, 1], got {high_cut}")
        cutoff_h = max(1, int(H * high_cut))
        cutoff_w = max(1, int(W * high_cut))
        mask = np.ones((H, W), dtype=np.float32)
        mask[:cutoff_h, :cutoff_w] = 0.0
    
    elif mode == "bandpass":
        if low_cut is None or high_cut is None:
            raise ValueError("low_cut and high_cut required for bandpass mode")
        if not 0.0 <= low_cut < high_cut <= 1.0:
            raise ValueError(
                f"Frequency cutoffs must satisfy 0 <= low_cut < high_cut <= 1, "
                f"got low_cut={low_cut}, high_cut={high_cut}"
            )
        low_idx_h = max(1, int(H * low_cut))
        high_idx_h = min(H - 1, int(H * high_cut))
        low_idx_w = max(1, int(W * low_cut))
        high_idx_w = min(W - 1, int(W * high_cut))
        mask[low_idx_h:high_idx_h, low_idx_w:high_idx_w] = 1.0
        # Also keep DC component (0, 0)
        mask[0, 0] = 1.0
    
    return mask


def apply_frequency_bandpass(
    tensor: np.ndarray,
    low_cut: Optional[float] = None,
    high_cut: Optional[float] = None,
    mode: str = "bandpass",
) -> np.ndarray:
    """
    Apply frequency-domain filtering to a tensor using DCT.
    
    Supports lowpass, highpass, and bandpass filtering modes.
    This is a unified replacement for apply_frequency_filter and apply_frequency_mask.
    
    Args:
        tensor: Input tensor [C, H, W] or [H, W]
        low_cut: Low frequency cutoff (fraction, 0.0 to 1.0). 
                 For lowpass mode, this is the cutoff.
                 For bandpass mode, this is the lower bound.
        high_cut: High frequency cutoff (fraction, 0.0 to 1.0).
                  For highpass mode, this is the cutoff.
                  For bandpass mode, this is the upper bound.
        mode: Filtering mode - "lowpass", "highpass", or "bandpass"
    
    Returns:
        Filtered tensor with same shape
    
    Raises:
        ValueError: If mode is invalid or cutoffs are out of range
    
    Examples:
        >>> # Lowpass filter (keep frequencies below 0.12)
        >>> filtered = apply_frequency_bandpass(tensor, low_cut=0.12, mode="lowpass")
        
        >>> # Highpass filter (keep frequencies above 0.05)
        >>> filtered = apply_frequency_bandpass(tensor, high_cut=0.05, mode="highpass")
        
        >>> # Bandpass filter (keep frequencies between 0.05 and 0.4)
        >>> filtered = apply_frequency_bandpass(tensor, low_cut=0.05, high_cut=0.4, mode="bandpass")
    """
    if mode not in {"lowpass", "highpass", "bandpass"}:
        raise ValueError(f"mode must be 'lowpass', 'highpass', or 'bandpass', got '{mode}'")
    
    # Handle 2D and 3D tensors
    if tensor.ndim == 2:
        tensor = tensor[np.newaxis, :, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    C, H, W = tensor.shape
    filtered = np.zeros_like(tensor)
    
    # Create frequency mask using shared function
    # For highpass mode, we need to map high_cut to low_cut parameter
    if mode == "highpass":
        mask = create_frequency_mask(H, W, mode=mode, high_cut=high_cut)
    else:
        mask = create_frequency_mask(H, W, mode=mode, low_cut=low_cut, high_cut=high_cut)
    
    # Apply per channel using orthonormal DCT
    for c in range(C):
        channel = tensor[c, :, :]
        # Forward DCT
        channel_dct = fft.dctn(channel, norm="ortho")
        # Apply mask
        channel_dct *= mask
        # Inverse DCT
        filtered[c, :, :] = fft.idctn(channel_dct, norm="ortho")
    
    if squeeze_output:
        filtered = filtered[0, :, :]
    
    return filtered.astype(np.float32)


class GFieldGenerator:
    """
    Unified G-field generator for watermark embedding and detection.
    
    This class provides a single canonical implementation for generating
    G-fields from PRF seeds. It supports both generation (embedding) and
    detection use cases.
    
    Responsibilities:
    - Generate deterministic G-fields from key streams
    - Apply frequency-domain filtering
    - Normalize to zero-mean and unit-variance
    - Support binary and continuous mapping modes
    - Support single G-fields and scheduled/per-timestep G-fields
    """

    def __init__(
        self,
        mapping_mode: str = "binary",
        bit_pos: int = 30,
        domain: str = "spatial",
        frequency_mode: str = "lowpass",
        low_freq_cutoff: float = 0.12,
        high_freq_cutoff: Optional[float] = None,
        normalize_zero_mean: bool = True,
        normalize_unit_variance: bool = True,
        continuous_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize G-field generator.
        
        Args:
            mapping_mode: "binary" for ±1 (Rademacher), "continuous" for [-1, 1], or "rademacher" (alias for binary)
            bit_pos: Bit position for extraction (0-63) for binary/rademacher mode
            domain: "spatial" or "frequency"
            frequency_mode: "lowpass", "highpass", or "bandpass"
            low_freq_cutoff: Low frequency cutoff fraction (0.0 to 1.0)
            high_freq_cutoff: High frequency cutoff fraction (0.0 to 1.0), required for bandpass mode
            normalize_zero_mean: Whether to normalize to zero mean
            normalize_unit_variance: Whether to normalize to unit variance
            continuous_range: Range for continuous mapping (low, high)
        """
        # Normalize mapping_mode (support "rademacher" as alias for "binary")
        if mapping_mode.lower() == "rademacher":
            mapping_mode = "binary"
        
        self.mapping_mode = mapping_mode.lower()
        self.bit_pos = bit_pos
        self.domain = domain.lower()
        self.frequency_mode = frequency_mode.lower()
        self.low_freq_cutoff = low_freq_cutoff
        self.high_freq_cutoff = high_freq_cutoff
        self.normalize_zero_mean = normalize_zero_mean
        self.normalize_unit_variance = normalize_unit_variance
        self.continuous_range = continuous_range

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.mapping_mode not in {"binary", "continuous"}:
            raise ValueError(
                f"mapping_mode must be 'binary' or 'continuous', got '{self.mapping_mode}'"
            )

        if self.domain not in {"spatial", "frequency"}:
            raise ValueError(f"domain must be 'spatial' or 'frequency', got '{self.domain}'")

        if self.domain == "frequency":
            if self.frequency_mode not in {"lowpass", "highpass", "bandpass"}:
                raise ValueError(
                    f"frequency_mode must be 'lowpass', 'highpass', or 'bandpass', "
                    f"got '{self.frequency_mode}'"
                )
            
            if self.frequency_mode == "bandpass" and self.high_freq_cutoff is None:
                raise ValueError("high_freq_cutoff required for bandpass mode")
            
            if not 0 < self.low_freq_cutoff <= 1.0:
                raise ValueError(
                    f"low_freq_cutoff must be in (0, 1], got {self.low_freq_cutoff}"
                )
            
            if self.high_freq_cutoff is not None:
                if not 0 < self.high_freq_cutoff <= 1.0:
                    raise ValueError(
                        f"high_freq_cutoff must be in (0, 1], got {self.high_freq_cutoff}"
                    )
                if self.low_freq_cutoff >= self.high_freq_cutoff:
                    raise ValueError(
                        f"low_freq_cutoff must be < high_freq_cutoff, "
                        f"got {self.low_freq_cutoff} >= {self.high_freq_cutoff}"
                    )

        if self.mapping_mode == "continuous" and self.continuous_range is None:
            raise ValueError("continuous_range required for continuous mapping")

    def generate_g_field(
        self,
        shape: Tuple[int, int, int],
        seeds: Union[List[int], Iterator[int]],
        timestep: Optional[int] = None,
        return_mask: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate a single G-field tensor from PRF seeds.
        
        Args:
            shape: (C, H, W) shape for G-field
            seeds: List or iterator of 64-bit integers from PRF
            timestep: Optional timestep for time-dependent features (currently unused)
            return_mask: If True, return frequency mask alongside G-field.
                        The mask identifies valid watermark positions based on
                        the exact same DCT mask geometry used in frequency filtering.
                        Only meaningful when domain="frequency".
        
        Returns:
            If return_mask=False: G-field tensor [C, H, W] as float32
            If return_mask=True: Tuple of (G-field [C, H, W], mask [H, W])
                - mask is binary {0, 1} indicating valid watermark positions
                - For spatial domain, mask is all ones [H, W] (all positions valid)
                - For frequency domain, mask is [H, W] with frequency support geometry
                - Note: mask is 2D [H, W] and should be broadcast to [C, H, W] when needed
                  (same mask applies to all channels)
        
        Examples:
            >>> # Standard usage (no mask)
            >>> G = generator.generate_g_field((4, 64, 64), seeds)
            
            >>> # With mask (for detection alignment)
            >>> G, mask = generator.generate_g_field((4, 64, 64), seeds, return_mask=True)
            >>> # mask shape: [64, 64] (same for all channels)
        """
        C, H, W = shape
        num_elements = C * H * W

        # Convert iterator to list if needed
        if isinstance(seeds, Iterator):
            values = []
            for _ in range(num_elements):
                val = next(seeds, None)
                if val is None:
                    raise ValueError("Seed stream exhausted before enough values generated")
                values.append(val)
        else:
            if len(seeds) < num_elements:
                raise ValueError(
                    f"Insufficient seeds: need {num_elements}, got {len(seeds)}"
                )
            values = seeds[:num_elements]

        # Convert to numpy array
        values_array = np.array(values, dtype=np.uint64)

        # Map to g-values
        g_values = self._map_to_gvalues(values_array)

        # Reshape to [C, H, W]
        G = g_values.reshape(shape).astype(np.float32)

        # Generate frequency mask if requested
        mask = None
        if return_mask:
            mask = self._get_frequency_mask(H, W)

        # Apply frequency filtering if needed
        if self.domain == "frequency":
            G = self._apply_frequency_filter(G)

        # Apply normalization
        if self.normalize_zero_mean or self.normalize_unit_variance:
            G = self._normalize(G)

        if return_mask:
            return G, mask
        return G

    def generate_schedule(
        self,
        shape: Tuple[int, int, int],
        timesteps: list[int],
        seeds: Union[List[int], Iterator[int]],
    ) -> Dict[int, np.ndarray]:
        """
        Generate G-fields for multiple timesteps from PRF seeds.
        
        Args:
            shape: (C, H, W) shape for G-fields
            timesteps: List of timesteps to generate G-fields for
            seeds: List or iterator of PRF seeds
        
        Returns:
            Dictionary mapping timestep -> G-field tensor
        """
        schedule = {}
        # Convert to list if iterator to allow multiple passes
        if isinstance(seeds, Iterator):
            seeds = list(seeds)
        
        C, H, W = shape
        elements_per_timestep = C * H * W
        total_elements = len(timesteps) * elements_per_timestep
        
        if len(seeds) < total_elements:
            raise ValueError(
                f"Insufficient seeds: need {total_elements}, got {len(seeds)}"
            )
        
        for i, t in enumerate(timesteps):
            start_idx = i * elements_per_timestep
            end_idx = start_idx + elements_per_timestep
            timestep_seeds = seeds[start_idx:end_idx]
            G_t = self.generate_g_field(shape, timestep_seeds, timestep=t)
            schedule[t] = G_t
        return schedule

    def _map_to_gvalues(self, values: np.ndarray) -> np.ndarray:
        """
        Map PRF seeds to g-values.
        
        Args:
            values: Array of 64-bit integers from PRF
        
        Returns:
            Array of g-values (binary ±1 or continuous)
        """
        if self.mapping_mode == "binary":
            # Extract bit and map to ±1 (Rademacher distribution)
            bits = ((values >> np.uint64(self.bit_pos)) & np.uint64(1)).astype(np.int32)
            g = (2 * bits - 1).astype(np.float32)
            return g

        elif self.mapping_mode == "continuous":
            # Map to [0, 1] then to [-1, 1]
            m = float(2**64)
            u = values.astype(np.float64) / m
            g = 2.0 * u - 1.0

            # Scale to continuous_range if specified
            if self.continuous_range is not None:
                low, high = self.continuous_range
                if high <= low:
                    raise ValueError(
                        f"continuous_range must satisfy low < high, got ({low}, {high})"
                    )
                scale = (high - low) / 2.0
                offset = (high + low) / 2.0
                g = g * scale + offset

            return g.astype(np.float32)

        else:
            raise ValueError(f"Unknown mapping mode: {self.mapping_mode}")

    def _get_frequency_mask(self, H: int, W: int) -> np.ndarray:
        """
        Get frequency mask for the configured filtering mode.
        
        This returns the exact mask geometry used in frequency filtering.
        The mask identifies which frequency bins carry watermark energy.
        
        Args:
            H: Height of frequency domain
            W: Width of frequency domain
        
        Returns:
            Frequency mask [H, W] for frequency domain, or [H, W] all-ones for spatial domain
            Values are in {0, 1} where 1 indicates valid watermark position
        """
        if self.domain == "spatial":
            # For spatial domain, all positions are valid
            return np.ones((H, W), dtype=np.float32)
        
        # For frequency domain, create mask based on filtering mode
        # Note: For highpass mode, low_freq_cutoff is used as the cutoff (frequencies above this are kept)
        if self.frequency_mode == "lowpass":
            return create_frequency_mask(H, W, mode="lowpass", low_cut=self.low_freq_cutoff)
        elif self.frequency_mode == "highpass":
            # For highpass, low_freq_cutoff acts as the high_cut parameter
            return create_frequency_mask(H, W, mode="highpass", high_cut=self.low_freq_cutoff)
        elif self.frequency_mode == "bandpass":
            if self.high_freq_cutoff is None:
                raise ValueError("high_freq_cutoff required for bandpass mode")
            return create_frequency_mask(
                H, W, 
                mode="bandpass", 
                low_cut=self.low_freq_cutoff, 
                high_cut=self.high_freq_cutoff
            )
        else:
            raise ValueError(f"Unknown frequency_mode: {self.frequency_mode}")

    def _apply_frequency_filter(self, G: np.ndarray) -> np.ndarray:
        """
        Apply frequency-domain filtering to G-field.
        
        Args:
            G: Spatial G-field [C, H, W]
        
        Returns:
            Filtered G-field [C, H, W]
        """
        # Use unified frequency bandpass function
        if self.frequency_mode == "lowpass":
            return apply_frequency_bandpass(G, low_cut=self.low_freq_cutoff, mode="lowpass")
        elif self.frequency_mode == "highpass":
            return apply_frequency_bandpass(G, high_cut=self.low_freq_cutoff, mode="highpass")
        elif self.frequency_mode == "bandpass":
            if self.high_freq_cutoff is None:
                raise ValueError("high_freq_cutoff required for bandpass mode")
            return apply_frequency_bandpass(
                G, 
                low_cut=self.low_freq_cutoff, 
                high_cut=self.high_freq_cutoff, 
                mode="bandpass"
            )
        else:
            raise ValueError(f"Unknown frequency_mode: {self.frequency_mode}")

    def _normalize(self, G: np.ndarray) -> np.ndarray:
        """
        Normalize G-field to zero-mean and/or unit-variance.
        
        Args:
            G: G-field tensor [C, H, W]
        
        Returns:
            Normalized G-field
        """
        G_norm = G.copy()
        eps = 1e-8

        # Zero-mean normalization (global across all dimensions)
        if self.normalize_zero_mean:
            global_mean = np.mean(G_norm)
            G_norm = G_norm - global_mean

        # Unit-variance normalization (global)
        if self.normalize_unit_variance:
            global_std = np.std(G_norm)
            G_norm = G_norm / (global_std + eps)
                  
        return G_norm


# ============================================================================
# Validation and Diagnostics
# ============================================================================


def validate_mask_alignment(
    G: np.ndarray,
    mask: np.ndarray,
    g_field_config: Optional[dict] = None,
    tolerance: float = 1e-6,
) -> Dict[str, any]:
    """
    Validate that mask geometry matches G-field and expected frequency support.
    
    This function performs lightweight diagnostics to confirm that:
    1. Mask shape matches flattened G-field
    2. Mask sum equals expected frequency support size
    3. Mask values are binary {0, 1}
    
    Args:
        G: G-field tensor [C, H, W]
        mask: Frequency mask [C, H, W] or [H, W]
        g_field_config: Optional G-field config for expected support calculation
        tolerance: Numerical tolerance for assertions
    
    Returns:
        Dictionary with diagnostic information:
        - mask_shape: Shape of mask
        - g_shape: Shape of G-field
        - mask_sum: Sum of mask (number of valid positions)
        - mask_pct: Percentage of positions that are valid
        - is_aligned: Whether mask shape matches flattened G
        - mean_g_before: Mean |G| before masking
        - mean_g_after: Mean |G| after masking (only valid positions)
        - diagnostics: List of diagnostic messages
    
    Raises:
        AssertionError: If mask geometry doesn't match expectations
    
    Example:
        >>> G, mask = generator.generate_g_field(shape, seeds, return_mask=True)
        >>> diag = validate_mask_alignment(G, mask, g_field_config)
        >>> print(f"Valid positions: {diag['mask_pct']:.1f}%")
    """
    diagnostics = []
    C, H, W = G.shape
    
    # Normalize mask shape
    if mask.ndim == 2:
        # Frequency domain: [H, W] -> [C, H, W]
        mask_expanded = np.broadcast_to(mask[np.newaxis, :, :], (C, H, W)).copy()
    elif mask.ndim == 3:
        # Already [C, H, W]
        mask_expanded = mask
    else:
        raise ValueError(f"Mask must be 2D [H, W] or 3D [C, H, W], got shape {mask.shape}")
    
    # Assert mask shape matches G-field
    assert mask_expanded.shape == G.shape, (
        f"Mask shape {mask_expanded.shape} doesn't match G-field shape {G.shape}"
    )
    diagnostics.append(f"✓ Mask shape matches G-field: {mask_expanded.shape}")
    
    # Assert mask is binary
    unique_values = np.unique(mask_expanded)
    assert np.allclose(unique_values, [0.0, 1.0]) or np.allclose(unique_values, [1.0]) or np.allclose(unique_values, [0.0]), (
        f"Mask must be binary {{0, 1}}, got unique values: {unique_values}"
    )
    diagnostics.append(f"✓ Mask is binary with values: {unique_values}")
    
    # Compute mask statistics
    mask_sum = float(mask_expanded.sum())
    total_elements = C * H * W
    mask_pct = 100.0 * mask_sum / total_elements
    
    # Compute G-field statistics
    G_flat = G.flatten()
    mask_flat = mask_expanded.flatten()
    mean_g_before = float(np.mean(np.abs(G_flat)))
    mean_g_after = float(np.mean(np.abs(G_flat[mask_flat > 0.5]))) if mask_sum > 0 else 0.0
    
    # Validate expected frequency support if config provided
    if g_field_config is not None:
        domain = g_field_config.get("domain", "spatial").lower()
        if domain == "frequency":
            # Compute expected support from config
            generator = GFieldGenerator(**g_field_config)
            expected_mask = generator._get_frequency_mask(H, W)  # [H, W]
            expected_sum = float(expected_mask.sum() * C)  # Broadcast to all channels
            
            # Allow small tolerance for floating point
            assert abs(mask_sum - expected_sum) < tolerance, (
                f"Mask sum {mask_sum} doesn't match expected support {expected_sum} "
                f"(diff: {abs(mask_sum - expected_sum)})"
            )
            diagnostics.append(f"✓ Mask sum matches expected frequency support: {mask_sum:.0f}")
    
    diagnostics.append(f"✓ Valid positions: {mask_pct:.2f}% ({mask_sum:.0f}/{total_elements})")
    diagnostics.append(f"✓ Mean |G| before masking: {mean_g_before:.6f}")
    diagnostics.append(f"✓ Mean |G| after masking (valid only): {mean_g_after:.6f}")
    
    return {
        "mask_shape": mask_expanded.shape,
        "g_shape": G.shape,
        "mask_sum": mask_sum,
        "mask_pct": mask_pct,
        "is_aligned": True,
        "mean_g_before": mean_g_before,
        "mean_g_after": mean_g_after,
        "diagnostics": diagnostics,
    }


def diagnose_mask_usage(
    G: np.ndarray,
    mask: np.ndarray,
    g_values: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Diagnose mask usage in likelihood/correlation computation.
    
    This function helps verify that masking improves detection SNR by:
    1. Computing correlation with and without masking
    2. Comparing scores for correct vs wrong keys
    3. Reporting mask effectiveness
    
    Args:
        G: G-field tensor [C, H, W]
        mask: Frequency mask [C, H, W] or [H, W]
        g_values: Optional g-values [N] for correlation computation
    
    Returns:
        Dictionary with diagnostic information:
        - masked_correlation: Correlation using only masked positions
        - unmasked_correlation: Correlation using all positions
        - snr_improvement: SNR improvement from masking
        - mask_effectiveness: Effectiveness metric
    """
    C, H, W = G.shape
    
    # Normalize mask shape
    if mask.ndim == 2:
        mask_expanded = np.broadcast_to(mask[np.newaxis, :, :], (C, H, W)).copy()
    else:
        mask_expanded = mask
    
    G_flat = G.flatten()
    mask_flat = mask_expanded.flatten()
    
    # Compute statistics
    valid_indices = mask_flat > 0.5
    n_valid = int(valid_indices.sum())
    n_total = len(G_flat)
    
    result = {
        "n_valid": n_valid,
        "n_total": n_total,
        "valid_pct": 100.0 * n_valid / n_total,
    }
    
    # If g-values provided, compute correlation
    if g_values is not None:
        g_flat = g_values.flatten() if g_values.ndim > 1 else g_values
        
        if len(g_flat) != len(G_flat):
            raise ValueError(f"g_values length {len(g_flat)} doesn't match G-field {len(G_flat)}")
        
        # Correlation with masking
        G_masked = G_flat[valid_indices]
        g_masked = g_flat[valid_indices]
        if len(G_masked) > 0:
            correlation_masked = float(np.corrcoef(G_masked, g_masked)[0, 1])
        else:
            correlation_masked = 0.0
        
        # Correlation without masking
        correlation_unmasked = float(np.corrcoef(G_flat, g_flat)[0, 1])
        
        result.update({
            "correlation_masked": correlation_masked,
            "correlation_unmasked": correlation_unmasked,
            "snr_improvement": abs(correlation_masked) / (abs(correlation_unmasked) + 1e-10),
        })
    
    return result


# ============================================================================
# Detection-specific convenience functions
# ============================================================================


def compute_g_expected(
    master_key: str,
    key_id: str,
    shape: Tuple[int, ...],
    prf_algorithm: str = "chacha20",
    mapping_mode: str = "rademacher",
    domain: str = "spatial",
    normalize_zero_mean: Optional[bool] = None,
    normalize_unit_variance: bool = False,
    return_mask: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to compute expected G-field for detection.
    
    This is the primary interface for detection. Given only the
    master_key and key_id, compute the expected watermark pattern.
    
    Args:
        master_key: Secret master key (detector's secret)
        key_id: Public key identifier
        shape: Shape of G-field (C, H, W) or (B, C, H, W)
        prf_algorithm: PRF algorithm to use
        mapping_mode: "rademacher" (alias for binary) or "continuous"
        domain: "spatial" or "frequency"
        normalize_zero_mean: Ensure G-field has zero mean.
                           Default: False for Rademacher (already mean-zero),
                           True for continuous.
        normalize_unit_variance: Ensure G-field has unit variance
        return_mask: If True, return frequency mask alongside G-field
        **kwargs: Additional arguments passed to GFieldGenerator (e.g., frequency_mode, low_freq_cutoff)
    
    Returns:
        If return_mask=False: G-field numpy array
        If return_mask=True: Tuple of (G-field, mask)
    
    Example:
        >>> G_expected = compute_g_expected(
        ...     master_key="secret_key",
        ...     key_id="abc123",
        ...     shape=(4, 64, 64),
        ... )
        >>> # With mask:
        >>> G_expected, mask = compute_g_expected(
        ...     master_key="secret_key",
        ...     key_id="abc123",
        ...     shape=(4, 64, 64),
        ...     domain="frequency",
        ...     frequency_mode="bandpass",
        ...     low_freq_cutoff=0.05,
        ...     high_freq_cutoff=0.4,
        ...     return_mask=True,
        ... )
    """
    from ..detection.prf import PRFKeyDerivation
    from ..core.config import PRFConfig
    
    # Calculate total elements
    n_elements = int(np.prod(shape))
    
    # Generate PRF seeds
    prf_config = PRFConfig(algorithm=prf_algorithm)
    prf = PRFKeyDerivation(master_key, prf_config)
    seeds = prf.generate_seeds(key_id, n_elements)
    
    # Determine normalization defaults
    if mapping_mode.lower() == "rademacher" or mapping_mode.lower() == "binary":
        if normalize_zero_mean is None:
            normalize_zero_mean = False  # Rademacher is already mean-zero
    else:
        if normalize_zero_mean is None:
            normalize_zero_mean = True
    
    # Create generator with kwargs (for frequency_mode, cutoffs, etc.)
    generator_kwargs = {
        "mapping_mode": mapping_mode,
        "domain": domain,
        "normalize_zero_mean": normalize_zero_mean,
        "normalize_unit_variance": normalize_unit_variance,
    }
    generator_kwargs.update(kwargs)
    generator = GFieldGenerator(**generator_kwargs)
    
    # Reshape to (C, H, W) if needed
    if len(shape) == 4:
        # (B, C, H, W) -> (C, H, W)
        g_field_shape = shape[1:]
    else:
        g_field_shape = shape
    
    # Generate G-field (and mask if requested)
    if return_mask:
        G, mask = generator.generate_g_field(g_field_shape, seeds, return_mask=True)
        # Reshape to original shape if needed
        if len(shape) == 4:
            G = np.broadcast_to(G[np.newaxis, :, :, :], shape)
            # For mask, broadcast appropriately
            if mask.ndim == 2:
                # [H, W] -> [B, C, H, W]
                C, H, W = shape[1], shape[2], shape[3]
                mask = np.broadcast_to(mask[np.newaxis, np.newaxis, :, :], shape).copy()
            elif mask.ndim == 3:
                # [C, H, W] -> [B, C, H, W]
                mask = np.broadcast_to(mask[np.newaxis, :, :, :], shape).copy()
        return G, mask
    else:
        G = generator.generate_g_field(g_field_shape, seeds, return_mask=False)
        # Reshape to original shape if needed
        if len(shape) == 4:
            G = np.broadcast_to(G[np.newaxis, :, :, :], shape)
        return G


def compute_g_expected_flat(
    master_key: str,
    key_id: str,
    n_elements: int,
    prf_algorithm: str = "chacha20",
    mapping_mode: str = "rademacher",
) -> np.ndarray:
    """
    Compute flat (1D) expected G-field.
    
    More efficient when only the dot-product matters.
    
    Args:
        master_key: Secret master key
        key_id: Public key identifier
        n_elements: Number of elements
        prf_algorithm: PRF algorithm
        mapping_mode: Value mapping mode
    
    Returns:
        1D numpy array of g-values
    """
    from ..detection.prf import PRFKeyDerivation
    from ..core.config import PRFConfig
    
    # Generate PRF seeds
    prf_config = PRFConfig(algorithm=prf_algorithm)
    prf = PRFKeyDerivation(master_key, prf_config)
    seeds = prf.generate_seeds(key_id, n_elements)
    
    # Create generator
    generator = GFieldGenerator(mapping_mode=mapping_mode)
    
    # Generate flat G-field (treat as single channel)
    shape = (1, 1, n_elements)
    G = generator.generate_g_field(shape, seeds)
    
    # Flatten
    return G.flatten()


# Backward compatibility: keep old function name as alias
apply_frequency_mask = apply_frequency_bandpass
