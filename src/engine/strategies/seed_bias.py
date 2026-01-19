"""
Seed Bias (Latent Initialization) Watermarking Strategy.

This strategy implements latent-initialization watermarking by injecting
a PRF-derived watermark pattern into the initial Gaussian noise z_T.

Mathematical Formulation:
    z_T = sqrt(1 - lambda^2) * epsilon + lambda * G
    
Where:
    - epsilon ~ N(0, I) is the random seed noise
    - G is the PRF-derived watermark field (normalized to zero-mean, unit-variance)
    - lambda is the bias strength (0 <= lambda < 1)

This preserves unit variance of z_T while injecting the watermark signal.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ...algorithms.g_field import GFieldGenerator
from ...core.config import SeedBiasConfig
from ...core.interfaces import WatermarkStrategy
from ...detection.prf import PRFKeyDerivation


class SeedBiasStrategy(WatermarkStrategy):
    """
    Watermarking strategy that initializes the diffusion process with a biased latent z_T.
    
    This strategy overrides the standard torch.randn initialization by mixing
    the random noise with a PRF-derived watermark pattern. The mixing uses
    spherical linear interpolation to preserve unit variance.
    
    Key Features:
    - Variance-preserving: z_T maintains unit variance
    - Frequency-filtered: G-field is low-to-mid frequency biased for robustness
    - PRF-based: Uses secure PRF for deterministic G-field generation
    - Deterministic: Same seed + key_id yields identical z_T
    """

    def __init__(
        self,
        config: SeedBiasConfig,
        master_key: str,
        latent_shape: Tuple[int, int, int] = (4, 64, 64),
        device: str = "cuda",
    ):
        """
        Initialize seed bias strategy.
        
        This strategy overrides the standard torch.randn initialization by
        mixing random noise with a PRF-derived watermark pattern.
        
        Warning:
            If lambda_strength > 0.15, visible artifacts are likely.
            Consider using lower values (0.05-0.10) for high-fidelity generation.
        
        Args:
            config: SeedBiasConfig with lambda_strength, frequency cutoffs, etc.
            master_key: Master key for PRF-based key derivation
            latent_shape: Latent tensor shape (C, H, W)
            device: Device to use for tensors
        """
        self.config = config
        self.master_key = master_key
        self.latent_shape = latent_shape
        self.device = device
        
        # Verify non-distortionary: warn if lambda is too high
        if config.lambda_strength > 0.15:
            import warnings
            warnings.warn(
                f"lambda_strength={config.lambda_strength} > 0.15 may cause "
                "visible artifacts. Consider using lower values (0.05-0.10) "
                "for high-fidelity generation.",
                UserWarning,
            )
        
        # Create PRF key derivation
        from ...core.config import PRFConfig
        prf_config = PRFConfig()  # Use defaults
        self.prf = PRFKeyDerivation(master_key, prf_config)
        
        # Create G-field generator
        self.g_field_generator = GFieldGenerator(
            mapping_mode="binary",  # Use binary for seed bias
            domain=config.domain,
            frequency_mode="bandpass",  # Use bandpass for seed bias
            low_freq_cutoff=config.low_freq_cutoff,
            high_freq_cutoff=config.high_freq_cutoff,
            normalize_zero_mean=True,
            normalize_unit_variance=True,
        )
        
        # State for current sample
        self._current_sample_id: Optional[str] = None
        self._current_prompt: Optional[str] = None
        self._current_seed: Optional[int] = None
        self._current_key_id: Optional[str] = None
        self._current_z_T: Optional[torch.Tensor] = None

    def get_initial_latent(
        self,
        shape: Tuple[int, int, int],
        seed: int,
        key_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate initial latent z_T with watermark bias.
        
        This method implements the core seed bias algorithm:
        1. Generate random noise epsilon using the provided seed
        2. Generate G-field using PRF-based key derivation
        3. Apply frequency filtering to G
        4. Normalize G to zero-mean, unit-variance
        5. Mix: z_T = sqrt(1 - lambda^2) * epsilon + lambda * G
        
        Args:
            shape: Latent shape (C, H, W)
            seed: Random seed for epsilon generation
            key_id: Public key identifier for PRF (if None, uses sample_id)
        
        Returns:
            Initial latent tensor z_T [1, C, H, W] on self.device
        """
        if key_id is None:
            key_id = self._current_key_id or self._current_sample_id or "default"
        
        # Step 1: Generate random noise epsilon
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        C, H, W = shape
        epsilon = torch.randn(
            (1, C, H, W),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        
        # Step 2: Generate G-field using PRF
        # IMPORTANT: Watermark key is based ONLY on key_id (not seed)
        # This allows detection without knowing the generation seed
        # The seed only affects diffusion randomness, not the watermark pattern
        prf_key_id = key_id
        num_elements = C * H * W
        prf_seeds = self.prf.generate_seeds(prf_key_id, num_elements)
        
        # Generate G-field with frequency filtering built-in
        # Update generator config for bandpass mode
        self.g_field_generator.frequency_mode = "bandpass"
        self.g_field_generator.high_freq_cutoff = self.config.high_freq_cutoff
        
        G_np = self.g_field_generator.generate_g_field(
            shape=shape,
            seeds=prf_seeds,
        )
        
        # Convert to torch tensor
        G = torch.from_numpy(G_np).float().to(self.device)
        G = G.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
        
        # Step 5: Spherical mixing
        lambda_val = self.config.lambda_strength
        sqrt_term = np.sqrt(1.0 - lambda_val**2)
        z_T = sqrt_term * epsilon + lambda_val * G
        
        return z_T

    def get_hook(self) -> Optional[Any]:
        """
        Returns None - seed bias strategy doesn't use hooks.
        
        This strategy modifies the initial latent, not the denoising process.
        The hook-based injection is handled by LatentInjectionStrategy.
        """
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns watermark embedding metadata for manifest.
        
        This method returns ONLY embedding-related metadata. Detection configuration
        is handled separately by the detection service and is not included here.
        
        Returns:
            Dictionary containing embedding metadata:
            - watermark_version: Schema version for seed-bias embedding
            - lambda_strength: Watermark injection strength
            - domain: Generation domain (frequency/spatial)
            - frequency cutoffs: Low and high frequency cutoff values
            - key_id: Public key identifier for PRF
            - sample_id: Unique sample identifier
        """
        return {
            "watermark_version": "seed_bias_v1",
            "mode": "seed_bias",
            "sample_id": self._current_sample_id,
            "sample_seed": self._current_seed,
            "key_id": self._current_key_id,
            "lambda_strength": self.config.lambda_strength,
            "domain": self.config.domain,
            "low_freq_cutoff": self.config.low_freq_cutoff,
            "high_freq_cutoff": self.config.high_freq_cutoff,
        }

    def prepare_for_sample(
        self,
        sample_id: str,
        prompt: str,
        seed: Optional[int] = None,
        key_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Prepare strategy for a new sample generation.
        
        Args:
            sample_id: Unique sample identifier
            prompt: Text prompt
            seed: Random seed (optional)
            key_id: Public key identifier for PRF (optional)
            **kwargs: Additional parameters
        """
        self._current_sample_id = sample_id
        self._current_prompt = prompt
        self._current_seed = seed
        self._current_key_id = key_id or sample_id

