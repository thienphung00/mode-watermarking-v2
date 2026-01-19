"""
Hybrid Detection Pipeline for Seed Bias Watermarking.

Implements a cascade detection system that balances speed and accuracy:
- Stage 1 (Fast Pass): Quick correlation check on z_0 (VAE-encoded latent)
- Stage 2 (Accurate Pass): Full DDIM inversion to z_T for better signal recovery

This pipeline is specifically designed for seed bias watermarking, where
the watermark is injected into the initial latent z_T.

Architecture:
- HybridDetector is a pure orchestrator that delegates:
  - Image encoding to detection/inversion.py
  - Signal extraction to detection/observe.py
  - Statistics computation to detection/statistics.py
- Context-aware signal extraction:
  - Stage 1: extract_whitened(z_0) - high-pass filter to suppress image content
  - Stage 2: extract_normalized(z_T) - zero-mean/unit-variance for noise-like latents
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .inversion import DDIMInverter
from .g_values import compute_g_values
from .prf import PRFKeyDerivation
from .statistics import compute_s_statistic
from ..core.config import PRFConfig


class HybridDetector:
    """
    Hybrid detection pipeline for seed bias watermarking.
    
    Pure orchestrator that delegates all low-level operations:
    - Image encoding/inversion → detection/inversion.py
    - Signal extraction → detection/observe.py
    - Statistics computation → detection/statistics.py
    
    Implements cascade detection:
    1. Fast Pass: Extract whitened signal from z_0 (VAE-encoded), correlate with G-field
    2. Accurate Pass (if ambiguous): Full inversion to z_T, extract normalized signal, correlate
    
    This balances speed (Stage 1 is fast) with accuracy (Stage 2 recovers
    the exact initial latent where watermark was embedded).
    """

    def __init__(
        self,
        pipeline: Any,  # StableDiffusionPipeline
        master_key: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        g_field_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid detector.
        
        Args:
            pipeline: Stable Diffusion pipeline (for VAE, UNet, scheduler)
            master_key: Master key for PRF-based G-field generation
            config: Detection configuration dict with:
                - detection_mode: "fast_only", "hybrid", or "full_inversion"
                - num_inference_steps: Number of steps for full inversion (default: 50)
                - threshold_high: High threshold for g-value mean (default: 0.52)
                - threshold_low: Low threshold for g-value mean (default: 0.48)
            device: Device for computation
            g_field_config: REQUIRED G-field configuration dict. Must match the config
                used during generation. This prevents configuration drift.
        """
        self.pipeline = pipeline
        self.master_key = master_key
        self.device = device or str(pipeline.device)
        
        # Store g_field_config (required for compute_g_values)
        if g_field_config is None:
            raise ValueError(
                "g_field_config is REQUIRED. It must match the configuration "
                "used during generation to ensure consistent g-value computation."
            )
        self.g_field_config = g_field_config
        
        # Default configuration
        # Thresholds are for mean(g-values), which should be ~0.5 for unwatermarked
        # and >0.5 for watermarked (e.g., 0.52+ indicates watermark presence)
        self.config = {
            "detection_mode": "hybrid",
            "num_inference_steps": 50,  # Full inversion uses full timeline
            "threshold_high": 0.52,  # Mean g-value threshold for detection
            "threshold_low": 0.48,   # Mean g-value threshold for rejection
            **(config or {}),
        }
        
        # Initialize PRF for G-field generation
        self.prf = PRFKeyDerivation(master_key, PRFConfig())
        
        # Create DDIM inverter for Stage 2 (full inversion)
        self.inverter = DDIMInverter(pipeline, device=self.device)

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        key_id: str,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect watermark in image using hybrid cascade approach.
        
        Args:
            image: Input image (PIL Image)
            key_id: Public key identifier for PRF-based G-field generation
            seed: Optional seed (if None, will try to infer from key_id)
            prompt: Optional prompt used during generation (for accurate inversion)
            guidance_scale: Optional guidance scale used during generation (for accurate inversion)
            num_inference_steps: Optional number of inference steps used during generation
        
        Returns:
            Dictionary with:
                - detected: bool (watermark detected or not)
                - stage: str ("fast" or "accurate")
                - score: float (correlation score)
                - threshold_high: float
                - threshold_low: float
                - metadata: dict with additional info
        """
        # Check if full_inversion mode (skip Stage 1)
        if self.config["detection_mode"] == "full_inversion":
            # Encode image to get z_0 (needed for Stage 2)
            allow_resize = False  # Require exact 512x512
            z_0 = self.inverter.encode_image(image, allow_resize=allow_resize)
            
            # Skip Stage 1, go directly to Stage 2
            result_stage2 = self._stage2_accurate_pass(
                image,
                key_id,
                seed,
                z_0,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            return {
                "detected": result_stage2["detected"],
                "stage": "accurate",
                "score": result_stage2["score"],
                "threshold_high": self.config["threshold_high"],
                "threshold_low": self.config["threshold_low"],
                "metadata": {
                    "stage1_score": None,  # Stage 1 skipped
                    "stage2_score": result_stage2["score"],
                    "stage1_conclusive": None,  # Stage 1 skipped
                    "inversion_steps": result_stage2["debug"]["inversion_steps"],
                    "inversion_params": result_stage2["debug"],
                    "note": "full_inversion mode: Stage 1 skipped",
                },
            }
        
        # Stage 1: Fast Pass (z_0 correlation)
        result_stage1 = self._stage1_fast_pass(image, key_id, seed)
        
        # Check if Stage 1 is conclusive
        if result_stage1["conclusive"]:
            return {
                "detected": result_stage1["detected"],
                "stage": "fast",
                "score": result_stage1["score"],
                "threshold_high": self.config["threshold_high"],
                "threshold_low": self.config["threshold_low"],
                "metadata": {
                    "stage1_score": result_stage1["score"],
                    "stage1_conclusive": True,
                },
            }
        
        # Stage 2: Accurate Pass (full inversion)
        if self.config["detection_mode"] == "fast_only":
            # Fast-only mode: return ambiguous result
            return {
                "detected": False,  # Conservative: don't detect if ambiguous
                "stage": "fast",
                "score": result_stage1["score"],
                "threshold_high": self.config["threshold_high"],
                "threshold_low": self.config["threshold_low"],
                "metadata": {
                    "stage1_score": result_stage1["score"],
                    "stage1_conclusive": False,
                    "note": "Fast-only mode: ambiguous result",
                },
            }
        
        # Run Stage 2: Full inversion to z_T with generation parameters
        result_stage2 = self._stage2_accurate_pass(
            image,
            key_id,
            seed,
            result_stage1["z_0"],
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        
        return {
            "detected": result_stage2["detected"],
            "stage": "accurate",
            "score": result_stage2["score"],
            "threshold_high": self.config["threshold_high"],
            "threshold_low": self.config["threshold_low"],
            "metadata": {
                "stage1_score": result_stage1["score"],
                "stage2_score": result_stage2["score"],
                "stage1_conclusive": False,
                "inversion_steps": result_stage2["debug"]["inversion_steps"],
                "inversion_params": result_stage2["debug"],
            },
        }

    def _stage1_fast_pass(
        self,
        image: Image.Image,
        key_id: str,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """
        Stage 1: Fast pass using z_0 (VAE-encoded latent).
        
        Uses key-dependent g-value computation via compute_g_values().
        Then computes S-statistic correlation for detection.
        
        Args:
            image: Input image
            key_id: Key identifier
            seed: Optional seed (ignored, kept for API compatibility)
        
        Returns:
            Dictionary with:
                - score: float (S-statistic score)
                - detected: bool (if conclusive)
                - conclusive: bool (whether result is conclusive)
                - z_0: torch.Tensor (encoded latent for Stage 2)
        """
        # Delegate image encoding to inversion module
        # For fast_only mode: allow_resize=True (convenience)
        # For hybrid/full_inversion: allow_resize=False (require exact 512x512)
        allow_resize = (self.config["detection_mode"] == "fast_only")
        z_0 = self.inverter.encode_image(image, allow_resize=allow_resize)
        
        # Compute g-values using canonical function
        # This generates expected G-field from key and compares against observed latent
        g, mask = compute_g_values(
            z_0,
            key_id,
            self.master_key,
            return_mask=True,
            g_field_config=self.g_field_config,
        )
        
        # For correlation-based detection, we still need the expected G-field
        # and observed signal. Extract them from the g-value computation.
        # Note: We regenerate the G-field here to use it for S-statistic computation.
        # This uses self.g_field_config to ensure consistency with generation.
        
        # Generate expected G-field for S-statistic computation using g_field_config
        C, H, W = z_0.shape[1:]  # Remove batch dimension
        shape = (C, H, W)
        num_elements = C * H * W
        prf_seeds = self.prf.generate_seeds(key_id, num_elements)
        
        from ..algorithms.g_field import GFieldGenerator
        
        # Use self.g_field_config instead of hardcoded values
        g_gen = GFieldGenerator(**self.g_field_config)
        
        G_expected_np = g_gen.generate_g_field(shape, prf_seeds)
        
        # For correlation, we need the observed signal (not just binary g-values)
        # Use sign of z_0 as observed signal for correlation
        z_0_np = z_0.detach().cpu().numpy()
        if z_0_np.ndim == 4:
            z_0_np = z_0_np[0]  # Remove batch dimension
        
        # Use sign of latent as observed signal
        g_observed_np = np.sign(z_0_np)
        
        # Compute S-statistic using correlation
        score = compute_s_statistic(g_observed_np, G_expected_np)
        
        # Check if conclusive
        threshold_high = self.config["threshold_high"]
        threshold_low = self.config["threshold_low"]
        
        detected = score > threshold_high
        conclusive = (score > threshold_high) or (score < threshold_low)
        
        return {
            "score": float(score),
            "detected": detected,
            "conclusive": conclusive,
            "z_0": z_0,
        }

    def _stage2_accurate_pass(
        self,
        image: Image.Image,
        key_id: str,
        seed: Optional[int],
        z_0: torch.Tensor,
        prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Accurate pass using full DDIM inversion.
        
        Performs full DDIM inversion to recover the initial latent z_T,
        which is the exact point where the SeedBias watermark was embedded.
        
        Uses context-aware signal extraction:
        - z_T is noise-like (image structure removed)
        - Apply normalized extraction (zero-mean/unit-variance) to preserve signal energy
        
        CRITICAL: Inversion must use the same parameters as generation:
        - Same num_inference_steps (scheduler timesteps)
        - Same guidance_scale
        - Same prompt (or empty for unconditional)
        - Same scheduler type and settings
        
        Args:
            image: Input image (unused, kept for API compatibility)
            key_id: Key identifier
            seed: Optional seed
            z_0: VAE-encoded latent from Stage 1
            prompt: Optional prompt used during generation (default: empty string for unconditional)
            guidance_scale: Optional guidance scale used during generation (default: 1.0 for unconditional)
            num_inference_steps: Optional number of inference steps (default: from config or 50)
        
        Returns:
            Dictionary with:
                - score: float (S-statistic score)
                - detected: bool
                - z_T: torch.Tensor (recovered initial latent)
                - conclusive: bool (always True for Stage 2)
                - debug: dict with inversion parameters used
        """
        # Use generation parameters if provided, otherwise use defaults
        # Defaults assume unconditional generation (prompt="", guidance_scale=1.0)
        inversion_prompt = prompt if prompt is not None else ""
        inversion_guidance_scale = guidance_scale if guidance_scale is not None else 1.0
        inversion_steps = num_inference_steps if num_inference_steps is not None else self.config["num_inference_steps"]
        
        # Perform full DDIM inversion to recover z_T
        # This uses ALL inference steps (full inversion, not partial)
        z_T = self.inverter.perform_full_inversion(
            z_0,
            num_inference_steps=inversion_steps,
            prompt=inversion_prompt,
            guidance_scale=inversion_guidance_scale,
        )
        
        # Compute g-values using canonical function
        # This generates expected G-field from key and compares against observed latent
        g, mask = compute_g_values(
            z_T,
            key_id,
            self.master_key,
            return_mask=True,
            g_field_config=self.g_field_config,
        )
        
        # For correlation-based detection, we still need the expected G-field
        # and observed signal. Extract them from the g-value computation.
        # Note: We regenerate the G-field here to use it for S-statistic computation.
        # This uses self.g_field_config to ensure consistency with generation.
        
        # Generate expected G-field for S-statistic computation using g_field_config
        C, H, W = z_T.shape[1:]
        shape = (C, H, W)
        num_elements = C * H * W
        prf_seeds = self.prf.generate_seeds(key_id, num_elements)
        
        from ..algorithms.g_field import GFieldGenerator
        
        # Use self.g_field_config instead of hardcoded values
        g_gen = GFieldGenerator(**self.g_field_config)
        
        G_expected_np = g_gen.generate_g_field(shape, prf_seeds)
        
        # For correlation, we need the observed signal (not just binary g-values)
        # Use sign of z_T as observed signal for correlation
        z_T_np = z_T.detach().cpu().numpy()
        if z_T_np.ndim == 4:
            z_T_np = z_T_np[0]  # Remove batch dimension
        
        # Use sign of latent as observed signal
        g_observed_np = np.sign(z_T_np)
        
        # Compute S-statistic using correlation
        score = compute_s_statistic(g_observed_np, G_expected_np)
        
        threshold_high = self.config["threshold_high"]
        detected = score > threshold_high
        
        return {
            "score": float(score),
            "detected": detected,
            "z_T": z_T,
            "conclusive": True,  # Stage 2 is always conclusive
            "debug": {
                "inversion_steps": inversion_steps,
                "inversion_prompt": inversion_prompt,
                "inversion_guidance_scale": inversion_guidance_scale,
            },
        }



def detect_seed_bias_watermark(
    image: Image.Image,
    key_id: str,
    master_key: str,
    pipeline: Any,
    g_field_config: Dict[str, Any],
    detection_mode: Literal["fast_only", "hybrid", "full_inversion"] = "hybrid",
    num_inference_steps: int = 50,
    threshold_high: float = 0.15,
    threshold_low: float = 0.05,
    seed: Optional[int] = None,
    prompt: Optional[str] = None,
    guidance_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for seed bias watermark detection.
    
    Args:
        image: Input image (PIL Image)
        key_id: Public key identifier
        master_key: Master key for PRF
        pipeline: Stable Diffusion pipeline
        g_field_config: REQUIRED G-field configuration dict. Must match the config
            used during generation. This prevents configuration drift.
        detection_mode: Detection mode ("fast_only", "hybrid", "full_inversion")
        num_inference_steps: Number of steps for full inversion (default: 50)
        threshold_high: High threshold for detection
        threshold_low: Low threshold for rejection
        seed: Optional seed
        prompt: Optional prompt used during generation (for accurate inversion)
        guidance_scale: Optional guidance scale used during generation (for accurate inversion)
    
    Returns:
        Detection result dictionary
    """
    config = {
        "detection_mode": detection_mode,
        "num_inference_steps": num_inference_steps,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
    }
    
    detector = HybridDetector(pipeline, master_key, config, g_field_config=g_field_config)
    return detector.detect(
        image,
        key_id,
        seed,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

