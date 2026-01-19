"""
SD Pipeline Factory and watermark hook context manager.

Implements Strategy 1 (Pre-Scheduler Injection) using UNet forward hook patching.

Strategy 1 Pipeline Flow:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  For each denoising step t:                                         │
    │                                                                     │
    │  1. noise_pred = unet(latents, t, encoder_hidden_states=...)        │
    │                      │                                              │
    │                      ▼                                              │
    │  2. noise_pred = strategy.apply_injection(...)  ◄── INJECT HERE    │
    │            (Strategy modifies in-place)                             │
    │                      │                                              │
    │                      ▼                                              │
    │  3. latents = scheduler.step(noise_pred, t, latents).prev_sample    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Key Implementation Details:
    - UNet forward is patched to intercept noise_pred BEFORE scheduler sees it
    - Strategy.apply_injection() modifies noise_pred IN-PLACE (no clone/copy)
    - Scheduler receives the MODIFIED noise prediction
    - Scheduler-agnostic: supports DDIM, UniPC, DPMSolver, etc.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch
from diffusers import StableDiffusionPipeline

from ..algorithms.scheduling import SchedulingFactory


# ============================================================================
# Latent Initialization Utilities
# ============================================================================


def prepare_initial_latents(
    pipeline: StableDiffusionPipeline,
    batch_size: int = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Prepare initial latent tensor z_T for generation.
    
    Args:
        pipeline: SD pipeline (for device, config, scale factors)
        batch_size: Number of latent samples to generate
        height: Image height (default: pipeline default)
        width: Image width (default: pipeline default)
        generator: Optional torch.Generator for reproducibility
        dtype: Tensor dtype (default: pipeline.unet.dtype)
    
    Returns:
        Initial latent tensor [B, C, H_latent, W_latent]
    """
    # Determine dimensions
    if height is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    if width is None:
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    if dtype is None:
        dtype = pipeline.unet.dtype
    
    # Calculate latent dimensions
    latent_height = height // pipeline.vae_scale_factor
    latent_width = width // pipeline.vae_scale_factor
    latent_channels = pipeline.unet.config.in_channels
    
    # Generate latent tensor
    latent_shape = (batch_size, latent_channels, latent_height, latent_width)
    z_T = torch.randn(
        latent_shape,
        generator=generator,
        device=pipeline.device,
        dtype=dtype,
    )
    
    return z_T


def compute_zT_hash(z_T: torch.Tensor) -> str:
    """
    Compute deterministic hash of initial latent tensor for metadata logging.
    
    This hash is used for metadata only. Detection uses PRF-based key derivation
    (key_id) and does not require zT_hash.
    
    Args:
        z_T: Initial latent tensor [1, C, H, W] (batch_size must be 1)
    
    Returns:
        32-character hex string hash
    
    Raises:
        ValueError: If batch_size != 1
    """
    import hashlib
    
    if z_T.dim() != 4:
        raise ValueError(f"z_T must be 4D [B, C, H, W], got shape {list(z_T.shape)}")
    
    if z_T.shape[0] != 1:
        raise ValueError(
            f"z_T batch size must be 1 for unique hash, got {z_T.shape[0]}. "
            f"Process samples individually."
        )
    
    # Normalize tensor for deterministic byte representation
    z_T_normalized = z_T.detach().cpu().float().contiguous()
    z_T_bytes = z_T_normalized.numpy().tobytes()
    
    # Compute SHA-256 hash (truncated to 128 bits)
    hasher = hashlib.sha256()
    hasher.update(z_T_bytes)
    return hasher.hexdigest()[:32]


# ============================================================================
# Pipeline Factory
# ============================================================================


def create_pipeline(
    diffusion_config: Any,
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """
    Create Stable Diffusion pipeline from configuration.
    
    Uses SchedulingFactory for scheduler-agnostic initialization.

    Args:
        diffusion_config: DiffusionConfig Pydantic model
        device: Device to load pipeline on

    Returns:
        Initialized StableDiffusionPipeline
    """
    # Determine dtype (only use FP16 for CUDA; MPS and CPU use FP32)
    # Note: dtype is set via .to(device, dtype=...) to avoid constructor warnings
    if device == "cuda" and diffusion_config.use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load pipeline without dtype (avoids constructor warnings)
    pipeline = StableDiffusionPipeline.from_pretrained(
        diffusion_config.model_id,
    )

    # Move to device and set dtype
    pipeline = pipeline.to(device, dtype=dtype)

    # Create scheduler via factory (scheduler-agnostic)
    scheduler = SchedulingFactory.create(diffusion_config)
    pipeline.scheduler = scheduler

    # Enable gradient checkpointing if requested
    if diffusion_config.gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()

    return pipeline


# ============================================================================
# Watermark Hook Context Manager
# ============================================================================


@contextmanager
def apply_watermark_hook(
    pipeline: StableDiffusionPipeline,
    strategy: Any,
) -> Generator[StableDiffusionPipeline, None, None]:
    """
    Context manager for applying watermark hook to pipeline (Strategy 1).

    Strategy 1 (Pre-Scheduler Injection):
        Hooks into UNet forward method to inject watermark bias into noise predictions
        BEFORE the scheduler step. This ensures the scheduler operates on watermarked
        noise, not clean noise.

        Pipeline flow:
            UNet.forward() → strategy.apply_injection() → return to scheduler

    Critical requirements:
        - noise_pred is modified IN-PLACE to preserve tensor identity
        - NO cloning or copying of noise_pred before scheduler receives it
        - Bias is NEVER applied after the scheduler step

    Args:
        pipeline: SD pipeline to patch
        strategy: Watermark strategy (must implement apply_injection method)

    Yields:
        Pipeline with watermark hook applied

    Example:
        >>> with apply_watermark_hook(pipeline, strategy):
        ...     image = pipeline(prompt="cat", num_inference_steps=50)
    """
    # Check if strategy provides a hook (NullStrategy returns None)
    if hasattr(strategy, 'get_hook'):
        hook = strategy.get_hook()
        if hook is None:
            # No hook needed (e.g., unwatermarked mode)
            yield pipeline
            return
    else:
        yield pipeline
        return

    # Store original UNet forward method
    original_unet_forward = pipeline.unet.forward
    
    # Track step index during generation
    step_counter = [0]
    
    # Get expected channels from UNet config
    expected_channels = pipeline.unet.config.in_channels

    def hooked_unet_forward(
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        UNet forward with Strategy 1 watermark injection.
        
        This wraps the original UNet forward to inject watermark bias
        via strategy.apply_injection() BEFORE returning to the scheduler.
        """
        # Validate input channels
        if sample.shape[1] != expected_channels:
            raise ValueError(
                f"UNet input has {sample.shape[1]} channels, "
                f"expected {expected_channels}. Shape: {sample.shape}"
            )
        
        # Call original UNet forward
        noise_pred = original_unet_forward(
            sample,
            timestep,
            encoder_hidden_states,
            **kwargs
        )
        
        # Handle tuple output (some UNet versions return (noise_pred, hidden_states))
        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        
        # Handle UNetOutput object (diffusers returns named tuple/dataclass)
        if hasattr(noise_pred, 'sample'):
            noise_pred = noise_pred.sample
        
        # Validate output shape
        if noise_pred.shape != sample.shape:
            raise ValueError(
                f"UNet output shape {noise_pred.shape} doesn't match "
                f"input shape {sample.shape}."
            )
        
        # Convert timestep to int
        if isinstance(timestep, torch.Tensor):
            t_value = int(timestep.item() if timestep.numel() == 1 else timestep.flatten()[0].item())
        else:
            t_value = int(timestep)
        
        # Get current step index
        step_index = step_counter[0]
        
        # Delegate injection to strategy
        # Strategy handles: prediction_type, multi-scale masks, alpha schedule, G-field, in-place modification
        result = hook(
            step_index=step_index,
            timestep=t_value,
            latents=sample,
            noise_pred=noise_pred,
        )
        
        # Extract modified noise_pred from result
        if result is not None and isinstance(result, dict) and "noise_pred" in result:
            noise_pred = result["noise_pred"]
        
        # Validate output after injection
        if noise_pred.shape[1] != expected_channels:
            raise ValueError(
                f"Hook output has {noise_pred.shape[1]} channels, "
                f"expected {expected_channels}."
            )
        
        # Ensure contiguity for chunk operation in guidance
        if not noise_pred.is_contiguous():
            noise_pred = noise_pred.contiguous()
        
        # Increment step counter
        step_counter[0] += 1
        
        return noise_pred

    try:
        # Patch UNet forward
        pipeline.unet.forward = hooked_unet_forward
        yield pipeline
    finally:
        # Restore original UNet forward
        pipeline.unet.forward = original_unet_forward


# ============================================================================
# Generation Functions
# ============================================================================


def generate_with_watermark(
    pipeline: StableDiffusionPipeline,
    strategy: Any,
    prompt: str,
    sample_id: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    latents: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Generate image with watermark using strategy pattern.

    This function implements per-sample watermark generation:
    1. Generate or use provided initial latent tensor z_T
    2. Compute zT_hash for metadata logging
    3. Prepare strategy with sample-specific G-fields (using PRF-based key derivation)
    4. Apply watermark hook during generation
    5. Store zT_hash in metadata for logging

    Args:
        pipeline: SD pipeline
        strategy: Watermark strategy
        prompt: Text prompt
        sample_id: Unique sample identifier
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed (optional)
        latents: Pre-generated initial latents (optional)
        **kwargs: Additional generation parameters

    Returns:
        Dictionary containing:
            - image: Generated PIL Image
            - metadata: Watermark metadata (includes zT_hash for logging)
            - initial_latents: Initial z_T tensor
            - zT_hash: Hash for metadata logging
    """
    from ..core.interfaces import NullStrategy, LatentInjectionStrategy
    from .strategy_factory import create_per_sample_strategy

    # CRITICAL: Ensure seed is set for deterministic/randomized generation
    # If seed is None and strategy supports custom latent generation, randomize it
    # This ensures seed-bias strategy is always used when available
    if seed is None and hasattr(strategy, "get_initial_latent"):
        import random
        seed = random.randint(0, 2**31 - 1)
    
    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Prepare initial latents
    if latents is not None:
        z_T = latents.clone()
        if z_T.device != pipeline.device:
            z_T = z_T.to(pipeline.device)
    else:
        # Check if strategy provides get_initial_latent() (e.g., SeedBiasStrategy)
        # Always use strategy's custom generation if available (seed is now guaranteed non-None if needed)
        if hasattr(strategy, "get_initial_latent"):
            # Use strategy's custom initial latent generation
            height = kwargs.get("height")
            width = kwargs.get("width")
            if height is None:
                height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            if width is None:
                width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
            
            latent_height = height // pipeline.vae_scale_factor
            latent_width = width // pipeline.vae_scale_factor
            latent_channels = pipeline.unet.config.in_channels
            latent_shape = (latent_channels, latent_height, latent_width)
            
            # Get key_id from strategy if available
            key_id = None
            if hasattr(strategy, "_current_key_id"):
                key_id = strategy._current_key_id
            
            z_T = strategy.get_initial_latent(
                shape=latent_shape,
                seed=seed,
                key_id=key_id,
            )
        else:
            # Standard initialization
            height = kwargs.get("height")
            width = kwargs.get("width")
            z_T = prepare_initial_latents(
                pipeline=pipeline,
                batch_size=1,
                height=height,
                width=width,
                generator=generator,
            )

    # Compute zT_hash for metadata logging
    zT_hash = compute_zT_hash(z_T)

    # Prepare strategy with sample-specific G-fields (if watermarked)
    # Uses zT_hash as key_id for PRF-based G-field generation
    if isinstance(strategy, LatentInjectionStrategy):
        create_per_sample_strategy(
            strategy=strategy,
            sample_id=sample_id,
            prompt=prompt,
            key_id=zT_hash,  # Use zT_hash as key_id for PRF seed generation
        )

    # Generate with watermark hook
    import logging
    pipeline_logger = logging.getLogger(__name__)
    pipeline_logger.info(f"Stable Diffusion pipeline called with prompt: '{prompt}'")
    
    # Determine autocast settings based on UNet dtype
    unet_dtype = pipeline.unet.dtype
    use_autocast = (unet_dtype == torch.float16)
    
    # Log dtype and autocast status for debugging
    pipeline_logger.info(
        f"UNet dtype: {unet_dtype}, autocast enabled: {use_autocast}, device: {pipeline.device.type}"
    )
    
    with apply_watermark_hook(pipeline, strategy):
        # Wrap pipeline call in autocast to ensure dtype consistency
        # This ensures timestep embeddings are created in the same dtype as UNet weights
        with torch.autocast(
            device_type=pipeline.device.type,
            dtype=unet_dtype,
            enabled=use_autocast,
        ):
            result = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                latents=z_T,
                return_dict=True,
                output_type="pil",
                **kwargs,
            )

    # Build metadata
    metadata = strategy.get_metadata()
    metadata.update({
        "prompt": prompt,
        "sample_id": sample_id,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "zT_hash": zT_hash,
    })

    return {
        "image": result.images[0],
        "metadata": metadata,
        "initial_latents": z_T,
        "zT_hash": zT_hash,
    }


def batch_generate_with_watermark(
    pipeline: StableDiffusionPipeline,
    strategy: Any,
    prompts: list[str],
    sample_ids: list[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seeds: Optional[list[int]] = None,
    **kwargs,
) -> list[dict]:
    """
    Generate multiple watermarked images.
    
    Each sample gets its own unique G-fields via PRF-based key derivation.
    Processes samples sequentially (batch size = 1) for unique G-fields.

    Args:
        pipeline: SD pipeline
        strategy: Watermark strategy
        prompts: List of text prompts
        sample_ids: List of unique sample identifiers
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        seeds: Optional list of seeds (one per sample)
        **kwargs: Additional generation parameters

    Returns:
        List of generation result dictionaries

    Raises:
        ValueError: If prompts and sample_ids have different lengths
    """
    if len(prompts) != len(sample_ids):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match "
            f"number of sample_ids ({len(sample_ids)})"
        )
    
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError(
            f"Number of seeds ({len(seeds)}) must match "
            f"number of prompts ({len(prompts)})"
        )
    
    results = []
    for i, (prompt, sample_id) in enumerate(zip(prompts, sample_ids)):
        seed = seeds[i] if seeds is not None else None
        
        result = generate_with_watermark(
            pipeline=pipeline,
            strategy=strategy,
            prompt=prompt,
            sample_id=sample_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs,
        )
        results.append(result)
    
    return results


# ============================================================================
# Batch Utilities
# ============================================================================


def compute_zT_hash_batch(latents_batch: torch.Tensor) -> list[str]:
    """
    Compute zT_hash for a batch of latent tensors.
    
    Args:
        latents_batch: Batch of latent tensors [B, C, H, W]
    
    Returns:
        List of zT_hash strings, one per sample
    
    Raises:
        ValueError: If input is not 4-dimensional
    """
    if latents_batch.dim() != 4:
        raise ValueError(
            f"latents_batch must be 4D [B, C, H, W], got shape {list(latents_batch.shape)}"
        )
    
    batch_size = latents_batch.shape[0]
    hashes = []
    
    for i in range(batch_size):
        z_T_i = latents_batch[i:i+1]
        hash_i = compute_zT_hash(z_T_i)
        hashes.append(hash_i)
    
    return hashes


def prepare_latents_with_hashes(
    pipeline: StableDiffusionPipeline,
    batch_size: int = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, list[str]]:
    """
    Generate initial latent tensors and compute their zT_hashes.
    
    Args:
        pipeline: SD pipeline
        batch_size: Number of latent tensors to generate
        height: Image height (default: pipeline default)
        width: Image width (default: pipeline default)
        seed: Random seed for reproducibility
        dtype: Tensor dtype (default: pipeline.unet.dtype)
    
    Returns:
        Tuple of (latents, hashes)
    """
    # Create generator
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Generate latents
    latents = prepare_initial_latents(
        pipeline=pipeline,
        batch_size=batch_size,
        height=height,
        width=width,
        generator=generator,
        dtype=dtype,
    )
    
    # Compute hashes
    hashes = compute_zT_hash_batch(latents)
    
    return latents, hashes
