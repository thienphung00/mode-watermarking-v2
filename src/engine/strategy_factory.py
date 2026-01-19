"""
Strategy factory for creating watermark strategies from configuration.

This module provides helper functions to instantiate the appropriate
WatermarkStrategy based on configuration.

Strategy 1 (Pre-Scheduler Injection):
    The factory creates strategies that inject watermark bias BEFORE the scheduler step:
    
    noise_pred = unet(latents, t, encoder_hidden_states=...)
    noise_pred = noise_pred + alpha_t * G_t     # Inject HERE
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    Key components:
    - TimestepMapper: Maps DDIM step_index → trained_timestep for G_t/alpha_t lookup
    - WatermarkHook: Performs IN-PLACE modification of noise_pred before scheduler
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..algorithms.g_field import GFieldGenerator
from ..algorithms.masks import MaskGenerator
from ..algorithms.scheduling import AlphaScheduler, compute_g_field_energy
from ..core.config import WatermarkedConfig, UnwatermarkedConfig, PRFConfig
from ..core.interfaces import (
    WatermarkStrategy,
    NullStrategy,
    LatentInjectionStrategy,
)
from ..detection.prf import PRFKeyDerivation
from .hooks import TimestepMapper
from .strategies.seed_bias import SeedBiasStrategy


def create_strategy_from_config(
    config: WatermarkedConfig | UnwatermarkedConfig,
    diffusion_config: any,
    device: str = "cuda",
    initial_sample_id: str = "default_sample",
) -> WatermarkStrategy:
    """
    Create appropriate watermark strategy from configuration.

    Args:
        config: Watermark configuration (watermarked or unwatermarked)
        diffusion_config: Diffusion configuration for timestep mapping
        device: Device to use for tensors
        initial_sample_id: Initial sample ID for G-field generation

    Returns:
        WatermarkStrategy instance (NullStrategy or LatentInjectionStrategy)
    """
    if isinstance(config, UnwatermarkedConfig):
        return NullStrategy()

    # Check if seed bias strategy is requested
    if isinstance(config, WatermarkedConfig):
        if config.algorithm_params.seed_bias is not None:
            return create_seed_bias_strategy(
                config, diffusion_config, device, initial_sample_id
            )
    
    # Watermarked mode - build full strategy (default: LatentInjectionStrategy)
    return create_watermark_strategy(config, diffusion_config, device, initial_sample_id)


def create_watermark_strategy(
    config: WatermarkedConfig,
    diffusion_config: any,
    device: str = "cuda",
    initial_sample_id: str = "default_sample",
) -> LatentInjectionStrategy:
    """
    Create watermark strategy for watermarked mode (Strategy 1).

    This function:
    1. Creates generators (G-field generator, alpha scheduler, mask generator)
    2. Creates base strategy without G-fields
    3. Calls create_per_sample_strategy() to generate initial G-fields

    Args:
        config: Watermarked configuration
        diffusion_config: Diffusion configuration
        device: Device to use (tensors will be preloaded here)
        initial_sample_id: Initial sample ID for G-field generation

    Returns:
        LatentInjectionStrategy instance with G-fields for initial_sample_id
    """
    # Create PRF key derivation (unified with detection system)
    prf_config = config.key_settings.prf_config
    prf = PRFKeyDerivation(config.key_settings.key_master, prf_config)

    # Create timestep mapper for DDIM step_index → trained_timestep conversion
    timestep_mapper = TimestepMapper(
        trained_timesteps=diffusion_config.trained_timesteps,
        inference_timesteps=diffusion_config.inference_timesteps,
        discretization="uniform",
    )

    # Get all trained timesteps for G-field generation
    trained_timesteps = timestep_mapper.get_all_trained_timesteps()

    # Create G-field generator
    g_field_config = config.algorithm_params.g_field
    g_gen = GFieldGenerator(
        mapping_mode=g_field_config.mapping_mode,
        domain=g_field_config.domain,
        frequency_mode=g_field_config.frequency_mode,
        low_freq_cutoff=g_field_config.low_freq_cutoff,
        normalize_zero_mean=g_field_config.normalize.get("zero_mean_per_timestep", True),
        normalize_unit_variance=g_field_config.normalize.get("unit_variance", False),
        continuous_range=g_field_config.continuous_range,
    )

    # Calculate required stream length
    shape_tuple = tuple(g_field_config.shape)
    elements_per_gfield = shape_tuple[0] * shape_tuple[1] * shape_tuple[2]
    num_timesteps = len(trained_timesteps)
    required_stream_len = num_timesteps * elements_per_gfield * 2  # 2x safety margin

    # Create alpha scheduler
    bias_config = config.algorithm_params.bias
    alpha_scheduler = AlphaScheduler(
        mode=bias_config.mode,
        target_snr=bias_config.target_snr,
        alpha_bounds=tuple(bias_config.alpha_bounds),
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end,
        num_diffusion_steps=diffusion_config.trained_timesteps,
    )

    # Create mask generator and generate mask (mask is sample-independent)
    mask_config = config.algorithm_params.mask
    mask_gen = MaskGenerator(
        mode=mask_config.mode,
        strength=mask_config.strength,
        band=mask_config.band,
        cutoff_freq=mask_config.cutoff_freq,
        bandwidth_fraction=mask_config.bandwidth_fraction,
    )
    mask = mask_gen.generate(shape_tuple)

    # Create base strategy with empty schedules (will be filled by create_per_sample_strategy)
    strategy = LatentInjectionStrategy(
        config=config,
        g_schedule={},  # Empty - will be populated
        alpha_schedule={},  # Empty - will be populated
        mask=mask,
        key_provider=prf,  # PRF instead of KeyDerivation
        timestep_mapper=timestep_mapper,
        device=device,
    )

    # Store generation context in strategy for per-sample regeneration
    strategy._g_field_generator = g_gen
    strategy._alpha_scheduler = alpha_scheduler
    strategy._trained_timesteps = trained_timesteps
    strategy._g_field_shape = shape_tuple
    strategy._stream_length = required_stream_len

    # Generate initial G-fields using initial_sample_id as key_id
    # The actual key_id will be provided when generate_with_watermark is called
    create_per_sample_strategy(
        strategy=strategy,
        sample_id=initial_sample_id,
        prompt="",  # Initial prompt (will be overwritten)
        key_id=initial_sample_id,  # Use sample_id as key_id initially
    )

    return strategy


def create_per_sample_strategy(
    strategy: LatentInjectionStrategy,
    sample_id: str,
    prompt: str,
    key_id: Optional[str] = None,
) -> LatentInjectionStrategy:
    """
    Generate unique G-fields for a specific sample using PRF-based seed generation.

    This function ensures per-sample determinism for watermark detection:
    1. Uses key_id (public identifier) for PRF seed generation
    2. Generates PRF seeds: PRF(master_key, key_id, index)
    3. Generates G-field schedule from PRF seeds
    4. Generates alpha schedule based on G-field energies
    5. Updates strategy with new schedules and metadata
    6. Invalidates cached hook (it uses old schedules)

    The key_id is critical for detection reproducibility:
    - Same key_id → same PRF seeds → same G-fields
    - This allows detection to regenerate identical G-fields for correlation
    - Detection uses only key_id (no metadata like zT_hash, sample_id, base_seed)

    Args:
        strategy: Strategy instance with generators stored
        sample_id: Unique sample identifier (for metadata only)
        prompt: Text prompt
        key_id: Public key identifier. If None, uses sample_id as key_id.

    Returns:
        Same strategy instance, now configured with sample-specific G-fields
    """
    config = strategy.config
    prf = strategy.key_provider  # PRFKeyDerivation instance

    # Use key_id if provided, otherwise use sample_id
    if key_id is None:
        key_id = sample_id

    # Check if regeneration is needed (skip if key_id matches)
    if (strategy._current_key_id == key_id):
        # Only update prompt if it changed (metadata update)
        if strategy._current_prompt != prompt:
            strategy._current_prompt = prompt
        return strategy

    # Generate PRF seeds for all timesteps
    C, H, W = strategy._g_field_shape
    elements_per_gfield = C * H * W
    num_timesteps = len(strategy._trained_timesteps)
    total_elements = num_timesteps * elements_per_gfield
    
    # Generate all PRF seeds at once
    prf_seeds = prf.generate_seeds(key_id, total_elements)

    # Generate G-field schedule from PRF seeds
    g_schedule = strategy._g_field_generator.generate_schedule(
        shape=strategy._g_field_shape,
        timesteps=strategy._trained_timesteps,
        seeds=prf_seeds,
    )

    # Compute G-field energies for alpha schedule
    g_field_energies = {t: compute_g_field_energy(G_t) for t, G_t in g_schedule.items()}

    # Generate alpha schedule
    bias_config = config.algorithm_params.bias
    envelope_config = bias_config.injection if hasattr(bias_config, 'injection') else None
    
    alpha_schedule = strategy._alpha_scheduler.generate_schedule(
        timesteps=strategy._trained_timesteps,
        g_field_energies=g_field_energies,
        latent_shape=strategy._g_field_shape,
        envelope_config=envelope_config,
    )

    # Update strategy with new schedules
    strategy.g_schedule = g_schedule
    strategy.alpha_schedule = alpha_schedule

    # Update sample metadata
    strategy._current_sample_id = sample_id
    strategy._current_prompt = prompt
    strategy._current_key_id = key_id  # Critical for detection (PRF-based)

    # Invalidate cached hook (it references old schedules)
    strategy._hook = None

    return strategy


def create_seed_bias_strategy(
    config: WatermarkedConfig,
    diffusion_config: any,
    device: str = "cuda",
    initial_sample_id: str = "default_sample",
) -> SeedBiasStrategy:
    """
    Create seed bias strategy for watermarked mode.
    
    This strategy initializes the diffusion process with a biased latent z_T
    by mixing random noise with a PRF-derived watermark pattern.
    
    Args:
        config: Watermarked configuration (must have algorithm_params.seed_bias)
        diffusion_config: Diffusion configuration
        device: Device to use for tensors
        initial_sample_id: Initial sample ID
    
    Returns:
        SeedBiasStrategy instance
    """
    seed_bias_config = config.algorithm_params.seed_bias
    if seed_bias_config is None:
        raise ValueError(
            "seed_bias config is required for SeedBiasStrategy. "
            "Set algorithm_params.seed_bias in config."
        )
    
    # Get latent shape from g_field config
    g_field_config = config.algorithm_params.g_field
    latent_shape = tuple(g_field_config.shape)
    
    # Create strategy
    strategy = SeedBiasStrategy(
        config=seed_bias_config,
        master_key=config.key_settings.key_master,
        latent_shape=latent_shape,
        device=device,
    )
    
    # Prepare for initial sample
    strategy.prepare_for_sample(
        sample_id=initial_sample_id,
        prompt="",
        key_id=initial_sample_id,
    )
    
    return strategy
