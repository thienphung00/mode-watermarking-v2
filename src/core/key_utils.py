"""
Centralized key derivation and cache key utilities.

This module provides a single source of truth for:
- Key fingerprint computation
- Latent cache key construction
- Key validation

All scripts MUST use these functions to ensure consistency.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

# Centralized constant for unwatermarked dummy key computation
# Used when computing g-values for unwatermarked samples where key_id is None
UNWATERMARKED_DUMMY_KEY = "__unwatermarked_dummy_key__"

from .config import (
    AppConfig,
    PRFConfig,
    WatermarkedConfig,
    compute_cache_key as _compute_cache_key,
    compute_key_fingerprint as _compute_key_fingerprint,
    extract_detector_geometry_signature,
    normalize_geometry_signature,
)

logger = logging.getLogger(__name__)


def derive_key_fingerprint(
    master_key: str,
    key_id: str,
    prf_config: Optional[PRFConfig] = None,
) -> str:
    """
    Returns deterministic short fingerprint used for:
    - cache paths
    - latent metadata validation
    
    This is the SINGLE SOURCE OF TRUTH for key fingerprint computation.
    All scripts must use this function, never recompute locally.
    
    Args:
        master_key: Secret master key
        key_id: Public key identifier
        prf_config: PRF configuration (default: ChaCha20 with 64-bit outputs)
        
    Returns:
        64-character hex string (SHA-256 hash)
    """
    return _compute_key_fingerprint(master_key, key_id, prf_config)


def build_latent_cache_key(
    image_id: str,
    config: AppConfig,
    num_inversion_steps: int,
    master_key: str,
    key_id: Optional[str] = None,
    guidance_scale: Optional[float] = None,
) -> str:
    """
    Deterministically builds the latent cache filename stem.
    
    Must include:
      - key_fingerprint
      - geometry hash
      - cfg hash
      - inversion steps
      - model id
      - guidance scale
      - inference steps
      - image index
    
    This is the SINGLE SOURCE OF TRUTH for cache key construction.
    No script may manually format cache filenames.
    
    Args:
        image_id: Image identifier (e.g., "sample_000001")
        config: AppConfig instance
        num_inversion_steps: Number of DDIM inversion steps
        master_key: Master key (required for watermarked configs)
        key_id: Key identifier (optional, defaults to config value)
        guidance_scale: Guidance scale (optional, defaults to config value)
        
    Returns:
        Deterministic cache key string
    """
    # Use guidance_scale from config if not provided
    if guidance_scale is None:
        guidance_scale = config.diffusion.guidance_scale
    
    # Get key_id from config if not provided
    if key_id is None and isinstance(config.watermark, WatermarkedConfig):
        key_id = config.watermark.key_settings.key_id
    
    # Use centralized compute_cache_key function
    cache_key = _compute_cache_key(
        image_id=image_id,
        config=config,
        num_inversion_steps=num_inversion_steps,
        master_key=master_key,
        key_id=key_id,
    )
    
    return cache_key


def compute_geometry_hash(config: AppConfig) -> str:
    """
    Compute deterministic hash of detector geometry signature.
    
    Args:
        config: AppConfig instance
        
    Returns:
        8-character hex hash
    """
    if not isinstance(config.watermark, WatermarkedConfig):
        return "unwatermarked"
    
    geometry_sig = extract_detector_geometry_signature(config.watermark)
    normalized_sig = normalize_geometry_signature(geometry_sig)
    sig_json = json.dumps(normalized_sig, sort_keys=True, separators=(',', ':'))
    sig_hash = hashlib.md5(sig_json.encode()).hexdigest()[:8]
    return sig_hash


def compute_config_hash(config: AppConfig) -> str:
    """
    Compute deterministic hash of configuration (excluding strength parameters).
    
    Args:
        config: AppConfig instance
        
    Returns:
        8-character hex hash
    """
    config_dict = config.model_dump(mode='json')
    if isinstance(config.watermark, WatermarkedConfig):
        # Remove strength parameters that don't affect geometry
        if 'algorithm_params' in config_dict.get('watermark', {}):
            algo_params = config_dict['watermark']['algorithm_params']
            if 'mask' in algo_params:
                algo_params['mask'].pop('strength', None)
    config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]
    return config_hash


def log_key_and_cache_identity(
    master_key: str,
    key_id: Optional[str],
    config: AppConfig,
    cache_root: Path,
    cache_namespace: str,
    num_inversion_steps: int,
) -> None:
    """
    Log key and cache identity at startup for diagnostics.
    
    This allows instant diagnosis when mismatch occurs.
    
    Args:
        master_key: Master key
        key_id: Key identifier (None for unwatermarked)
        config: AppConfig instance
        cache_root: Root cache directory
        cache_namespace: Cache namespace/subdirectory
        num_inversion_steps: Number of inversion steps
    """
    logger.info("[KEY]")
    
    if isinstance(config.watermark, WatermarkedConfig):
        if key_id is None:
            key_id = config.watermark.key_settings.key_id
        prf_config = config.watermark.key_settings.prf_config
        
        key_fingerprint = derive_key_fingerprint(master_key, key_id, prf_config)
        
        logger.info(f"  master_key_fingerprint = {key_fingerprint[:16]}...")
        logger.info(f"  key_id = {key_id}")
        logger.info(f"  prf_algorithm = {prf_config.algorithm}")
        logger.info(f"  prf_output_bits = {prf_config.output_bits}")
    else:
        logger.info("  mode = unwatermarked")
        key_fingerprint = None
        key_id = None
    
    logger.info("[CACHE]")
    logger.info(f"  cache_root = {cache_root}")
    logger.info(f"  cache_namespace = {cache_namespace}")
    
    geometry_hash = compute_geometry_hash(config)
    config_hash = compute_config_hash(config)
    
    logger.info(f"  geometry_hash = {geometry_hash}")
    logger.info(f"  config_hash = {config_hash}")
    logger.info(f"  num_inversion_steps = {num_inversion_steps}")
    logger.info(f"  model_id = {config.diffusion.model_id}")
    logger.info(f"  guidance_scale = {config.diffusion.guidance_scale}")
    logger.info(f"  inference_steps = {config.diffusion.inference_timesteps}")
    
    # Log example cache key for sample 0
    if key_fingerprint is not None:
        example_cache_key = build_latent_cache_key(
            image_id="sample_000000",
            config=config,
            num_inversion_steps=num_inversion_steps,
            master_key=master_key,
            key_id=key_id,
        )
        logger.info(f"  example_cache_key (sample_000000) = {example_cache_key}")

