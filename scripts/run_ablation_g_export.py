#!/usr/bin/env python3
"""
Phase 1: Export g-values per family for likelihood training.

This script:
1. Loads families from Phase 0
2. For each family, picks the first config as representative
3. Generates images (watermarked and clean)
4. Inverts to get latents
5. Computes g-values + mask
6. Exports per-family g-datasets (no detector, no log_odds, no ROC)

Output:
    experiments/g_datasets/family_001/
      g_wm.npy      [num_samples, N_eff]
      g_clean.npy   [num_samples, N_eff]
      mask.npy      [num_samples, N_eff]
      meta.json     {family_id, config_used, N_eff, num_samples, signature}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig, compute_cache_key, WatermarkedConfig
from src.engine.pipeline import create_pipeline, generate_with_watermark
from src.engine.strategy_factory import create_strategy_from_config
from src.detection.inversion import DDIMInverter
from src.detection.g_values import compute_g_values, g_field_config_to_dict
from scripts.utils import setup_logging, get_device, load_prompt_list


def generate_dataset(
    config: AppConfig,
    prompts: List[str],
    output_dir: Path,
    num_samples: int,
    key_id: str,
    device: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate watermarked and clean datasets.
    
    Reused from run_watermark_ablation.py.
    
    Args:
        config: Watermark configuration
        prompts: List of prompts to use
        output_dir: Output directory for images
        num_samples: Number of samples to generate
        key_id: Key identifier for watermarked samples
        device: Device to run generation on
        seed: Random seed
        
    Returns:
        Tuple of (watermarked_manifest, clean_manifest)
    """
    logger = setup_logging()
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directories
    wm_dir = output_dir / "watermarked"
    clean_dir = output_dir / "unwatermarked"
    wm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pipeline
    logger.info("Loading Stable Diffusion pipeline...")
    pipeline = create_pipeline(config.diffusion, device=device)
    
    # Create strategies
    wm_strategy = create_strategy_from_config(
        config.watermark,
        config.diffusion,
        device=device,
    )
    
    # Import unwatermarked config
    from src.core.config import UnwatermarkedConfig
    clean_config = UnwatermarkedConfig(mode="unwatermarked")
    clean_strategy = create_strategy_from_config(
        clean_config,
        config.diffusion,
        device=device,
    )
    
    # Generate samples
    watermarked_manifest = []
    clean_manifest = []
    
    # Use same prompts and seeds for both watermarked and clean
    np.random.seed(seed)
    sample_seeds = [np.random.randint(0, 2**31) for _ in range(num_samples)]
    sample_prompts = [prompts[i % len(prompts)] for i in range(num_samples)]
    
    logger.info(f"Generating {num_samples} samples...")
    
    for i in tqdm(range(num_samples), desc="Generating"):
        prompt = sample_prompts[i]
        sample_seed = sample_seeds[i]
        sample_id = f"sample_{i:06d}"
        
        # Generate watermarked
        wm_result = generate_with_watermark(
            pipeline=pipeline,
            strategy=wm_strategy,
            prompt=prompt,
            sample_id=sample_id,
            num_inference_steps=config.diffusion.inference_timesteps,
            guidance_scale=config.diffusion.guidance_scale,
            seed=sample_seed,
        )
        
        wm_image_path = wm_dir / f"{sample_id}.png"
        wm_result["image"].save(wm_image_path)
        
        watermarked_manifest.append({
            "image_path": str(wm_image_path.relative_to(output_dir)),
            "label": 1,
            "key_id": key_id,
            "prompt": prompt,
            "seed": int(sample_seed),
            "sample_id": sample_id,
        })
        
        # Generate clean (unwatermarked)
        clean_result = generate_with_watermark(
            pipeline=pipeline,
            strategy=clean_strategy,
            prompt=prompt,
            sample_id=sample_id,
            num_inference_steps=config.diffusion.inference_timesteps,
            guidance_scale=config.diffusion.guidance_scale,
            seed=sample_seed,
        )
        
        clean_image_path = clean_dir / f"{sample_id}.png"
        clean_result["image"].save(clean_image_path)
        
        clean_manifest.append({
            "image_path": str(clean_image_path.relative_to(output_dir)),
            "label": 0,
            "key_id": None,
            "prompt": prompt,
            "seed": int(sample_seed),
            "sample_id": sample_id,
        })
    
    return watermarked_manifest, clean_manifest


def compute_g_values_for_image(
    image_path: Path,
    config: AppConfig,
    master_key: str,
    key_id: Optional[str],
    device: str,
    pipeline: Any,
    inverter: DDIMInverter,
    num_inversion_steps: int = 25,
    latent_cache_dir: Optional[Path] = None,
    model_id: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute g-values and mask for a single image.
    
    Args:
        image_path: Path to image
        config: Watermark configuration
        master_key: Master key for PRF
        key_id: Key identifier (None for clean samples)
        device: Device to run on
        pipeline: Pre-created diffusion pipeline
        inverter: Pre-created DDIM inverter
        num_inversion_steps: Number of inversion steps
        latent_cache_dir: Optional cache directory for latents
        model_id: Model ID for cache key determinism
        
    Returns:
        Tuple of (g_binary, mask) where:
        - g_binary: [N_eff] binary g-values
        - mask: [N_eff] binary mask (all ones if no masking)
    """
    logger = logging.getLogger(__name__)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Check cache for latent
    latent_T = None
    cache_hit = False
    if latent_cache_dir is not None:
        image_id = image_path.stem
        
        # Use comprehensive cache key that includes geometry signature, config, etc.
        # For latent inversion, we need: image_id, config, num_inversion_steps
        # CRITICAL: Include master_key and key_id for key isolation
        cache_key = compute_cache_key(
            image_id=image_id,
            config=config,
            num_inversion_steps=num_inversion_steps,
            master_key=master_key,
            key_id=key_id,
        )
        cache_path = latent_cache_dir / f"{cache_key}.pt"
        
        if cache_path.exists():
            try:
                # Load cached latent
                cached_data = torch.load(cache_path, map_location=device)
                
                # Handle both old format (just tensor) and new format (dict with metadata)
                if isinstance(cached_data, dict):
                    latent_T = cached_data["latent"]
                    cached_metadata = cached_data.get("metadata", {})
                    
                    # CRITICAL: Verify key fingerprint matches (hard fail on mismatch)
                    from src.core.config import compute_key_fingerprint
                    if isinstance(config.watermark, WatermarkedConfig):
                        expected_key_fingerprint = compute_key_fingerprint(
                            master_key,
                            key_id,
                            config.watermark.key_settings.prf_config
                        )
                        cached_key_fingerprint = cached_metadata.get("key_fingerprint")
                        
                        if cached_key_fingerprint is None:
                            # Old cache format without key fingerprint - invalidate
                            logger.error(
                                f"Cache entry {cache_path} missing key_fingerprint. "
                                f"This cache was created before key isolation fixes. "
                                f"Invalidating cache."
                            )
                            latent_T = None
                        elif cached_key_fingerprint != expected_key_fingerprint:
                            # Key mismatch - hard fail
                            raise RuntimeError(
                                f"KEY MISMATCH: Cached latent was created with different key. "
                                f"Cached key_fingerprint: {cached_key_fingerprint[:16]}..., "
                                f"Expected key_fingerprint: {expected_key_fingerprint[:16]}.... "
                                f"Cache path: {cache_path}. "
                                f"This artifact cannot be reused with a different key. "
                                f"Delete the cache or use the correct key."
                            )
                    
                    # VALIDATION: Check metadata matches current request
                    expected_metadata = {
                        "image_id": image_id,
                        "num_inversion_steps": num_inversion_steps,
                        "model_id": config.diffusion.model_id,
                    }
                    
                    for key, expected_value in expected_metadata.items():
                        if key in cached_metadata and cached_metadata[key] != expected_value:
                            logger.warning(
                                f"Cache metadata mismatch for {key}: "
                                f"cached={cached_metadata.get(key)}, expected={expected_value}. "
                                f"Invalidating cache."
                            )
                            latent_T = None
                            break
                else:
                    # Old format: just tensor - invalidate (no key fingerprint)
                    logger.error(
                        f"Cache entry {cache_path} uses old format without key fingerprint. "
                        f"Invalidating cache."
                    )
                    latent_T = None
                
                # Validate shape
                if latent_T is not None:
                    if latent_T.shape[0] == 1 and latent_T.shape[1] == 4 and len(latent_T.shape) == 4:
                        cache_hit = True
                        logger.debug(f"Cache hit: {cache_path}")
                    else:
                        logger.warning(f"Invalid cache shape {latent_T.shape}, recomputing")
                        latent_T = None
            except Exception as e:
                logger.warning(f"Failed to load cached latent {cache_path}: {e}")
                latent_T = None
    
    # Invert to zT if not cached
    if latent_T is None:
        if latent_cache_dir is not None and not cache_hit:
            logger.debug(f"Cache miss: {cache_path}")
        latent_T = inverter.invert(
            image,
            num_inference_steps=num_inversion_steps,
            prompt="",
            guidance_scale=1.0,
        )  # [1, 4, 64, 64]
        
        # Save to cache with metadata
        if latent_cache_dir is not None:
            latent_cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                # Save with metadata for validation
                image_id = image_path.stem
                
                # CRITICAL: Store key fingerprint in metadata
                from src.core.config import compute_key_fingerprint
                key_fingerprint = None
                if isinstance(config.watermark, WatermarkedConfig):
                    key_fingerprint = compute_key_fingerprint(
                        master_key,
                        key_id,
                        config.watermark.key_settings.prf_config
                    )
                
                cache_data = {
                    "latent": latent_T,
                    "metadata": {
                        "image_id": image_id,
                        "num_inversion_steps": num_inversion_steps,
                        "model_id": config.diffusion.model_id,
                        "cache_key": cache_key,
                        "key_fingerprint": key_fingerprint,
                        "key_id": key_id,
                        "prf_algorithm": config.watermark.key_settings.prf_config.algorithm if isinstance(config.watermark, WatermarkedConfig) else None,
                    }
                }
                torch.save(cache_data, cache_path)
                logger.debug(f"Cached latent with metadata: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache latent {cache_path}: {e}")
    
    # Get g-field config
    g_field_config = g_field_config_to_dict(config.watermark.algorithm_params.g_field)
    
    # Compute g-values and mask
    computation_key = key_id if key_id is not None else "__unwatermarked_dummy_key__"
    g, mask = compute_g_values(
        latent_T,
        computation_key,
        master_key,
        return_mask=True,
        g_field_config=g_field_config,
        latent_type="zT",
    )  # g: [1, N] or [N], mask: [1, N] or [N]
    
    # Ensure 1D
    if g.dim() > 1:
        g = g.flatten()
    if mask is not None and mask.dim() > 1:
        mask = mask.flatten()
    
    # Apply mask: select only valid positions
    if mask is not None:
        mask_bool = mask > 0.5
        g_valid = g[mask_bool]  # [N_eff]
        mask_valid = mask[mask_bool]  # [N_eff]
    else:
        g_valid = g  # [N]
        mask_valid = torch.ones_like(g)  # [N]
    
    # Ensure binary {0, 1}
    g_binary = (g_valid > 0).float()
    
    return g_binary, mask_valid


def export_family_g_dataset(
    family_id: str,
    family_dir: Path,
    config_path: Path,
    prompts: List[str],
    num_samples: int,
    master_key: str,
    device: str,
    cache_dir: Path,
    output_dir: Path,
    num_inversion_steps: int = 25,
) -> None:
    """
    Export g-dataset for a single family.
    
    Args:
        family_id: Family identifier
        family_dir: Family directory (contains signature.json, configs.json)
        config_path: Path to representative config YAML
        prompts: List of prompts
        num_samples: Number of samples per class
        master_key: Master key
        device: Device to run on
        cache_dir: Cache directory for datasets
        output_dir: Output directory for g-datasets
        num_inversion_steps: Number of inversion steps
    """
    logger = setup_logging()
    
    # Load config
    logger.info(f"Loading config: {config_path}")
    config = AppConfig.from_yaml(str(config_path))
    config_name = config_path.stem
    
    # Load family signature
    signature_path = family_dir / "signature.json"
    with open(signature_path, "r") as f:
        signature = json.load(f)
    
    # Create cache directory for this config
    config_cache_dir = cache_dir / config_name
    config_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Latent cache directory
    latent_cache_dir = config_cache_dir / "latents"
    
    # Check if dataset exists
    manifest_path = config_cache_dir / "manifest.json"
    if manifest_path.exists():
        logger.info(f"Using cached dataset: {config_cache_dir}")
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        watermarked_manifest = manifest_data["watermarked"]
        clean_manifest = manifest_data["clean"]
    else:
        logger.info(f"Generating dataset: {config_cache_dir}")
        # Generate dataset
        key_id = config.watermark.key_settings.key_id
        watermarked_manifest, clean_manifest = generate_dataset(
            config=config,
            prompts=prompts,
            output_dir=config_cache_dir,
            num_samples=num_samples,
            key_id=key_id,
            device=device,
        )
        
        # Save manifest
        with open(manifest_path, "w") as f:
            json.dump({
                "watermarked": watermarked_manifest,
                "clean": clean_manifest,
            }, f, indent=2)
    
    # Initialize pipeline and inverter once per config
    logger.info("Initializing pipeline and inverter...")
    pipeline = create_pipeline(config.diffusion, device=device)
    inverter = DDIMInverter(pipeline, device=device)
    
    # Process watermarked samples
    logger.info(f"Processing {len(watermarked_manifest)} watermarked samples...")
    g_wm_list = []
    mask_list = []
    
    for entry in tqdm(watermarked_manifest, desc="Watermarked"):
        image_path = config_cache_dir / entry["image_path"]
        key_id = entry["key_id"]
        
        g_binary, mask_valid = compute_g_values_for_image(
            image_path=image_path,
            config=config,
            master_key=master_key,
            key_id=key_id,
            device=device,
            pipeline=pipeline,
            inverter=inverter,
            num_inversion_steps=num_inversion_steps,
            latent_cache_dir=latent_cache_dir,
            model_id=config.diffusion.model_id,
        )
        
        g_wm_list.append(g_binary.cpu().numpy())
        mask_list.append(mask_valid.cpu().numpy())
    
    # Process clean samples
    logger.info(f"Processing {len(clean_manifest)} clean samples...")
    g_clean_list = []
    
    for entry in tqdm(clean_manifest, desc="Clean"):
        image_path = config_cache_dir / entry["image_path"]
        
        g_binary, mask_valid = compute_g_values_for_image(
            image_path=image_path,
            config=config,
            master_key=master_key,
            key_id=None,
            device=device,
            pipeline=pipeline,
            inverter=inverter,
            num_inversion_steps=num_inversion_steps,
            latent_cache_dir=latent_cache_dir,
            model_id=config.diffusion.model_id,
        )
        
        g_clean_list.append(g_binary.cpu().numpy())
        # Verify mask is consistent (should be same for all samples in family)
        if len(mask_list) > 0:
            if not np.array_equal(mask_valid.cpu().numpy(), mask_list[0]):
                logger.warning(f"Mask mismatch for clean sample {entry['sample_id']}")
    
    # Stack into arrays
    g_wm = np.stack(g_wm_list)  # [num_samples, N_eff]
    g_clean = np.stack(g_clean_list)  # [num_samples, N_eff]
    mask = np.stack(mask_list)  # [num_samples, N_eff]
    
    # Verify all masks are identical (they should be)
    if not np.allclose(mask, mask[0:1], atol=1e-6):
        logger.warning("Masks are not identical across samples, using first mask")
    mask = mask[0:1]  # [1, N_eff]
    
    # Get N_eff from mask
    N_eff = int((mask[0] > 0.5).sum())
    
    # Create output directory
    family_output_dir = output_dir / family_id
    family_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(family_output_dir / "g_wm.npy", g_wm)
    np.save(family_output_dir / "g_clean.npy", g_clean)
    np.save(family_output_dir / "mask.npy", mask)
    
    # CRITICAL: Compute and store key fingerprint
    from src.core.config import compute_key_fingerprint
    key_fingerprint = None
    key_id = None
    prf_algorithm = None
    if isinstance(config.watermark, WatermarkedConfig):
        key_id = config.watermark.key_settings.key_id
        prf_algorithm = config.watermark.key_settings.prf_config.algorithm
        key_fingerprint = compute_key_fingerprint(
            master_key,
            key_id,
            config.watermark.key_settings.prf_config
        )
    
    # Save metadata
    meta = {
        "family_id": family_id,
        "config_used": str(config_path),
        "N_eff": N_eff,
        "num_samples": num_samples,
        "signature": signature,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "key_fingerprint": key_fingerprint,
        "key_id": key_id,
        "prf_algorithm": prf_algorithm,
    }
    
    meta_path = family_output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"✓ Exported g-dataset for {family_id}:")
    logger.info(f"    g_wm: {g_wm.shape}")
    logger.info(f"    g_clean: {g_clean.shape}")
    logger.info(f"    mask: {mask.shape}")
    logger.info(f"    N_eff: {N_eff}")
    logger.info(f"    Saved to: {family_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Export g-values per family for likelihood training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablation_g_export.py \\
    --families-dir experiments/watermark_ablation/families \\
    --cache-dir experiments/watermark_ablation/cache \\
    --output-dir experiments/g_datasets \\
    --num-samples 500 \\
    --device cuda \\
    --master-key "your_secret_key"
        """
    )
    parser.add_argument(
        "--families-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/families"),
        help="Directory containing family groupings (from Phase 0)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/cache"),
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/g_datasets"),
        help="Output directory for g-datasets",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Path to prompts file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples per class (watermarked/clean)",
    )
    parser.add_argument(
        "--master-key",
        type=str,
        required=True,
        help="Master key for PRF",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        default=25,
        help="Number of DDIM inversion steps",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/configs"),
        help="Directory containing config YAML files (for resolving relative paths)",
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompt_list(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Find family directories
    family_dirs = sorted([d for d in args.families_dir.iterdir() if d.is_dir() and d.name.startswith("family_")])
    
    if not family_dirs:
        raise ValueError(f"No family directories found in {args.families_dir}")
    
    logger.info(f"Found {len(family_dirs)} families")
    
    # Process each family
    for family_dir in family_dirs:
        family_id = family_dir.name
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing family: {family_id}")
        logger.info(f"{'='*80}")
        
        # Load configs.json
        configs_path = family_dir / "configs.json"
        if not configs_path.exists():
            logger.warning(f"configs.json not found for {family_id}, skipping")
            continue
        
        with open(configs_path, "r") as f:
            config_paths = json.load(f)
        
        if not config_paths:
            logger.warning(f"No configs found for {family_id}, skipping")
            continue
        
        # Pick first config as representative
        config_path = args.configs_dir / config_paths[0]
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, skipping")
            continue
        
        try:
            export_family_g_dataset(
                family_id=family_id,
                family_dir=family_dir,
                config_path=config_path,
                prompts=prompts,
                num_samples=args.num_samples,
                master_key=args.master_key,
                device=args.device,
                cache_dir=args.cache_dir,
                output_dir=args.output_dir,
                num_inversion_steps=args.num_inversion_steps,
            )
        except Exception as e:
            logger.error(f"✗ Failed to process {family_id}: {e}", exc_info=True)
            continue
    
    logger.info("\n✓ G-export complete!")


if __name__ == "__main__":
    main()

