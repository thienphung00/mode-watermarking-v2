#!/usr/bin/env python3
"""
Run watermark ablation experiment.

This script:
1. Loads each config from experiments/watermark_ablation/configs/
2. Generates or reuses cached datasets for each config
3. Runs detection on watermarked and clean samples
4. Collects per-sample statistics: log_odds, N_eff, p_hat
5. Saves results to experiments/watermark_ablation/results/{config_name}.json

Usage:
    python scripts/run_watermark_ablation.py \
        --prompts-file data/coco/prompts_test.txt \
        --num-samples 100 \
        --master-key "your_secret_key" \
        --device cuda \
        --cache-dir experiments/watermark_ablation/cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig
from src.engine.pipeline import create_pipeline, generate_with_watermark
from src.engine.strategy_factory import create_strategy_from_config
from src.detection.inversion import DDIMInverter
from src.detection.g_values import compute_g_values, g_field_config_to_dict
from src.models.detectors import BayesianDetector
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


def compute_detection_statistics(
    image_path: Path,
    config: AppConfig,
    detector: Optional[BayesianDetector],
    master_key: str,
    key_id: Optional[str],
    device: str,
    pipeline: Any,
    inverter: DDIMInverter,
    num_inversion_steps: int = 25,
    latent_cache_dir: Optional[Path] = None,
    model_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute detection statistics for a single image.
    
    Args:
        image_path: Path to image
        config: Watermark configuration
        detector: Bayesian detector instance (None if using analytic log-odds)
        master_key: Master key for PRF
        key_id: Key identifier (None for clean samples)
        device: Device to run on
        pipeline: Pre-created diffusion pipeline (reused per config)
        inverter: Pre-created DDIM inverter (reused per config)
        num_inversion_steps: Number of inversion steps
        latent_cache_dir: Optional cache directory for latents
        model_id: Model ID for cache key determinism
        
    Returns:
        Dictionary with log_odds, N_eff, p_hat, S
    """
    logger = logging.getLogger(__name__)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Check cache for latent (Fix 5: deterministic cache key includes model version)
    latent_T = None
    cache_hit = False
    if latent_cache_dir is not None:
        # Create deterministic cache key based on image, inversion steps, and model version
        image_id = image_path.stem
        model_hash = hashlib.md5(model_id.encode() if model_id else b"unknown").hexdigest()[:8]
        cache_key = f"{image_id}_steps{num_inversion_steps}_model{model_hash}"
        cache_path = latent_cache_dir / f"{cache_key}.pt"
        
        if cache_path.exists():
            try:
                latent_T = torch.load(cache_path, map_location=device)
                # Validate cache integrity (shape consistency)
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
        
        # Save to cache (Fix 5: cache after inversion)
        if latent_cache_dir is not None:
            latent_cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(latent_T, cache_path)
                logger.debug(f"Cached latent: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache latent {cache_path}: {e}")
    
    # Get g-field config
    g_field_config = g_field_config_to_dict(config.watermark.algorithm_params.g_field)
    
    # Compute g-values and mask
    # For clean samples, use dummy key for computation
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
        g_valid = g[mask > 0.5]  # [N_eff]
    else:
        g_valid = g  # [N]
    
    # Compute N_eff
    if mask is not None:
        N_eff = int((mask > 0.5).sum().item())
    else:
        N_eff = int(g.numel())
    
    # g is already binary {0, 1} from compute_g_values (sign agreement)
    # But we need to ensure it's properly formatted
    g_binary = (g_valid > 0).float()  # [N_eff], already binary
    
    # Compute S = sum(g_valid) where g_valid are binary {0, 1}
    S = float(g_binary.sum().item())
    
    # Ensure batch dimension for detector
    if g_binary.dim() == 1:
        g_binary = g_binary.unsqueeze(0)  # [1, N_eff]
    if mask is not None:
        mask_valid = mask[mask > 0.5].unsqueeze(0)  # [1, N_eff]
    else:
        mask_valid = None
    
    # Compute p_hat = S / N_eff
    p_hat = S / N_eff if N_eff > 0 else 0.0
    
    # Compute log_odds using detector or analytic fallback (Fix 2)
    if detector is not None and detector.use_trained:
        # Use trained likelihood detector
        try:
            result = detector.score(g_binary, mask_valid)
            log_odds = result["log_odds"].item()
        except Exception as e:
            logger.warning(f"Bayesian detector failed, falling back to analytic log-odds: {e}")
            log_odds = compute_analytic_log_odds(S, N_eff, p_hat)
    else:
        # Use analytic log-odds from S, N_eff, p_hat (Fix 2: log warning when using fallback)
        if detector is None:
            logger.warning("Using analytic log-odds (no likelihood model provided). Set LIKELIHOOD_PARAMS_PATH for trained detector.")
        log_odds = compute_analytic_log_odds(S, N_eff, p_hat)
    
    return {
        "log_odds": float(log_odds),
        "N_eff": int(N_eff),
        "p_hat": float(p_hat),
        "S": float(S),
    }


def compute_analytic_log_odds(S: float, N_eff: int, p_hat: float) -> float:
    """
    Compute analytic log-odds from S, N_eff, and p_hat.
    
    Uses the formula:
    log_odds = S * log(p1 / p0) + (N_eff - S) * log((1 - p1) / (1 - p0))
    
    where:
    - p1 = p_hat (estimated probability for watermarked)
    - p0 = 0.5 (uniform prior for clean)
    
    Args:
        S: Sum of binary g-values (number of 1s)
        N_eff: Effective number of positions
        p_hat: Estimated probability p_hat = S / N_eff
        
    Returns:
        Log-odds ratio
    """
    eps = 1e-6
    p1 = np.clip(p_hat, eps, 1 - eps)
    p0 = 0.5
    
    log_odds = (
        S * np.log(p1 / p0) +
        (N_eff - S) * np.log((1 - p1) / (1 - p0))
    )
    
    return float(log_odds)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_experiment_for_config(
    config_path: Path,
    prompts: List[str],
    num_samples: int,
    master_key: str,
    device: str,
    cache_dir: Path,
    num_inversion_steps: int = 25,
    likelihood_params_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run experiment for a single config.
    
    Args:
        config_path: Path to config YAML file
        prompts: List of prompts
        num_samples: Number of samples per class
        master_key: Master key
        device: Device to run on
        cache_dir: Cache directory for datasets
        num_inversion_steps: Number of inversion steps
        likelihood_params_path: Optional path to likelihood parameters file
        
    Returns:
        Result dictionary with separate wm and clean statistics
    """
    logger = setup_logging()
    
    # Load config
    logger.info(f"Loading config: {config_path}")
    config = AppConfig.from_yaml(str(config_path))
    config_name = config_path.stem
    
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
    
    # Initialize pipeline and inverter once per config (Fix 3)
    logger.info("Initializing pipeline and inverter...")
    pipeline = create_pipeline(config.diffusion, device=device)
    inverter = DDIMInverter(pipeline, device=device)
    
    # Create Bayesian detector (Fix 2)
    detector = None
    likelihood_model_path = None
    if likelihood_params_path is not None:
        # Fail fast if file doesn't exist (Fix 2 requirement)
        if not likelihood_params_path.exists():
            raise FileNotFoundError(
                f"Likelihood params file not found: {likelihood_params_path}. "
                "Set LIKELIHOOD_PARAMS_PATH environment variable or --likelihood-params argument to a valid path."
            )
        try:
            detector = BayesianDetector(
                likelihood_params_path=str(likelihood_params_path),
                threshold=0.5,
            )
            likelihood_model_path = str(likelihood_params_path)
            logger.info(f"Loaded likelihood detector from: {likelihood_params_path}")
        except Exception as e:
            logger.warning(
                f"Failed to load likelihood detector from {likelihood_params_path}: {e}. "
                "Falling back to analytic log-odds."
            )
            likelihood_model_path = None
    else:
        logger.info("No likelihood params provided. Using analytic log-odds.")
        likelihood_model_path = None
    
    # Process watermarked samples
    logger.info(f"Processing {len(watermarked_manifest)} watermarked samples...")
    log_odds_wm = []
    S_wm = []
    N_eff_wm = []
    p_hat_wm = []
    
    for entry in tqdm(watermarked_manifest, desc="Watermarked"):
        image_path = config_cache_dir / entry["image_path"]
        key_id = entry["key_id"]
        
        stats = compute_detection_statistics(
            image_path=image_path,
            config=config,
            detector=detector,
            master_key=master_key,
            key_id=key_id,
            device=device,
            pipeline=pipeline,
            inverter=inverter,
            num_inversion_steps=num_inversion_steps,
            latent_cache_dir=latent_cache_dir,
            model_id=config.diffusion.model_id,
        )
        
        log_odds_wm.append(stats["log_odds"])
        S_wm.append(stats["S"])
        N_eff_wm.append(stats["N_eff"])
        p_hat_wm.append(stats["p_hat"])
    
    # Process clean samples
    logger.info(f"Processing {len(clean_manifest)} clean samples...")
    log_odds_clean = []
    S_clean = []
    N_eff_clean = []
    p_hat_clean = []
    
    for entry in tqdm(clean_manifest, desc="Clean"):
        image_path = config_cache_dir / entry["image_path"]
        
        stats = compute_detection_statistics(
            image_path=image_path,
            config=config,
            detector=detector,
            master_key=master_key,
            key_id=None,
            device=device,
            pipeline=pipeline,
            inverter=inverter,
            num_inversion_steps=num_inversion_steps,
            latent_cache_dir=latent_cache_dir,
            model_id=config.diffusion.model_id,
        )
        
        log_odds_clean.append(stats["log_odds"])
        S_clean.append(stats["S"])
        N_eff_clean.append(stats["N_eff"])
        p_hat_clean.append(stats["p_hat"])
    
    # Build result dictionary with separate wm and clean statistics (Fix 4)
    result = {
        "config_name": config_name,
        "config_path": str(config_path),
        "num_samples": num_samples,
        "num_inversion_steps": num_inversion_steps,
        "likelihood_model_path": likelihood_model_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_commit(),
        # Separate watermarked and clean statistics
        "log_odds_wm": log_odds_wm,
        "log_odds_clean": log_odds_clean,
        "S_wm": S_wm,
        "S_clean": S_clean,
        "N_eff_wm": N_eff_wm,
        "N_eff_clean": N_eff_clean,
        "p_hat_wm": p_hat_wm,
        "p_hat_clean": p_hat_clean,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run watermark ablation experiment"
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/configs"),
        help="Directory containing config files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/results"),
        help="Directory to save results",
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
        default=100,
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
        "--cache-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/cache"),
        help="Cache directory for datasets",
    )
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        default=25,
        help="Number of DDIM inversion steps",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run only this config (by name, without .yaml)",
    )
    parser.add_argument(
        "--likelihood-params",
        type=Path,
        default=None,
        help="Path to likelihood parameters JSON file (for trained detector)",
    )
    
    args = parser.parse_args()
    
    # Support LIKELIHOOD_PARAMS_PATH environment variable
    likelihood_params_path = args.likelihood_params
    if likelihood_params_path is None:
        env_likelihood_path = os.getenv("LIKELIHOOD_PARAMS_PATH")
        if env_likelihood_path:
            likelihood_params_path = Path(env_likelihood_path)
    
    # Setup
    logger = setup_logging()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompt_list(args.prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Find config files
    if args.config:
        config_files = [args.configs_dir / f"{args.config}.yaml"]
        if not config_files[0].exists():
            raise FileNotFoundError(f"Config not found: {config_files[0]}")
    else:
        config_files = sorted(args.configs_dir.glob("*.yaml"))
    
    logger.info(f"Found {len(config_files)} config files")
    
    # Run experiment for each config
    for config_path in config_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing config: {config_path.name}")
        logger.info(f"{'='*80}")
        
        try:
            result = run_experiment_for_config(
                config_path=config_path,
                prompts=prompts,
                num_samples=args.num_samples,
                master_key=args.master_key,
                device=args.device,
                cache_dir=args.cache_dir,
                num_inversion_steps=args.num_inversion_steps,
                likelihood_params_path=likelihood_params_path,
            )
            
            # Save result
            result_path = args.results_dir / f"{result['config_name']}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"✓ Saved results to: {result_path}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {config_path.name}: {e}", exc_info=True)
            continue
    
    logger.info("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()

