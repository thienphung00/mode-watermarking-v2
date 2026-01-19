#!/usr/bin/env python3
"""
Phase 3: Run detection ablation using trained likelihood models.

This script:
1. Loads families from Phase 0
2. Loads likelihood models per family
3. For each config in each family:
   - Loads cached images + latents
   - Computes g-values
   - Runs BayesianDetector using family likelihood
   - Computes log_odds and ROC metrics
4. Saves per-config results

No image generation, no g export, no likelihood training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig, compute_cache_key, WatermarkedConfig
from src.engine.pipeline import create_pipeline
from src.detection.inversion import DDIMInverter
from src.detection.g_values import compute_g_values, g_field_config_to_dict
from src.models.detectors import BayesianDetector
from scripts.utils import setup_logging, get_device


def compute_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve from scores and labels.
    
    Args:
        scores: Detection scores (log_odds) [N]
        labels: True labels (0 or 1) [N]
        num_thresholds: Number of threshold points for ROC curve
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        thresholds = np.linspace(scores.max() + 1.0, scores.min() - 1.0, num_thresholds)
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), thresholds
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    unique_scores = np.unique(scores)
    
    if len(unique_scores) <= num_thresholds:
        thresholds = np.concatenate([
            [scores.max() + 1.0],
            np.sort(unique_scores)[::-1],
            [scores.min() - 1.0],
        ])
    else:
        thresholds = np.linspace(scores.max() + 1.0, scores.min() - 1.0, num_thresholds)
    
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    
    for i, threshold in enumerate(thresholds):
        above_threshold = sorted_scores > threshold
        tp = np.sum(sorted_labels[above_threshold] == 1)
        fp = np.sum(sorted_labels[above_threshold] == 0)
        
        tpr[i] = tp / n_pos if n_pos > 0 else 0.0
        fpr[i] = fp / n_neg if n_neg > 0 else 0.0
    
    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve (AUC) using trapezoidal rule.
    
    Args:
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        
    Returns:
        AUC score (0.0 to 1.0)
    """
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    
    return float(auc)


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


def compute_detection_statistics(
    image_path: Path,
    config: AppConfig,
    detector: BayesianDetector,
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
    
    Reused from run_watermark_ablation.py.
    
    Args:
        image_path: Path to image
        config: Watermark configuration
        detector: Bayesian detector instance
        master_key: Master key for PRF
        key_id: Key identifier (None for clean samples)
        device: Device to run on
        pipeline: Pre-created diffusion pipeline
        inverter: Pre-created DDIM inverter
        num_inversion_steps: Number of inversion steps
        latent_cache_dir: Optional cache directory for latents
        model_id: Model ID for cache key determinism
        
    Returns:
        Dictionary with log_odds, N_eff, p_hat, S
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
        g_valid = g[mask > 0.5]  # [N_eff]
    else:
        g_valid = g  # [N]
    
    # Compute N_eff
    if mask is not None:
        N_eff = int((mask > 0.5).sum().item())
    else:
        N_eff = int(g.numel())
    
    # Ensure binary {0, 1}
    g_binary = (g_valid > 0).float()  # [N_eff]
    
    # Compute S = sum(g_valid)
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
    
        # Compute log_odds using detector
        if detector.use_trained:
            try:
                # CRITICAL: Verify key fingerprint when scoring
                from src.core.config import compute_key_fingerprint
                expected_key_fingerprint = None
                if isinstance(config.watermark, WatermarkedConfig):
                    expected_key_fingerprint = compute_key_fingerprint(
                        master_key,
                        key_id,
                        config.watermark.key_settings.prf_config
                    )
                
                result = detector.score(g_binary, mask_valid, expected_key_fingerprint=expected_key_fingerprint)
                log_odds = result["log_odds"].item()
            except Exception as e:
                logger.warning(f"Bayesian detector failed: {e}")
                log_odds = 0.0
        else:
            logger.warning("Detector not trained, log_odds = 0.0")
            log_odds = 0.0
    
    return {
        "log_odds": float(log_odds),
        "N_eff": int(N_eff),
        "p_hat": float(p_hat),
        "S": float(S),
    }


def run_detection_for_config(
    config_path: Path,
    family_id: str,
    likelihood_model_path: Path,
    master_key: str,
    device: str,
    cache_dir: Path,
    num_inversion_steps: int = 25,
) -> Optional[Dict[str, Any]]:
    """
    Run detection for a single config.
    
    Args:
        config_path: Path to config YAML file
        family_id: Family identifier
        likelihood_model_path: Path to likelihood model JSON
        master_key: Master key
        device: Device to run on
        cache_dir: Cache directory for datasets
        num_inversion_steps: Number of inversion steps
        
    Returns:
        Result dictionary with detection statistics and ROC metrics, or None if skipped
    """
    logger = setup_logging()
    
    # Load config
    logger.info(f"Loading config: {config_path}")
    config = AppConfig.from_yaml(str(config_path))
    config_name = config_path.stem
    
    # Create cache directory for this config
    config_cache_dir = cache_dir / config_name
    latent_cache_dir = config_cache_dir / "latents"
    
    # Check if dataset exists
    manifest_path = config_cache_dir / "manifest.json"
    if not manifest_path.exists():
        # Phase 1 only processes the first config per family, so other configs won't have cache
        logger.warning(
            f"Dataset not found: {config_cache_dir}. "
            f"Phase 1 only creates cache for the first config per family. "
            f"Skipping {config_name}."
        )
        return None
    
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    watermarked_manifest = manifest_data["watermarked"]
    clean_manifest = manifest_data["clean"]
    
    # Initialize pipeline and inverter once per config
    logger.info("Initializing pipeline and inverter...")
    pipeline = create_pipeline(config.diffusion, device=device)
    inverter = DDIMInverter(pipeline, device=device)
    
    # Load Bayesian detector
    if not likelihood_model_path.exists():
        raise FileNotFoundError(
            f"Likelihood model not found: {likelihood_model_path}. "
            f"Run Phase 2 to train likelihood model for family {family_id}."
        )
    
    detector = BayesianDetector(
        likelihood_params_path=str(likelihood_model_path),
        threshold=0.5,
    )
    logger.info(f"Loaded likelihood detector from: {likelihood_model_path}")
    
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
    
    # Compute ROC metrics
    scores = np.array(log_odds_wm + log_odds_clean)
    labels = np.array([1] * len(log_odds_wm) + [0] * len(log_odds_clean))
    
    fpr, tpr, thresholds = compute_roc_curve(scores, labels)
    auc = compute_auc(fpr, tpr)
    
    # Build result dictionary
    result = {
        "config_name": config_name,
        "config_path": str(config_path),
        "family_id": family_id,
        "num_samples": len(watermarked_manifest) + len(clean_manifest),
        "num_inversion_steps": num_inversion_steps,
        "likelihood_model_path": str(likelihood_model_path),
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
        # ROC metrics
        "auc": auc,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Run detection ablation using trained likelihood models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ablation_detection.py \\
    --families-dir experiments/watermark_ablation/families \\
    --likelihood-dir experiments/likelihood_models \\
    --cache-dir experiments/watermark_ablation/cache \\
    --results-dir experiments/watermark_ablation/results \\
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
        "--likelihood-dir",
        type=Path,
        default=Path("experiments/likelihood_models"),
        help="Directory containing likelihood models (from Phase 2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/cache"),
        help="Cache directory for datasets (from Phase 1)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/configs"),
        help="Directory containing config YAML files (for resolving relative paths)",
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
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Load likelihood model for this family
        likelihood_model_path = args.likelihood_dir / f"{family_id}.json"
        if not likelihood_model_path.exists():
            logger.error(
                f"Likelihood model not found: {likelihood_model_path}. "
                f"Run Phase 2 to train likelihood model for {family_id}."
            )
            continue
        
        # Process each config in the family
        for config_path_str in config_paths:
            config_path = args.configs_dir / config_path_str
            if not config_path.exists():
                logger.warning(f"Config not found: {config_path}, skipping")
                continue
            
            config_name = config_path.stem
            
            logger.info(f"\nProcessing config: {config_name}")
            
            try:
                result = run_detection_for_config(
                    config_path=config_path,
                    family_id=family_id,
                    likelihood_model_path=likelihood_model_path,
                    master_key=args.master_key,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    num_inversion_steps=args.num_inversion_steps,
                )
                
                # Skip if result is None (config was skipped)
                if result is None:
                    continue
                
                # Save result
                result_path = args.results_dir / f"{config_name}.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"✓ Saved results to: {result_path}")
                logger.info(f"  AUC: {result['auc']:.4f}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {config_name}: {e}", exc_info=True)
                continue
    
    logger.info("\n✓ Detection ablation complete!")


if __name__ == "__main__":
    main()

