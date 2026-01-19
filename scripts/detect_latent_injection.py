#!/usr/bin/env python3
"""
Detect Latent Injection watermarks in images.

This script is specialized for Latent Injection watermarking strategy and provides
flexible extraction methods and inversion options.

IMPORTANT: Seed-Independent Detection
    - The watermark is based ONLY on key_id (not the generation seed)
    - You do NOT need to know the seed used during generation
    - This makes real-world detection practical

IMPORTANT REQUIREMENTS:
    - num_inversion_steps MUST match the value used during generation
    - Only unconditional detection (guidance_scale=1.0) is currently supported

Extraction Methods:
    - whitened: High-pass filter to suppress image content (for z_0)
    - normalized: Zero-mean/unit-variance normalization (for z_T)
    - raw: No processing (for debugging)

Usage:
    # Fast detection (VAE encoding only, whitened extraction)
    python scripts/detect_latent_injection.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/latentinjection.yaml \
        --extraction-method whitened

    # Accurate detection (DDIM inversion, normalized extraction)
    python scripts/detect_latent_injection.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/latentinjection.yaml \
        --use-ddim-inversion \
        --extraction-method normalized \
        --num-inversion-steps 50

    # Batch detection
    python scripts/detect_latent_injection.py \
        --image-dir outputs/generated \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/latentinjection.yaml \
        --extraction-method whitened \
        --output results.json

    # With custom threshold
    python scripts/detect_latent_injection.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/latentinjection.yaml \
        --extraction-method whitened \
        --threshold 2.0 \
        --alpha 0.001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image

from src.core.config import AppConfig, WatermarkedConfig
from src.engine.pipeline import create_pipeline
from src.detection.inversion import DDIMInverter, encode_image_to_latent
from src.detection.observe import observe_latent_numpy
from src.detection.prf import PRFKeyDerivation
from src.algorithms.g_field import GFieldGenerator
from src.detection.statistics import compute_s_statistic, compute_p_value, DetectionResult
from src.detection.calibration import calibrate_theoretical
from scripts.utils import (
    setup_logging,
    get_device,
    save_metadata,
)


def detect_single_image(
    image_path: Path,
    key_id: str,
    master_key: str,
    config: Optional[AppConfig] = None,
    pipeline: Optional[Any] = None,
    vae: Optional[Any] = None,
    use_ddim_inversion: bool = False,
    num_inversion_steps: int = 50,
    extraction_method: str = "whitened",
    threshold: Optional[float] = None,
    alpha: float = 0.01,
    device: str = "cuda",
    logger: Optional[Any] = None,
) -> DetectionResult:
    """
    Detect latent injection watermark in a single image.
    
    Args:
        image_path: Path to image file
        key_id: Public key identifier
        master_key: Secret master key
        config: Application configuration (required for G-field shape)
        pipeline: Stable Diffusion pipeline (for DDIM inversion)
        vae: VAE encoder (for simple encoding)
        use_ddim_inversion: Use full DDIM inversion
        num_inversion_steps: Steps for DDIM inversion
        extraction_method: G_observed extraction method ("whitened", "normalized", "raw")
        threshold: Detection threshold (overrides alpha)
        alpha: Target FPR for threshold
        device: Compute device
        logger: Logger instance
        
    Returns:
        DetectionResult with S-statistic, p-value, and decision
    """
    if logger is None:
        logger = setup_logging()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Determine G-field shape from config or use default
    if config and isinstance(config.watermark, WatermarkedConfig):
        g_shape = tuple(config.watermark.algorithm_params.g_field.shape)
        prf_config = config.watermark.key_settings.prf_config
        g_field_config = config.watermark.algorithm_params.g_field
    else:
        # Default shape for SD 1.x
        g_shape = (4, 64, 64)
        from src.core.config import PRFConfig, GFieldConfig
        prf_config = PRFConfig()
        g_field_config = GFieldConfig()
    
    # Get latent representation
    if use_ddim_inversion:
        if pipeline is None:
            raise ValueError("Pipeline is required for DDIM inversion")
        logger.info(f"Using DDIM inversion ({num_inversion_steps} steps)...")
        inverter = DDIMInverter(pipeline, device=device)
        latent = inverter.invert(
            image,
            num_inference_steps=num_inversion_steps,
            guidance_scale=1.0,  # Only unconditional inversion supported
        )
        latent_np = latent.cpu().numpy().squeeze(0)  # Remove batch dim
    elif vae is not None:
        logger.info("Using VAE encoding (z_0 only, no inversion)...")
        # For latentinjection detection: allow resizing for convenience
        latent = encode_image_to_latent(
            image, 
            vae, 
            device=device,
            allow_resize=True,  # Fast-path detection allows resizing
        )
        latent_np = latent.cpu().numpy().squeeze(0)
    else:
        raise ValueError("Must provide either pipeline (for DDIM) or vae (for encoding)")
    
    # Extract G_observed
    logger.info(f"Extraction method: {extraction_method}")
    g_observed = observe_latent_numpy(
        latent_np,
        method=extraction_method,
    )
    
    # Compute expected G-field using PRF (same as generation)
    prf = PRFKeyDerivation(master_key, prf_config)
    num_elements = g_shape[0] * g_shape[1] * g_shape[2]
    prf_seeds = prf.generate_seeds(key_id, num_elements)
    
    g_gen = GFieldGenerator(
        mapping_mode=g_field_config.mapping_mode,
        domain=g_field_config.domain,
        frequency_mode=g_field_config.frequency_mode,
        low_freq_cutoff=g_field_config.low_freq_cutoff,
        normalize_zero_mean=True,
        normalize_unit_variance=True,
    )
    g_expected = g_gen.generate_g_field(g_shape, prf_seeds)
    
    # Ensure shapes match
    if g_observed.shape != g_expected.shape:
        from scipy.ndimage import zoom
        scale_factors = [g_expected.shape[i] / g_observed.shape[i] for i in range(3)]
        g_observed = zoom(g_observed, scale_factors, order=1)
        logger.warning(f"Resized G_observed from {g_observed.shape} to {g_expected.shape}")
    
    # Compute S-statistic
    s_stat = compute_s_statistic(
        g_observed=g_observed,
        g_expected=g_expected,
    )
    
    # Compute p-value
    p_value = compute_p_value(s_stat)
    
    # Determine threshold
    if threshold is None:
        threshold = calibrate_theoretical(alpha)
    
    # Make decision
    is_watermarked = s_stat > threshold
    
    result = DetectionResult(
        s_statistic=float(s_stat),
        p_value=float(p_value),
        is_watermarked=is_watermarked,
        threshold=float(threshold),
        confidence=float(1.0 - p_value),
        n_elements=int(np.prod(g_shape)),
    )
    
    return result


def detect_batch(
    image_paths: List[Path],
    key_id: str,
    master_key: str,
    config: Optional[AppConfig] = None,
    pipeline: Optional[Any] = None,
    vae: Optional[Any] = None,
    use_ddim_inversion: bool = False,
    num_inversion_steps: int = 50,
    extraction_method: str = "whitened",
    threshold: Optional[float] = None,
    alpha: float = 0.01,
    device: str = "cuda",
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Detect latent injection watermarks in multiple images.
    
    Args:
        image_paths: List of image file paths
        key_id: Public key identifier
        master_key: Secret master key
        config: Application configuration
        pipeline: Stable Diffusion pipeline
        vae: VAE encoder
        use_ddim_inversion: Use full DDIM inversion
        num_inversion_steps: Steps for DDIM inversion
        extraction_method: G_observed extraction method
        threshold: Detection threshold
        alpha: Target FPR for threshold
        device: Compute device
        logger: Logger instance
        
    Returns:
        List of detection result dictionaries
    """
    if logger is None:
        logger = setup_logging()
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing [{i}/{len(image_paths)}]: {image_path.name}")
        
        try:
            result = detect_single_image(
                image_path=image_path,
                key_id=key_id,
                master_key=master_key,
                config=config,
                pipeline=pipeline,
                vae=vae,
                use_ddim_inversion=use_ddim_inversion,
                num_inversion_steps=num_inversion_steps,
                extraction_method=extraction_method,
                threshold=threshold,
                alpha=alpha,
                device=device,
                logger=logger,
            )
            
            results.append({
                "image_path": str(image_path),
                **result.to_dict(),
            })
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append({
                "image_path": str(image_path),
                "error": str(e),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect Latent Injection watermarks in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to single image file",
    )
    input_group.add_argument(
        "--image-dir",
        type=str,
        help="Path to directory containing images",
    )
    
    # Required parameters
    parser.add_argument(
        "--key-id",
        type=str,
        required=True,
        help="Public key identifier (must match generation)",
    )
    parser.add_argument(
        "--master-key",
        type=str,
        required=True,
        help="Secret master key (must match generation)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (must include latent injection configuration)",
    )
    
    # Extraction method
    parser.add_argument(
        "--extraction-method",
        type=str,
        default="whitened",
        choices=["whitened", "normalized", "raw"],
        help="G_observed extraction method: whitened (high-pass filter, for z_0), normalized (zero-mean/unit-variance, for z_T), raw (no processing). Default: whitened",
    )
    
    # Inversion options
    parser.add_argument(
        "--use-ddim-inversion",
        action="store_true",
        help="Use full DDIM inversion (more accurate but slower)",
    )
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        default=50,
        help="Number of steps for DDIM inversion (MUST match generation). Default: 50",
    )
    
    # Thresholds
    parser.add_argument(
        "--threshold",
        type=float,
        help="Detection threshold (overrides --alpha)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Target FPR for threshold (used if threshold not provided). Default: 0.01",
    )
    
    # Performance
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level)
    
    logger.info("=" * 80)
    logger.info("Latent Injection Watermark Detection")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = AppConfig.from_yaml(args.config)
    
    # Validate config has latent injection (not seedbias)
    if hasattr(config.watermark, 'algorithm_params') and \
       config.watermark.algorithm_params.seed_bias is not None:
        logger.warning("Config includes seedbias configuration. Consider using detect_seed_bias.py instead.")
    
    # Create pipeline/vae if needed
    pipeline = None
    vae = None
    
    if args.use_ddim_inversion or args.extraction_method != "raw":
        device = get_device(args.device, use_fp16=args.fp16)
        logger.info("Creating pipeline...")
        pipeline = create_pipeline(config.diffusion, device=device)
        vae = pipeline.vae
        logger.info(f"Device: {device}")
    
    # Collect image paths
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + \
                      list(image_dir.glob("*.jpeg"))
        image_paths = sorted(image_paths)
    
    if not image_paths:
        raise ValueError("No images found")
    
    logger.info(f"Processing {len(image_paths)} image(s)")
    logger.info(f"Key ID: {args.key_id}")
    logger.info(f"Extraction method: {args.extraction_method}")
    logger.info(f"DDIM inversion: {args.use_ddim_inversion}")
    if args.use_ddim_inversion:
        logger.info(f"Inversion steps: {args.num_inversion_steps} (MUST match generation!)")
    
    # Run detection
    if len(image_paths) == 1:
        result = detect_single_image(
            image_path=image_paths[0],
            key_id=args.key_id,
            master_key=args.master_key,
            config=config,
            pipeline=pipeline,
            vae=vae,
            use_ddim_inversion=args.use_ddim_inversion,
            num_inversion_steps=args.num_inversion_steps,
            extraction_method=args.extraction_method,
            threshold=args.threshold,
            alpha=args.alpha,
            device=device if pipeline else "cpu",
            logger=logger,
        )
        
        logger.info("=" * 80)
        logger.info("Detection Results")
        logger.info("=" * 80)
        logger.info(f"S-statistic: {result.s_statistic:.4f}")
        logger.info(f"p-value: {result.p_value:.6f}")
        logger.info(f"Threshold: {result.threshold:.4f}")
        logger.info(f"Confidence: {result.confidence*100:.2f}%")
        logger.info(f"Elements: {result.n_elements:,}")
        logger.info(f"Decision: {'✅ WATERMARK DETECTED' if result.is_watermarked else '❌ No watermark detected'}")
        
        results = [{
            "image_path": str(image_paths[0]),
            **result.to_dict(),
        }]
    else:
        results = detect_batch(
            image_paths=image_paths,
            key_id=args.key_id,
            master_key=args.master_key,
            config=config,
            pipeline=pipeline,
            vae=vae,
            use_ddim_inversion=args.use_ddim_inversion,
            num_inversion_steps=args.num_inversion_steps,
            extraction_method=args.extraction_method,
            threshold=args.threshold,
            alpha=args.alpha,
            device=device if pipeline else "cpu",
            logger=logger,
        )
        
        # Summary
        detected = sum(1 for r in results if r.get("is_watermarked", False))
        logger.info("=" * 80)
        logger.info(f"Summary: {detected}/{len(results)} images detected as watermarked")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

