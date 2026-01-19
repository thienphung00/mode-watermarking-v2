#!/usr/bin/env python3
"""
Detect watermarks in images using PRF-based system.

Supports batch detection, multiple extraction methods, and optional DDIM inversion.
Uses only key_id + master_key (no metadata like zT_hash needed).

IMPORTANT: Seed-Independent Detection
    - The watermark is based ONLY on key_id (not the generation seed)
    - You do NOT need to know the seed used during generation
    - The --seed parameter is optional and ignored for watermark detection
    - This makes real-world detection practical

IMPORTANT REQUIREMENTS:
    - num_inversion_steps MUST match the value used during generation
    - For seedbias hybrid/full_inversion modes: images MUST be exactly 512x512
    - Only unconditional detection (guidance_scale=1.0) is currently supported

Usage:
    # Detect watermark in single image (NO SEED NEEDED!)
    python scripts/detect_watermark.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/defaults.yaml

    # Batch detection
    python scripts/detect_watermark.py \
        --image-dir outputs/generated \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/defaults.yaml \
        --output results.json

    # Seedbias detection with full inversion (requires 512x512 images)
    python scripts/detect_watermark.py \
        --image watermarked_512x512.png \
        --key-id abc123 \
        --strategy seedbias \
        --detection-mode full_inversion \
        --num-inversion-steps 50 \
        --config configs/defaults.yaml

    # With DDIM inversion (num_inversion_steps must match generation!)
    python scripts/detect_watermark.py \
        --image img.png \
        --key-id abc123 \
        --use-ddim-inversion \
        --num-inversion-steps 50

    # With custom threshold
    python scripts/detect_watermark.py \
        --image img.png \
        --key-id abc123 \
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
from src.detection.pipeline import HybridDetector, detect_seed_bias_watermark
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
    strategy: Optional[str] = None,  # "latentinjection" or "seedbias"
    detection_mode: str = "hybrid",  # For seedbias: "fast_only", "hybrid", "full_inversion"
    seed: Optional[int] = None,  # For seedbias detection
    logger: Optional[Any] = None,
) -> DetectionResult:
    """
    Detect watermark in a single image.
    
    Args:
        image_path: Path to image file
        key_id: Public key identifier
        master_key: Secret master key
        config: Application configuration (optional, for G-field shape)
        pipeline: SD pipeline (optional, for DDIM inversion)
        vae: VAE encoder (optional, for simple encoding)
        use_ddim_inversion: Use full DDIM inversion
        num_inversion_steps: Steps for DDIM inversion
        extraction_method: G_observed extraction method
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
    
    # Auto-detect strategy from config if not provided
    if strategy is None and config and isinstance(config.watermark, WatermarkedConfig):
        if config.watermark.algorithm_params.seed_bias is not None:
            strategy = "seedbias"
        else:
            strategy = "latentinjection"
    
    # Use seedbias detection if strategy is seedbias
    if strategy == "seedbias":
        if pipeline is None:
            raise ValueError("Pipeline is required for seedbias detection")
        
        # Validate image size for hybrid/full_inversion modes
        if detection_mode in ["hybrid", "full_inversion"] and image.size != (512, 512):
            logger.warning(
                f"Image size is {image.size[0]}x{image.size[1]}. "
                f"Detection mode '{detection_mode}' requires exactly 512x512 for mathematical correctness. "
                f"Consider resizing the image or using detection_mode='fast_only'."
            )
            # Note: The error will be raised by the detection pipeline itself
        
        logger.info(f"Using seedbias detection (mode={detection_mode})...")
        threshold_high_val = threshold if threshold else calibrate_theoretical(alpha)
        threshold_low_val = threshold * 0.3 if threshold else calibrate_theoretical(alpha) * 0.3
        logger.info(f"Thresholds: high={threshold_high_val:.4f}, low={threshold_low_val:.4f}")
        
        result_dict = detect_seed_bias_watermark(
            image=image,
            key_id=key_id,
            master_key=master_key,
            pipeline=pipeline,
            detection_mode=detection_mode,
            num_inference_steps=num_inversion_steps,  # MUST match generation steps
            threshold_high=threshold_high_val,
            threshold_low=threshold_low_val,
            seed=seed,
            guidance_scale=1.0,  # Only unconditional inversion supported
        )
        
        # Debug: show which stage was used and scores
        logger.info(f"Detection stage: {result_dict.get('stage', 'unknown')}")
        if 'metadata' in result_dict:
            if 'stage1_score' in result_dict['metadata']:
                logger.info(f"Stage 1 score: {result_dict['metadata']['stage1_score']:.4f}")
            if 'stage2_score' in result_dict['metadata']:
                logger.info(f"Stage 2 score: {result_dict['metadata']['stage2_score']:.4f}")
            logger.info(f"Stage 1 conclusive: {result_dict['metadata'].get('stage1_conclusive', 'N/A')}")
        
        # Convert to DetectionResult
        return DetectionResult(
            s_statistic=result_dict["score"],
            p_value=compute_p_value(result_dict["score"]),
            is_watermarked=result_dict["detected"],
            threshold=result_dict["threshold_high"],
            confidence=1.0 - compute_p_value(result_dict["score"]),
            n_elements=0,  # Not available from hybrid detector
        )
    
    # Standard latent injection detection
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
    if use_ddim_inversion and pipeline is not None:
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
        n_elements=int(g_observed.size),
    )
    
    return result


def detect_batch(
    image_paths: List[Path],
    key_id: str,
    master_key: str,
    config: Optional[AppConfig] = None,
    pipeline: Optional[Any] = None,
    vae: Optional[Any] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Detect watermarks in multiple images.
    
    Args:
        image_paths: List of image paths
        key_id: Public key identifier (same for all images)
        master_key: Secret master key
        config: Application configuration
        pipeline: SD pipeline
        vae: VAE encoder
        **kwargs: Additional arguments for detect_single_image
        
    Returns:
        List of detection result dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = detect_single_image(
                image_path=image_path,
                key_id=key_id,
                master_key=master_key,
                config=config,
                pipeline=pipeline,
                vae=vae,
                **kwargs,
            )
            
            results.append({
                "image_path": str(image_path),
                **result.to_dict(),
            })
        except Exception as e:
            results.append({
                "image_path": str(image_path),
                "error": str(e),
            })
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect watermarks in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory of images for batch detection",
    )
    parser.add_argument(
        "--latent",
        type=str,
        help="Path to latent tensor file (.pt)",
    )
    
    # Key settings
    parser.add_argument(
        "--key-id",
        type=str,
        required=True,
        help="Public key identifier",
    )
    parser.add_argument(
        "--master-key",
        type=str,
        required=True,
        help="Secret master key (keep secret!)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (for G-field shape)",
    )
    
    # Detection parameters
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["latentinjection", "seedbias"],
        help="Watermarking strategy (auto-detected from config if not provided)",
    )
    parser.add_argument(
        "--extraction-method",
        type=str,
        default="whitened",
        choices=["whitened", "normalized", "raw", "sign"],
        help="G_observed extraction method (for latentinjection strategy)",
    )
    parser.add_argument(
        "--use-ddim-inversion",
        action="store_true",
        help="Use full DDIM inversion (for latentinjection strategy, requires --config)",
    )
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        default=50,
        help="Number of DDIM inversion steps (MUST match generation steps for correctness)",
    )
    parser.add_argument(
        "--detection-mode",
        type=str,
        default="hybrid",
        choices=["fast_only", "hybrid", "full_inversion"],
        help="Detection mode for seedbias strategy (hybrid/full_inversion require exact 512x512 images)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Detection threshold (overrides --alpha)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Target FPR for threshold (default: 0.01)",
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
    logger.info("Watermark Detection")
    logger.info("=" * 80)
    
    # Validate input
    if not args.image and not args.image_dir and not args.latent:
        raise ValueError("Must provide --image, --image-dir, or --latent")
    
    # Load configuration if provided
    config = None
    pipeline = None
    vae = None
    
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = AppConfig.from_yaml(args.config)
        
        # Create pipeline if needed
        if args.use_ddim_inversion or args.extraction_method != "raw":
            device = get_device(args.device, use_fp16=args.fp16)
            logger.info("Creating pipeline...")
            pipeline = create_pipeline(config.diffusion, device=device)
            vae = pipeline.vae
    
    # Get device
    device = get_device(args.device, use_fp16=args.fp16)
    logger.info(f"Device: {device}")
    
    # Collect image paths
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        image_paths = sorted(image_paths)
    else:
        # Latent-based detection (not implemented in this version)
        raise NotImplementedError("Latent-based detection not yet implemented")
    
    if not image_paths:
        raise ValueError("No images found")
    
    logger.info(f"Processing {len(image_paths)} image(s)")
    logger.info(f"Key ID: {args.key_id}")
    logger.info(f"Strategy: {args.strategy or 'auto-detect'}")
    logger.info(f"Detection mode: {args.detection_mode}")
    logger.info(f"Extraction method: {args.extraction_method}")
    logger.info(f"DDIM inversion: {args.use_ddim_inversion}")
    if args.use_ddim_inversion or args.detection_mode in ["hybrid", "full_inversion"]:
        logger.info(f"Inversion steps: {args.num_inversion_steps} (MUST match generation!)")
    if args.detection_mode in ["hybrid", "full_inversion"]:
        logger.warning("Note: hybrid/full_inversion modes require images to be exactly 512x512")
    
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
            device=device,
            strategy=args.strategy,
            detection_mode=args.detection_mode,
            seed=None,  # Seed is no longer used for detection
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
            device=device,
            strategy=args.strategy,
            detection_mode=args.detection_mode,
            seed=None,  # Seed is no longer used for detection
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
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    from typing import Any
    main()

