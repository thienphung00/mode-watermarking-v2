#!/usr/bin/env python3
"""
Detect Seed Bias watermarks in images.

This script is specialized for Seed Bias watermarking strategy and provides
three detection modes optimized for different use cases.

IMPORTANT: Seed-Independent Detection
    - The watermark is based ONLY on key_id (not the generation seed)
    - You do NOT need to know the seed used during generation
    - This makes real-world detection practical

IMPORTANT REQUIREMENTS:
    - num_inversion_steps MUST match the value used during generation
    - For hybrid/full_inversion modes: images MUST be exactly 512x512
    - Only unconditional detection (guidance_scale=1.0) is currently supported

Detection Modes:
    - fast_only: Stage 1 only (250ms, 92% AUC) - Fast screening
    - hybrid: Stage 1 → Stage 2 if inconclusive (adaptive, 92-100% AUC) - Balanced
    - full_inversion: Stage 2 only (13.4s, 100% AUC) - Maximum accuracy

Usage:
    # Fast detection (Stage 1 only)
    python scripts/detect_seed_bias.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/seedbias.yaml \
        --detection-mode fast_only

    # Balanced detection (hybrid mode - recommended)
    python scripts/detect_seed_bias.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/seedbias.yaml \
        --detection-mode hybrid

    # Maximum accuracy (full inversion)
    python scripts/detect_seed_bias.py \
        --image watermarked_512x512.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/seedbias.yaml \
        --detection-mode full_inversion \
        --num-inversion-steps 50

    # Batch detection
    python scripts/detect_seed_bias.py \
        --image-dir outputs/generated \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/seedbias.yaml \
        --detection-mode hybrid \
        --output results.json

    # With custom thresholds
    python scripts/detect_seed_bias.py \
        --image watermarked.png \
        --key-id batch_001 \
        --master-key secret_key \
        --config configs/experiments/seedbias.yaml \
        --threshold-high 0.15 \
        --threshold-low 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

from src.core.config import AppConfig
from src.engine.pipeline import create_pipeline
from src.detection.pipeline import detect_seed_bias_watermark
from src.detection.statistics import compute_p_value, DetectionResult
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
    pipeline: Any,
    config: Optional[AppConfig] = None,
    detection_mode: str = "hybrid",
    num_inference_steps: int = 50,
    threshold_high: Optional[float] = None,
    threshold_low: Optional[float] = None,
    alpha: float = 0.01,
    logger: Optional[Any] = None,
) -> DetectionResult:
    """
    Detect seed bias watermark in a single image.
    
    Args:
        image_path: Path to image file
        key_id: Public key identifier
        master_key: Secret master key
        pipeline: Stable Diffusion pipeline (required)
        config: Application configuration (optional)
        detection_mode: Detection mode ("fast_only", "hybrid", "full_inversion")
        num_inference_steps: Number of steps for full inversion (must match generation)
        threshold_high: High threshold for detection (overrides alpha)
        threshold_low: Low threshold for rejection (overrides alpha)
        alpha: Target FPR for threshold (if thresholds not provided)
        logger: Logger instance
        
    Returns:
        DetectionResult with S-statistic, p-value, and decision
    """
    if logger is None:
        logger = setup_logging()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Validate image size for hybrid/full_inversion modes
    if detection_mode in ["hybrid", "full_inversion"] and image.size != (512, 512):
        logger.warning(
            f"Image size is {image.size[0]}x{image.size[1]}. "
            f"Detection mode '{detection_mode}' requires exactly 512x512 for mathematical correctness. "
            f"Consider resizing the image or using detection_mode='fast_only'."
        )
    
    # Set thresholds
    if threshold_high is None:
        threshold_high = calibrate_theoretical(alpha)
    if threshold_low is None:
        threshold_low = threshold_high * 0.3
    
    logger.info(f"Using seedbias detection (mode={detection_mode})...")
    logger.info(f"Thresholds: high={threshold_high:.4f}, low={threshold_low:.4f}")
    
    # Run detection
    result_dict = detect_seed_bias_watermark(
        image=image,
        key_id=key_id,
        master_key=master_key,
        pipeline=pipeline,
        detection_mode=detection_mode,
        num_inference_steps=num_inference_steps,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        seed=None,  # Seed is no longer used for detection
        guidance_scale=1.0,  # Only unconditional inversion supported
    )
    
    # Debug: show which stage was used and scores
    logger.info(f"Detection stage: {result_dict.get('stage', 'unknown')}")
    if 'metadata' in result_dict:
        if 'stage1_score' in result_dict['metadata'] and result_dict['metadata']['stage1_score'] is not None:
            logger.info(f"Stage 1 score: {result_dict['metadata']['stage1_score']:.4f}")
        if 'stage2_score' in result_dict['metadata']:
            logger.info(f"Stage 2 score: {result_dict['metadata']['stage2_score']:.4f}")
        if 'stage1_conclusive' in result_dict['metadata']:
            logger.info(f"Stage 1 conclusive: {result_dict['metadata']['stage1_conclusive']}")
    
    # Convert to DetectionResult
    return DetectionResult(
        s_statistic=result_dict["score"],
        p_value=compute_p_value(result_dict["score"]),
        is_watermarked=result_dict["detected"],
        threshold=result_dict["threshold_high"],
        confidence=1.0 - compute_p_value(result_dict["score"]),
        n_elements=0,  # Not available from hybrid detector
    )


def detect_batch(
    image_paths: List[Path],
    key_id: str,
    master_key: str,
    pipeline: Any,
    config: Optional[AppConfig] = None,
    detection_mode: str = "hybrid",
    num_inference_steps: int = 50,
    threshold_high: Optional[float] = None,
    threshold_low: Optional[float] = None,
    alpha: float = 0.01,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Detect seed bias watermarks in multiple images.
    
    Args:
        image_paths: List of image file paths
        key_id: Public key identifier
        master_key: Secret master key
        pipeline: Stable Diffusion pipeline
        config: Application configuration (optional)
        detection_mode: Detection mode
        num_inference_steps: Number of steps for full inversion
        threshold_high: High threshold for detection
        threshold_low: Low threshold for rejection
        alpha: Target FPR for threshold
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
                pipeline=pipeline,
                config=config,
                detection_mode=detection_mode,
                num_inference_steps=num_inference_steps,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                alpha=alpha,
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
        description="Detect Seed Bias watermarks in images",
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
        help="Path to config file (must include seedbias configuration)",
    )
    
    # Detection mode
    parser.add_argument(
        "--detection-mode",
        type=str,
        default="hybrid",
        choices=["fast_only", "hybrid", "full_inversion"],
        help="Detection mode: fast_only (250ms, 92%% AUC), hybrid (adaptive, 92-100%% AUC), full_inversion (13.4s, 100%% AUC). Default: hybrid",
    )
    parser.add_argument(
        "--num-inversion-steps",
        type=int,
        default=50,
        help="Number of inference steps for full inversion (MUST match generation). Default: 50",
    )
    
    # Thresholds
    parser.add_argument(
        "--threshold-high",
        type=float,
        help="High threshold for detection (overrides --alpha)",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        help="Low threshold for rejection (overrides --alpha)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Target FPR for threshold (used if thresholds not provided). Default: 0.01",
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
    logger.info("Seed Bias Watermark Detection")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = AppConfig.from_yaml(args.config)
    
    # Validate config has seedbias
    if not hasattr(config.watermark, 'algorithm_params') or \
       config.watermark.algorithm_params.seed_bias is None:
        raise ValueError("Config must include seedbias configuration")
    
    # Create pipeline (required for seedbias detection)
    device = get_device(args.device, use_fp16=args.fp16)
    logger.info("Creating pipeline...")
    pipeline = create_pipeline(config.diffusion, device=device)
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
    logger.info(f"Detection mode: {args.detection_mode}")
    logger.info(f"Inversion steps: {args.num_inversion_steps} (MUST match generation!)")
    if args.detection_mode in ["hybrid", "full_inversion"]:
        logger.warning("Note: hybrid/full_inversion modes require images to be exactly 512x512")
    
    # Run detection
    if len(image_paths) == 1:
        result = detect_single_image(
            image_path=image_paths[0],
            key_id=args.key_id,
            master_key=args.master_key,
            pipeline=pipeline,
            config=config,
            detection_mode=args.detection_mode,
            num_inference_steps=args.num_inversion_steps,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            alpha=args.alpha,
            logger=logger,
        )
        
        logger.info("=" * 80)
        logger.info("Detection Results")
        logger.info("=" * 80)
        logger.info(f"S-statistic: {result.s_statistic:.4f}")
        logger.info(f"p-value: {result.p_value:.6f}")
        logger.info(f"Threshold: {result.threshold:.4f}")
        logger.info(f"Confidence: {result.confidence*100:.2f}%")
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
            pipeline=pipeline,
            config=config,
            detection_mode=args.detection_mode,
            num_inference_steps=args.num_inversion_steps,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
            alpha=args.alpha,
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

