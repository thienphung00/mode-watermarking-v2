#!/usr/bin/env python3
"""
Generate training images (watermarked and/or unwatermarked) for detector training.

This script generates images in batches and produces a simplified manifest format
for detector training. It supports three modes:
- watermarked: Generate only watermarked images
- unwatermarked: Generate only unwatermarked images
- both: Generate parallel batches of watermarked and unwatermarked images

Usage:
    # Generate watermarked images only
    python scripts/generate_training_images.py \
        --mode watermarked \
        --config configs/experiments/watermarked.yaml \
        --prompts-file data/coco/prompts_train.txt \
        --output-dir outputs/train \
        --batch-size 32 \
        --num-images 1000 \
        --key-id batch_001

    # Generate unwatermarked images only
    python scripts/generate_training_images.py \
        --mode unwatermarked \
        --unwatermarked-config configs/experiments/unwatermarked.yaml \
        --prompts-file data/coco/prompts_train.txt \
        --output-dir outputs/train \
        --batch-size 32 \
        --num-images 1000

    # Generate both watermarked and unwatermarked
    python scripts/generate_training_images.py \
        --mode both \
        --config configs/experiments/watermarked.yaml \
        --unwatermarked-config configs/experiments/unwatermarked.yaml \
        --prompts-file data/coco/prompts_train.txt \
        --output-dir outputs/train \
        --batch-size 32 \
        --num-images 1000 \
        --key-id batch_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig, WatermarkedConfig, UnwatermarkedConfig
from src.core.interfaces import LatentInjectionStrategy
from src.engine.pipeline import (
    create_pipeline,
    generate_with_watermark,
)
from src.engine.strategy_factory import (
    create_strategy_from_config,
    create_per_sample_strategy,
)
from src.detection.g_values import g_field_config_to_dict
from scripts.utils import (
    setup_logging,
    load_prompt_list,
    get_device,
    enable_torch_compile,
)


def save_image(image: Image.Image, output_path: Path) -> None:
    """
    Save PIL image to file.
    
    Args:
        image: PIL Image to save
        output_path: Path to save image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)




def generate_batch(
    pipeline: Any,
    strategy: Any,
    prompts: List[str],
    seeds: Optional[List[int]],
    output_dir: Path,
    mode: str,
    key_id: Optional[str],
    num_inference_steps: int,
    guidance_scale: float,
    start_index: int = 0,
    logger: Any = None,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of images.
    
    Args:
        pipeline: Stable Diffusion pipeline
        strategy: Watermark strategy
        prompts: List of prompts for this batch
        seeds: Optional list of seeds (one per prompt)
        output_dir: Output directory for images
        mode: Generation mode ("watermarked" or "unwatermarked")
        key_id: Key identifier for watermarked mode (passed unchanged, constant per dataset)
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale
        start_index: Starting index for sample IDs (for matching IDs across modes)
        logger: Logger instance
        
    Returns:
        List of result dictionaries with image_path, label, key_id, and seed
    """
    results = []
    label = 1 if mode == "watermarked" else 0
    
    for i, prompt in enumerate(prompts):
        global_idx = start_index + i
        sample_id = f"img_{global_idx:06d}"
        seed = seeds[i] if seeds else None
        
        # For watermarked mode, prepare per-sample strategy with key_id
        # key_id is passed unchanged (constant per dataset, per-sample randomness comes from seed)
        if mode == "watermarked":
            # Check if it's LatentInjectionStrategy (needs per-sample regeneration)
            if isinstance(strategy, LatentInjectionStrategy):
                create_per_sample_strategy(
                    strategy=strategy,
                    sample_id=sample_id,
                    prompt=prompt,
                    key_id=key_id,  # Use key_id unchanged, constant per dataset
                )
            # SeedBiasStrategy doesn't need per-sample regeneration - it uses seed directly
        
        # Generate image
        result = generate_with_watermark(
            pipeline=pipeline,
            strategy=strategy,
            prompt=prompt,
            sample_id=sample_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Save image
        image = result["image"]
        image_filename = f"{sample_id}.png"
        image_path = output_dir / image_filename
        save_image(image, image_path)
        
        # Build manifest entry (g-values are computed later by precompute_inverted_g_values.py)
        # key_id is passed unchanged (constant per dataset split)
        results.append({
            "image_path": str(image_path),  # Will be made relative later
            "label": label,
            "key_id": key_id if mode == "watermarked" else None,  # Constant key_id, or None for unwatermarked
            "seed": seed,
        })
    
    return results


def generate_training_images(
    watermarked_config: Optional[AppConfig],
    unwatermarked_config: Optional[AppConfig],
    prompts: List[str],
    output_dir: Path,
    mode: str,
    batch_size: int,
    num_images: int,
    key_id: Optional[str] = None,
    master_key: Optional[str] = None,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = None,
    device: str = "cuda",
    use_fp16: bool = True,
    use_torch_compile: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """
    Generate training images in batches.
    
    Args:
        watermarked_config: Application configuration for watermarked mode
        unwatermarked_config: Application configuration for unwatermarked mode
        prompts: List of text prompts
        output_dir: Base output directory
        mode: Generation mode ("watermarked", "unwatermarked", or "both")
        batch_size: Number of images per batch
        num_images: Total number of images to generate per mode
        key_id: Public key identifier (for watermarked mode)
        master_key: Master key override (optional)
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale (uses config default if None)
        device: Device to use
        use_fp16: Use FP16 precision
        use_torch_compile: Enable torch.compile
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logging()
    
    # Override configs if needed
    if watermarked_config:
        if use_fp16:
            watermarked_config.diffusion.use_fp16 = True
        if guidance_scale is not None:
            watermarked_config.diffusion.guidance_scale = guidance_scale
        
        # Override master key if provided
        if master_key and isinstance(watermarked_config.watermark, WatermarkedConfig):
            watermarked_config.watermark.key_settings.key_master = master_key
        
        # Override key_id if provided
        if key_id and isinstance(watermarked_config.watermark, WatermarkedConfig):
            watermarked_config.watermark.key_settings.key_id = key_id
    
    if unwatermarked_config:
        if use_fp16:
            unwatermarked_config.diffusion.use_fp16 = True
        if guidance_scale is not None:
            unwatermarked_config.diffusion.guidance_scale = guidance_scale
    
    logger.info(f"Generating training images")
    logger.info(f"Mode: {mode}")
    logger.info(f"Total images per mode: {num_images}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"FP16: {use_fp16}")
    logger.info(f"Steps: {num_inference_steps}")
    
    # Extract and log g-field config from watermarked_config for verification only
    # (G-values are computed later by precompute_inverted_g_values.py, not during generation)
    if watermarked_config and isinstance(watermarked_config.watermark, WatermarkedConfig):
        if hasattr(watermarked_config.watermark, 'algorithm_params') and hasattr(watermarked_config.watermark.algorithm_params, 'g_field'):
            g_field_config_dict = g_field_config_to_dict(
                watermarked_config.watermark.algorithm_params.g_field,
                algorithm_params=watermarked_config.watermark.algorithm_params,
            )
            logger.info(f"G-field config (from watermarked config, for verification): {json.dumps(g_field_config_dict, indent=2)}")
            # Compute hash for reproducibility and matching across generation/precompute/training/detection
            json_str = json.dumps(g_field_config_dict, sort_keys=True)
            g_field_config_hash = hashlib.sha256(json_str.encode()).hexdigest()[:16]
            logger.info(f"G-field config hash: {g_field_config_hash}")
    
    # Limit prompts to num_images
    if len(prompts) > num_images:
        prompts = prompts[:num_images]
    elif len(prompts) < num_images:
        # Repeat prompts if needed
        prompts = (prompts * ((num_images // len(prompts)) + 1))[:num_images]
    
    # Create output directories
    output_dir = Path(output_dir)
    watermarked_dir = output_dir / "watermarked"
    unwatermarked_dir = output_dir / "unwatermarked"
    manifests_dir = output_dir / "manifests"
    
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    unwatermarked_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest file
    manifest_path = manifests_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()  # Start fresh
    
    # Determine which modes to generate
    modes_to_generate = []
    if mode == "both":
        modes_to_generate = ["watermarked", "unwatermarked"]
    else:
        modes_to_generate = [mode]
    
    # Create pipeline once (reuse for all batches)
    # Use watermarked config if available, otherwise unwatermarked config
    pipeline_config = watermarked_config.diffusion if watermarked_config else unwatermarked_config.diffusion
    logger.info("Creating diffusion pipeline...")
    pipeline = create_pipeline(pipeline_config, device=device)
    
    if use_torch_compile:
        logger.info("Enabling torch.compile...")
        pipeline.unet = enable_torch_compile(pipeline.unet)
    
    # Create strategies for each mode
    strategies = {}
    for gen_mode in modes_to_generate:
        if gen_mode == "watermarked":
            if watermarked_config is None:
                raise ValueError("Watermarked config is required for watermarked mode")
            # Create watermarked strategy
            strategy = create_strategy_from_config(
                watermarked_config.watermark,
                watermarked_config.diffusion,
                device=device,
            )
            strategies["watermarked"] = strategy
            logger.info(f"Created watermarked strategy: {type(strategy).__name__}")
        else:  # unwatermarked
            if unwatermarked_config is None:
                raise ValueError("Unwatermarked config is required for unwatermarked mode")
            # Create unwatermarked strategy from config file
            strategy = create_strategy_from_config(
                unwatermarked_config.watermark,
                unwatermarked_config.diffusion,
                device=device,
            )
            strategies["unwatermarked"] = strategy
            logger.info(f"Created unwatermarked strategy: {type(strategy).__name__}")
    
    # Generate images in batches
    num_batches = (num_images + batch_size - 1) // batch_size
    
    # Create progress bars
    progress_bars = {}
    for gen_mode in modes_to_generate:
        progress_bars[gen_mode] = tqdm(total=num_images, desc=f"Generating {gen_mode}")
    
    logger.info(f"\nGenerating images in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        # Get prompts for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_prompts = prompts[start_idx:end_idx]
        
        if not batch_prompts:
            break
        
        # Generate seeds for this batch (same seeds for all modes)
        batch_seeds = list(range(start_idx, end_idx))
        
        # Generate batch for each mode (using same prompts and seeds)
        for gen_mode in modes_to_generate:
            strategy = strategies[gen_mode]
            mode_output_dir = watermarked_dir if gen_mode == "watermarked" else unwatermarked_dir
            
            # Generate batch
            batch_results = generate_batch(
                pipeline=pipeline,
                strategy=strategy,
                prompts=batch_prompts,
                seeds=batch_seeds,
                output_dir=mode_output_dir,
                mode=gen_mode,
                key_id=key_id,
                num_inference_steps=num_inference_steps,
                guidance_scale=pipeline_config.guidance_scale,
                start_index=start_idx,
                logger=logger,
            )
            
            # Write manifest entries (g-values will be computed later by precompute_inverted_g_values.py)
            for result in batch_results:
                # Image path is absolute, make it relative to output_dir
                image_path_obj = Path(result["image_path"])
                if image_path_obj.is_absolute():
                    # Make relative to output_dir
                    rel_image_path = image_path_obj.relative_to(output_dir)
                else:
                    # Already relative, prepend mode directory
                    rel_image_path = Path(gen_mode) / image_path_obj.name
                
                # Write manifest entry (clean format: image_path, label, key_id, seed)
                entry = {
                    "image_path": str(rel_image_path),
                    "label": result["label"],
                    "key_id": result.get("key_id"),  # Constant key_id for watermarked, None for unwatermarked
                    "seed": result.get("seed"),
                }
                
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            
            progress_bars[gen_mode].update(len(batch_results))
    
    # Close progress bars
    for pbar in progress_bars.values():
        pbar.close()
    
    for gen_mode in modes_to_generate:
        logger.info(f"✓ Generated {num_images} {gen_mode} images")
    
    logger.info(f"\n✅ Generation complete!")
    logger.info(f"Manifest saved to {manifest_path}")
    logger.info(f"Total entries: {num_images * len(modes_to_generate)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training images (watermarked and/or unwatermarked)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (watermarked.yaml, required if --mode is 'watermarked' or 'both')",
    )
    parser.add_argument(
        "--unwatermarked-config",
        type=str,
        help="Path to YAML config file for unwatermarked mode (required if --mode is 'both' or 'unwatermarked')",
    )
    
    # Input
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="data/coco/prompts_train.txt",
        help="Path to text file with prompts (one per line)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train",
        help="Output directory for images and manifest",
    )
    
    # Generation mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["watermarked", "unwatermarked", "both"],
        help="Generation mode: watermarked, unwatermarked, or both",
    )
    
    # Batch parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per batch",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        required=True,
        help="Total number of images to generate per mode",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        help="Number of batches to generate (alternative to --num-images)",
    )
    
    # Watermark settings
    parser.add_argument(
        "--key-id",
        type=str,
        help="Public key identifier (for watermarked mode)",
    )
    parser.add_argument(
        "--master-key",
        type=str,
        help="Master key override (optional, keep secret!)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Guidance scale (overrides config)",
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
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for UNet",
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
    
    # Validate required arguments based on mode
    if args.mode == "both":
        if not args.config:
            raise ValueError("--config is required when --mode is 'both'")
        if not args.unwatermarked_config:
            raise ValueError("--unwatermarked-config is required when --mode is 'both'")
    elif args.mode == "unwatermarked":
        if not args.unwatermarked_config:
            raise ValueError("--unwatermarked-config is required when --mode is 'unwatermarked'")
    else:  # watermarked
        if not args.config:
            raise ValueError("--config is required when --mode is 'watermarked'")
    
    # Load configurations based on mode
    watermarked_config = None
    unwatermarked_config = None
    
    if args.mode == "both":
        logger.info(f"Loading watermarked config from {args.config}")
        watermarked_config = AppConfig.from_yaml(args.config)
        logger.info(f"Loading unwatermarked config from {args.unwatermarked_config}")
        unwatermarked_config = AppConfig.from_yaml(args.unwatermarked_config)
    elif args.mode == "unwatermarked":
        logger.info(f"Loading unwatermarked config from {args.unwatermarked_config}")
        unwatermarked_config = AppConfig.from_yaml(args.unwatermarked_config)
    else:  # watermarked
        logger.info(f"Loading config from {args.config}")
        watermarked_config = AppConfig.from_yaml(args.config)
    
    # Determine num_images
    if args.num_batches:
        num_images = args.num_batches * args.batch_size
        logger.info(f"Using --num-batches: {args.num_batches} batches × {args.batch_size} = {num_images} images")
    else:
        num_images = args.num_images
    
    # Load prompts
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    logger.info(f"Loading prompts from {prompts_file}")
    prompts = load_prompt_list(prompts_file)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Get device
    device = get_device(args.device, use_fp16=args.fp16)
    
    # Generate images
    generate_training_images(
        watermarked_config=watermarked_config,
        unwatermarked_config=unwatermarked_config,
        prompts=prompts,
        output_dir=Path(args.output_dir),
        mode=args.mode,
        batch_size=args.batch_size,
        num_images=num_images,
        key_id=args.key_id,
        master_key=args.master_key,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        use_fp16=args.fp16,
        use_torch_compile=args.torch_compile,
        logger=logger,
    )
    
    logger.info("✅ All done!")


if __name__ == "__main__":
    main()

