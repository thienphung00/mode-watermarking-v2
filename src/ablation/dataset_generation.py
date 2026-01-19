"""
Dataset generation for ablation experiments.

Reusable functions for generating watermarked and clean datasets
with caching support.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..core.config import AppConfig, UnwatermarkedConfig
from ..engine.pipeline import create_pipeline, generate_with_watermark
from ..engine.strategy_factory import create_strategy_from_config
from scripts.utils import setup_logging


logger = setup_logging()


def generate_ablation_dataset(
    config: AppConfig,
    prompts: List[str],
    output_dir: Path,
    num_samples: int,
    key_id: str,
    device: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate watermarked and clean datasets for ablation.
    
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

