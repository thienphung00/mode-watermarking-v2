#!/usr/bin/env python3
"""
One-time preprocessing script to encode images to VAE latents.

This script:
1. Loads images from an existing manifest (JSON or JSONL)
2. Loads Stable Diffusion VAE once
3. Encodes each image to latents using VAE
4. Converts latents to fp16
5. Saves latents as .pt files
6. Creates a new latent_manifest.jsonl mapping latent_path → label → metadata

Usage:
    python scripts/preprocess_latents.py \
        --manifest path/to/manifest.jsonl \
        --output-dir path/to/latents \
        --config configs/experiments/unwatermarked.yaml
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from src.core.config import AppConfig
from src.engine.pipeline import create_pipeline
from scripts.utils import setup_logging, get_device


def load_manifest(manifest_path: str) -> list[Dict[str, Any]]:
    """
    Load manifest file (JSON or JSONL).
    
    Args:
        manifest_path: Path to manifest file
        
    Returns:
        List of manifest entries
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    entries = []
    if manifest_path.suffix == ".json":
        with open(manifest_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = data.get("samples", [data])
    elif manifest_path.suffix == ".jsonl":
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")
    
    return entries


def extract_label(entry: Dict[str, Any]) -> int:
    """
    Extract binary label from manifest entry.
    
    Args:
        entry: Manifest entry dictionary
        
    Returns:
        Label (0=unwatermarked, 1=watermarked)
    """
    # Priority 1: Explicit label field
    if "label" in entry:
        label = entry["label"]
        if isinstance(label, (int, bool)):
            return int(label)
        if isinstance(label, str):
            return 1 if label.lower() in ["watermarked", "1", "true"] else 0
    
    # Priority 2: Infer from image_path
    image_path = entry.get("image_path", "")
    if "watermarked" in str(image_path).lower():
        return 1
    if "unwatermarked" in str(image_path).lower():
        return 0
    
    # Priority 3: Infer from metadata mode
    metadata = entry.get("metadata", {})
    if isinstance(metadata, dict):
        mode = metadata.get("mode", "")
        if "distortion" in str(mode).lower() or "watermark" in str(mode).lower():
            return 1
    
    # Default: assume unwatermarked if unclear
    return 0


def encode_image_to_latent(
    image_path: Path,
    vae,
    device: str,
    vae_scale_factor: float = 0.18215,
) -> torch.Tensor:
    """
    Encode image to VAE latent.
    
    Args:
        image_path: Path to image file
        vae: VAE encoder model
        device: Device to run VAE on
        vae_scale_factor: VAE scaling factor (default: 0.18215)
        
    Returns:
        Latent tensor [4, 64, 64] in fp16 on CPU
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocessing transforms
    preprocess = T.Compose([
        T.Resize(512, interpolation=InterpolationMode.LANCZOS),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ])
    
    # Preprocess image
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Encode with VAE (on GPU)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        
        # Apply VAE scaling
        latent = latent * vae_scale_factor
        
        # Remove batch dimension and convert to fp16
        latent = latent.squeeze(0).half()
    
    # Move to CPU for storage
    latent = latent.cpu()
    
    return latent


def preprocess_latents(
    manifest_path: str,
    output_dir: str,
    config: AppConfig,
    device: str,
    seed: int = 42,
) -> None:
    """
    Preprocess images to latents.
    
    Args:
        manifest_path: Path to input manifest file
        output_dir: Directory to save latents and output manifest
        config: Application configuration
        device: Device to run VAE on
        seed: Random seed for deterministic encoding
    """
    logger = setup_logging()
    
    # Set seed for deterministic encoding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    entries = load_manifest(manifest_path)
    logger.info(f"Found {len(entries)} images to process")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for latent files
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VAE once
    logger.info("Loading Stable Diffusion pipeline...")
    pipeline = create_pipeline(config.diffusion, device=device)
    vae = pipeline.vae
    vae.eval()
    logger.info("VAE loaded and ready")
    
    # Get VAE scale factor from pipeline
    vae_scale_factor = getattr(pipeline, "vae_scale_factor", 0.18215)
    
    # Process each image
    latent_manifest_entries = []
    manifest_base = Path(manifest_path).parent
    
    logger.info("Encoding images to latents...")
    for i, entry in enumerate(tqdm(entries, desc="Encoding")):
        # Resolve image path
        image_path = entry.get("image_path") or entry.get("path")
        if image_path is None:
            logger.warning(f"Entry {i} has no image_path, skipping")
            continue
        
        image_path = Path(image_path)
        if not image_path.is_absolute():
            # Try relative to manifest directory first
            candidate = manifest_base / image_path
            if candidate.exists():
                image_path = candidate
            # If not found, try as-is (might be absolute after all)
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue
        
        # Encode image to latent
        try:
            latent = encode_image_to_latent(
                image_path=image_path,
                vae=vae,
                device=device,
                vae_scale_factor=vae_scale_factor,
            )
            
            # Validate latent shape
            assert latent.shape == (4, 64, 64), f"Unexpected latent shape: {latent.shape}, expected (4, 64, 64)"
            assert latent.dtype == torch.float16, f"Expected fp16, got {latent.dtype}"
            
            # Generate unique filename for latent
            latent_filename = f"{uuid.uuid4().hex[:16]}.pt"
            latent_path = latents_dir / latent_filename
            
            # Save latent tensor
            torch.save(latent, latent_path)
            
            # Extract label
            label = extract_label(entry)
            
            # Extract minimal metadata
            entry_metadata = entry.get("metadata", {})
            if not isinstance(entry_metadata, dict):
                entry_metadata = {}
            
            metadata = {
                "original_image_path": str(image_path),
            }
            
            # Add metadata fields if present
            for key in ["key_id", "g", "mode", "sample_id"]:
                if key in entry_metadata:
                    metadata[key] = entry_metadata[key]
            
            # Create latent manifest entry
            latent_entry = {
                "latent_path": str(latent_path.relative_to(output_dir)),
                "label": label,
                "metadata": metadata,
            }
            
            latent_manifest_entries.append(latent_entry)
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            continue
    
    # Save latent manifest
    latent_manifest_path = output_dir / "latent_manifest.jsonl"
    logger.info(f"Saving latent manifest to {latent_manifest_path}")
    with open(latent_manifest_path, "w") as f:
        for entry in latent_manifest_entries:
            f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Processed: {len(latent_manifest_entries)} images")
    logger.info(f"  Latents saved to: {latents_dir}")
    logger.info(f"  Manifest saved to: {latent_manifest_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess images to VAE latents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input manifest file (JSON or JSONL)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save latents and output manifest",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (for VAE loading)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic encoding (default: 42)",
    )
    
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
    logger.info("Latent Preprocessing")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = AppConfig.from_yaml(args.config)
    
    # Get device
    device = get_device(args.device, use_fp16=config.diffusion.use_fp16)
    logger.info(f"Device: {device}")
    
    # Run preprocessing
    preprocess_latents(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=config,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

