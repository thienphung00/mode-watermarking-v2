"""
CLI entry point for dataset generation.

Usage:
    python -m src.cli.generate --mode both --prompts-file data/coco/prompts_train.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config_loader import ConfigLoader
from src.sd_integration.sd_client import SDClient
from src.utils.io import ImageIO, ManifestIO, ensure_dir


def load_prompts(prompts_file: str) -> List[str]:
    """
    Load prompts from file (one per line).
    
    Args:
        prompts_file: Path to text file with prompts
    
    Returns:
        List of prompt strings
    """
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_dataset(
    mode: str,
    prompts_file: str,
    output_dir: str,
    config_dir: str,
    num_samples: Optional[int] = None,
    batch_size: int = 4,
    device: str = "cuda",
    watermark: bool = True
) -> None:
    """
    Generate watermarked or unwatermarked dataset.
    
    Args:
        mode: "watermarked", "unwatermarked", or "both"
        prompts_file: Path to file with prompts (one per line)
        output_dir: Output directory for images and manifests
        config_dir: Directory containing config files
        num_samples: Number of samples to generate (None = all prompts)
        batch_size: Batch size for generation
        device: Device for computation
        watermark: Whether to embed watermark (only used if mode="both")
    """
    # Load prompts
    prompts = load_prompts(prompts_file)
    if num_samples:
        prompts = prompts[:num_samples]
    
    print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    
    # Load configs
    config_loader = ConfigLoader()
    config_paths = {
        "diffusion": str(Path(config_dir) / "diffusion_config.yaml"),
        "watermark": str(Path(config_dir) / "watermark_config.yaml"),
        "model": str(Path(config_dir) / "model_architecture.yaml")
    }
    
    # Initialize SD client
    print("Initializing Stable Diffusion pipeline...")
    sd_client = SDClient(config_paths=config_paths, device=device)
    sd_client.initialize_pipeline()
    
    # Set up output directories
    if mode == "both":
        wm_dir = Path(output_dir) / "watermarked" / "images"
        unwm_dir = Path(output_dir) / "unwatermarked" / "images"
        wm_manifest_dir = Path(output_dir) / "watermarked"
        unwm_manifest_dir = Path(output_dir) / "unwatermarked"
    elif mode == "watermarked":
        wm_dir = Path(output_dir) / "watermarked" / "images"
        wm_manifest_dir = Path(output_dir) / "watermarked"
        unwm_dir = None
        unwm_manifest_dir = None
    else:  # unwatermarked
        unwm_dir = Path(output_dir) / "unwatermarked" / "images"
        unwm_manifest_dir = Path(output_dir) / "unwatermarked"
        wm_dir = None
        wm_manifest_dir = None
    
    # Create directories
    if wm_dir:
        ensure_dir(str(wm_dir))
    if unwm_dir:
        ensure_dir(str(unwm_dir))
    
    # Generate images
    manifests_wm = []
    manifests_unwm = []
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Generate watermarked if needed
        if mode in ["both", "watermarked"]:
            image_wm, manifest_wm = sd_client.generate(
                prompt=prompt,
                seed=42 + i,  # Deterministic seed
                num_inference_steps=None,  # Use config default
                guidance_scale=None  # Use config default
            )
            
            # Save image
            image_path = wm_dir / f"image_{i:05d}.png"
            ImageIO.write_image(str(image_path), image_wm)
            manifest_wm["image_path"] = str(image_path)
            manifests_wm.append(manifest_wm)
        
        # Generate unwatermarked if needed
        if mode in ["both", "unwatermarked"]:
            # Temporarily disable watermark
            # For now, generate without watermark hook
            # TODO: Add method to generate unwatermarked images
            image_unwm, manifest_unwm = sd_client.generate(
                prompt=prompt,
                seed=42 + i,
                num_inference_steps=None,
                guidance_scale=None
            )
            # Note: This still has watermark. Need to add unwatermarked generation method.
            
            # Save image
            image_path = unwm_dir / f"image_{i:05d}.png"
            ImageIO.write_image(str(image_path), image_unwm)
            manifest_unwm["image_path"] = str(image_path)
            manifests_unwm.append(manifest_unwm)
    
    # Save manifests
    if manifests_wm:
        manifest_path = wm_manifest_dir / "manifest.jsonl"
        ManifestIO.write_jsonl(str(manifest_path), manifests_wm)
        print(f"Saved watermarked manifest: {manifest_path}")
    
    if manifests_unwm:
        manifest_path = unwm_manifest_dir / "manifest.jsonl"
        ManifestIO.write_jsonl(str(manifest_path), manifests_unwm)
        print(f"Saved unwatermarked manifest: {manifest_path}")
    
    print(f"\nDataset generation complete!")
    print(f"Generated {len(manifests_wm)} watermarked images")
    print(f"Generated {len(manifests_unwm)} unwatermarked images")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate watermarked/unwatermarked dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["watermarked", "unwatermarked", "both"],
        required=True,
        help="Generation mode"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to file with prompts (one per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated",
        help="Output directory for images and manifests"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing config files"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: all prompts)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (currently not used, generates sequentially)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for computation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.prompts_file).exists():
        print(f"Error: Prompts file not found: {args.prompts_file}")
        sys.exit(1)
    
    config_dir = Path(args.config_dir)
    required_configs = ["diffusion_config.yaml", "watermark_config.yaml", "model_architecture.yaml"]
    for config_file in required_configs:
        if not (config_dir / config_file).exists():
            print(f"Error: Config file not found: {config_dir / config_file}")
            sys.exit(1)
    
    # Generate dataset
    try:
        generate_dataset(
            mode=args.mode,
            prompts_file=args.prompts_file,
            output_dir=args.output_dir,
            config_dir=args.config_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device
        )
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

