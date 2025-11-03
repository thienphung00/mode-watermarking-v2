"""
CLI entry point for detector training.

Usage:
    python -m src.cli.train --detector unet --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config_loader import ConfigLoader
from src.training.train import train_unet_detector, train_bayesian_detector
from src.sd_integration.sd_client import SDClient


def train_detector(
    detector_type: str,
    train_config_path: str,
    config_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda"
) -> None:
    """
    Train detector model.
    
    Args:
        detector_type: "unet" or "bayesian"
        train_config_path: Path to train_config.yaml
        config_dir: Directory containing other configs (default: same as train_config parent)
        resume_from_checkpoint: Path to checkpoint to resume from
        output_dir: Override output directory from config
        device: Device for training
    """
    # Determine config directory
    if config_dir is None:
        config_dir = str(Path(train_config_path).parent)
    
    # Load train config
    config_loader = ConfigLoader()
    train_config = config_loader.load_yaml(train_config_path)
    
    # Update device if specified
    if "common" in train_config:
        train_config["common"]["device"] = device
    
    # Prepare config paths
    config_paths = {
        "train": train_config_path
    }
    
    # For UNet training, may need SD pipeline for VAE access
    sd_pipeline = None
    if detector_type == "unet":
        # Initialize SD pipeline if needed for VAE encoder
        try:
            sd_config_paths = {
                "diffusion": str(Path(config_dir) / "diffusion_config.yaml"),
                "watermark": str(Path(config_dir) / "watermark_config.yaml"),
                "model": str(Path(config_dir) / "model_architecture.yaml")
            }
            sd_client = SDClient(config_paths=sd_config_paths, device=device)
            sd_client.initialize_pipeline()
            sd_pipeline = sd_client._pipeline
        except Exception as e:
            print(f"Warning: Could not initialize SD pipeline for VAE access: {e}")
            print("Training will proceed without VAE encoder (latent-based training disabled)")
    
    # Train detector
    print(f"\n{'='*60}")
    print(f"Training {detector_type.upper()} detector")
    print(f"{'='*60}\n")
    
    if detector_type == "unet":
        results = train_unet_detector(
            config_paths=config_paths,
            sd_pipeline=sd_pipeline,
            resume_from_checkpoint=resume_from_checkpoint
        )
    elif detector_type == "bayesian":
        results = train_bayesian_detector(
            config_paths=config_paths,
            resume_from_checkpoint=resume_from_checkpoint
        )
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best checkpoint: {results.get('best_checkpoint', 'N/A')}")
    print(f"Final metrics: {results.get('final_metrics', {})}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train watermark detector"
    )
    parser.add_argument(
        "--detector",
        choices=["unet", "bayesian"],
        required=True,
        help="Detector type to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to train_config.yaml"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing other configs (default: same as train config parent)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Check GPU availability
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Using CPU.")
                args.device = "cpu"
        except ImportError:
            print("Warning: PyTorch not available. Using CPU.")
            args.device = "cpu"
    
    # Train detector
    try:
        train_detector(
            detector_type=args.detector,
            train_config_path=args.config,
            config_dir=args.config_dir,
            resume_from_checkpoint=args.resume_from,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

