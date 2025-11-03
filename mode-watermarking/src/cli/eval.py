"""
CLI entry point for evaluation.

Usage:
    python -m src.cli.eval --test-manifest data/splits/test.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.config_loader import ConfigLoader
from src.evaluation.eval import run_full_evaluation
from src.sd_integration.sd_client import SDClient


def run_evaluation(
    test_manifest: str,
    eval_config_path: str,
    config_dir: Optional[str] = None,
    detector_type: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda"
) -> None:
    """
    Run full evaluation pipeline.
    
    Args:
        test_manifest: Path to test manifest file
        eval_config_path: Path to eval_config.yaml
        config_dir: Directory containing other configs (default: same as eval config parent)
        detector_type: Override detector type ("unet" or "bayesian")
        checkpoint_path: Override checkpoint path
        output_dir: Override output directory
        device: Device for computation
    """
    # Determine config directory
    if config_dir is None:
        config_dir = str(Path(eval_config_path).parent)
    
    # Paths for required configs
    watermark_cfg_path = str(Path(config_dir) / "watermark_config.yaml")
    diffusion_cfg_path = str(Path(config_dir) / "diffusion_config.yaml")
    model_cfg_path = str(Path(config_dir) / "model_architecture.yaml")
    
    # Initialize SD pipeline for VAE access
    print("Initializing Stable Diffusion pipeline...")
    try:
        sd_config_paths = {
            "diffusion": diffusion_cfg_path,
            "watermark": watermark_cfg_path,
            "model": model_cfg_path
        }
        sd_client = SDClient(config_paths=sd_config_paths, device=device)
        sd_client.initialize_pipeline()
        sd_pipeline = sd_client._pipeline
    except Exception as e:
        print(f"Warning: Could not initialize SD pipeline: {e}")
        sd_pipeline = None
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("Running Evaluation Pipeline")
    print(f"{'='*60}\n")
    
    results = run_full_evaluation(
        test_manifest=test_manifest,
        eval_config_path=eval_config_path,
        watermark_cfg_path=watermark_cfg_path,
        diffusion_cfg_path=diffusion_cfg_path,
        model_arch_cfg_path=model_cfg_path,
        sd_pipeline=sd_pipeline,
        output_dir=output_dir or "outputs/evaluation"
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results.get('output_dir', 'N/A')}")
    if results.get("detection"):
        detection = results["detection"]
        metrics = detection.get("metrics", {})
        print(f"\nDetection Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if isinstance(metrics.get('accuracy'), float) else f"  Accuracy: N/A")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), float) else f"  Precision: N/A")
        print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), float) else f"  Recall: N/A")
        calibration = detection.get("calibration", {})
        print(f"  TPR@1%FPR: {calibration.get('tpr_at_target_fpr', 'N/A'):.4f}" if isinstance(calibration.get('tpr_at_target_fpr'), float) else f"  TPR@1%FPR: N/A")
        print(f"  AUC-ROC: {calibration.get('auc_roc', 'N/A'):.4f}" if isinstance(calibration.get('auc_roc'), float) else f"  AUC-ROC: N/A")
    if results.get("quality"):
        quality = results["quality"]
        overall = quality.get("overall", {})
        print(f"\nQuality Metrics:")
        for metric_name, metric_data in overall.items():
            if isinstance(metric_data, dict) and "mean" in metric_data:
                print(f"  {metric_name.upper()}: {metric_data['mean']:.4f} ± {metric_data.get('std', 0):.4f}")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline"
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        required=True,
        help="Path to test manifest file"
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing configs (default: same as eval config parent)"
    )
    parser.add_argument(
        "--detector-type",
        type=str,
        choices=["unet", "bayesian"],
        default=None,
        help="Override detector type from config"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
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
    if not Path(args.test_manifest).exists():
        print(f"Error: Test manifest not found: {args.test_manifest}")
        sys.exit(1)
    
    if not Path(args.eval_config).exists():
        print(f"Error: Eval config not found: {args.eval_config}")
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
    
    # Run evaluation
    try:
        run_evaluation(
            test_manifest=args.test_manifest,
            eval_config_path=args.eval_config,
            config_dir=args.config_dir,
            detector_type=args.detector_type,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

