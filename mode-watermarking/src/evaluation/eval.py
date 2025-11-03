"""
Master evaluation orchestrator.

Coordinates detection evaluation and quality metrics evaluation,
generates comprehensive evaluation reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..config.config_loader import ConfigLoader
from ..data.dataset import load_split_manifest
from ..utils.io import ImageIO, ManifestIO, ensure_dir
from ..utils.logger import ExperimentLogger, LoggerConfig
from .quality_metrics import batch_compute_quality_metrics
from .visualize import generate_evaluation_report
from ..detection.recovery import batch_recover_g_values
from ..detection.correlate import batch_compute_s_statistics
from ..detection.calibrate import calibrate_thresholds, compute_detection_metrics


def run_detection_evaluation(
    test_manifest: str,
    detector_type: str,  # "unet" or "bayesian"
    detector_checkpoint: str,
    watermark_cfg: Dict[str, Any],
    diffusion_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    sd_pipeline: Optional[Any] = None,  # SD pipeline for VAE access
    output_dir: str = "outputs/evaluation",
    batch_size: int = 16,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run detection evaluation pipeline.
    
    Pipeline:
    1. Load test images from manifest
    2. Recover g-values (detection/recovery.py)
    3. Compute S-statistics (detection/correlate.py)
    4. Calibrate thresholds and compute metrics (detection/calibrate.py)
    
    Args:
        test_manifest: Path to test manifest file
        detector_type: Type of detector ("unet" or "bayesian")
        detector_checkpoint: Path to detector checkpoint
        watermark_cfg: Watermark configuration
        diffusion_cfg: Diffusion configuration
        eval_cfg: Evaluation configuration
        sd_pipeline: SD pipeline for VAE encoder access
        output_dir: Output directory for results
        batch_size: Batch size for processing
        device: Device for computation
    
    Returns:
        Detection evaluation results dictionary
    """
    ensure_dir(output_dir)
    
    # Load manifest
    manifest_entries = ManifestIO.read_jsonl(test_manifest)
    
    # Get VAE encoder from SD pipeline
    if sd_pipeline is None:
        raise ValueError("SD pipeline required for VAE encoder access")
    vae_encoder = sd_pipeline.vae
    
    # Recovery settings
    recovery_cfg = eval_cfg.get("detection", {}).get("recovery", {})
    
    # Recover g-values
    print("Recovering g-values from images...")
    recovery_results = batch_recover_g_values(
        manifest_path=test_manifest,
        vae_encoder=vae_encoder,
        watermark_cfg=watermark_cfg,
        diffusion_cfg=diffusion_cfg,
        batch_size=batch_size,
        num_workers=eval_cfg.get("test_data", {}).get("num_workers", 2),
        device=device,
        save_results=True,
        output_dir=str(Path(output_dir) / "recovery")
    )
    
    # Compute S-statistics
    print("Computing S-statistics...")
    correlation_cfg = eval_cfg.get("detection", {}).get("correlation", {})
    s_scores = batch_compute_s_statistics(
        recovery_results=recovery_results,
        watermark_cfg=watermark_cfg,
        method=correlation_cfg.get("method", "correlation")
    )
    
    # Extract labels from manifest
    labels = []
    for entry in manifest_entries:
        # Label is typically in manifest or can be inferred from filename/path
        # For now, assume manifest has "is_watermarked" field
        is_watermarked = entry.get("is_watermarked", entry.get("label", 0))
        labels.append(1 if is_watermarked else 0)
    labels = np.array(labels)
    
    # Calibrate thresholds
    print("Calibrating thresholds...")
    calibration_cfg = eval_cfg.get("detection", {}).get("calibration", {})
    calibration_results = calibrate_thresholds(
        s_scores=s_scores,
        labels=labels,
        target_fpr=calibration_cfg.get("target_fpr", 0.01),
        significance_level=calibration_cfg.get("significance_level", 0.05)
    )
    
    # Compute detection metrics at operating threshold
    detection_metrics = compute_detection_metrics(
        s_scores=s_scores,
        labels=labels,
        threshold=calibration_results["threshold"]
    )
    
    # Combine results
    detection_results = {
        "s_scores": s_scores.tolist(),
        "labels": labels.tolist(),
        "calibration": calibration_results,
        "metrics": detection_metrics
    }
    
    # Save results
    import json
    results_path = Path(output_dir) / "detection_results.json"
    with open(results_path, "w") as f:
        json.dump(detection_results, f, indent=2)
    
    print(f"Detection evaluation complete. Results saved to {results_path}")
    
    return detection_results


def run_quality_evaluation(
    test_manifest: str,
    eval_cfg: Dict[str, Any],
    output_dir: str = "outputs/evaluation",
    batch_size: int = 16,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run quality metrics evaluation pipeline.
    
    Pipeline:
    1. Load watermarked + original image pairs from manifest
    2. Compute quality metrics (evaluation/quality_metrics.py)
    3. Aggregate statistics (mean, std, per-mode)
    
    Args:
        test_manifest: Path to test manifest file
        eval_cfg: Evaluation configuration
        output_dir: Output directory
        batch_size: Batch size for processing
        device: Device for computation
    
    Returns:
        Quality evaluation results dictionary
    """
    ensure_dir(output_dir)
    
    # Load manifest
    manifest_entries = ManifestIO.read_jsonl(test_manifest)
    
    # Get metrics to compute
    metrics = eval_cfg.get("quality", {}).get("metrics", ["psnr", "ssim", "lpips"])
    
    # Load image pairs
    watermarked_images = []
    original_images = []
    modes = []  # Track mode for each image
    
    for entry in manifest_entries:
        wm_path = entry.get("image_path")
        orig_path = entry.get("original_image_path")  # Assume manifest has this
        
        if not wm_path or not orig_path:
            continue
        
        # Load images
        wm_img = ImageIO.read_image(wm_path)
        orig_img = ImageIO.read_image(orig_path)
        
        wm_img = Image.fromarray(wm_img)
        orig_img = Image.fromarray(orig_img)
        
        watermarked_images.append(wm_img)
        original_images.append(orig_img)
        
        # Track mode
        mode = entry.get("mode", "non_distortionary")
        modes.append(mode)
    
    # Compute quality metrics
    print(f"Computing quality metrics for {len(watermarked_images)} image pairs...")
    quality_results = batch_compute_quality_metrics(
        watermarked_images=watermarked_images,
        original_images=original_images,
        metrics=metrics,
        batch_size=batch_size,
        device=device
    )
    
    # Aggregate by mode
    modes_array = np.array(modes)
    quality_by_mode = {}
    
    for mode in set(modes):
        mode_mask = modes_array == mode
        mode_results = {}
        
        for metric_name, metric_values in quality_results.items():
            if isinstance(metric_values, np.ndarray):
                mode_values = metric_values[mode_mask]
                mode_results[metric_name] = {
                    "mean": float(np.nanmean(mode_values)),
                    "std": float(np.nanstd(mode_values)),
                    "min": float(np.nanmin(mode_values)),
                    "max": float(np.nanmax(mode_values)),
                    "values": mode_values.tolist()
                }
            else:
                # Single value (e.g., FID)
                mode_results[metric_name] = metric_values
        
        quality_by_mode[mode] = mode_results
    
    # Overall statistics
    overall_results = {}
    for metric_name, metric_values in quality_results.items():
        if isinstance(metric_values, np.ndarray):
            overall_results[metric_name] = {
                "mean": float(np.nanmean(metric_values)),
                "std": float(np.nanstd(metric_values)),
                "values": metric_values.tolist()
            }
        else:
            overall_results[metric_name] = metric_values
    
    results = {
        "overall": overall_results,
        "by_mode": quality_by_mode,
        "num_images": len(watermarked_images)
    }
    
    # Save results
    import json
    results_path = Path(output_dir) / "quality_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Quality evaluation complete. Results saved to {results_path}")
    
    return results


def run_full_evaluation(
    test_manifest: str,
    eval_config_path: Optional[str] = None,
    watermark_cfg_path: Optional[str] = None,
    diffusion_cfg_path: Optional[str] = None,
    model_arch_cfg_path: Optional[str] = None,
    sd_pipeline: Optional[Any] = None,
    output_dir: str = "outputs/evaluation"
) -> Dict[str, Any]:
    """
    Master evaluation orchestrator.
    
    Combines detection + quality evaluation + visualization.
    
    Args:
        test_manifest: Path to test manifest file
        eval_config_path: Path to eval_config.yaml (default: configs/eval_config.yaml)
        watermark_cfg_path: Path to watermark_config.yaml (default: configs/watermark_config.yaml)
        diffusion_cfg_path: Path to diffusion_config.yaml (default: configs/diffusion_config.yaml)
        model_arch_cfg_path: Path to model_architecture.yaml (default: configs/model_architecture.yaml)
        sd_pipeline: SD pipeline instance (optional, will be created if not provided)
        output_dir: Output directory for all results
    
    Returns:
        Comprehensive evaluation report dictionary
    """
    # Default config paths
    if eval_config_path is None:
        eval_config_path = "configs/eval_config.yaml"
    if watermark_cfg_path is None:
        watermark_cfg_path = "configs/watermark_config.yaml"
    if diffusion_cfg_path is None:
        diffusion_cfg_path = "configs/diffusion_config.yaml"
    if model_arch_cfg_path is None:
        model_arch_cfg_path = "configs/model_architecture.yaml"
    
    # Load configs
    config_loader = ConfigLoader()
    eval_cfg = config_loader.load_yaml(eval_config_path)
    watermark_cfg = config_loader.load_yaml(watermark_cfg_path)
    diffusion_cfg = config_loader.load_yaml(diffusion_cfg_path)
    
    ensure_dir(output_dir)
    
    # Initialize logger
    log_cfg = eval_cfg.get("visualization", {}).get("logging", {})
    logger_config = LoggerConfig(
        backend="tensorboard",
        project=log_cfg.get("wandb_project"),
        run_name="evaluation",
        log_dir=str(Path(output_dir) / "logs"),
        tags=["evaluation", "detection", "quality"]
    )
    logger = ExperimentLogger(logger_config)
    
    # Run detection evaluation
    detection_results = None
    if eval_cfg.get("detection", {}).get("enabled", True):
        detector_cfg = eval_cfg.get("detector", {})
        detection_results = run_detection_evaluation(
            test_manifest=test_manifest,
            detector_type=detector_cfg.get("detector_type", "unet"),
            detector_checkpoint=detector_cfg.get("checkpoint_path"),
            watermark_cfg=watermark_cfg,
            diffusion_cfg=diffusion_cfg,
            eval_cfg=eval_cfg,
            sd_pipeline=sd_pipeline,
            output_dir=str(Path(output_dir) / "detection"),
            batch_size=eval_cfg.get("test_data", {}).get("batch_size", 16),
            device=detector_cfg.get("device", "cuda")
        )
    
    # Run quality evaluation
    quality_results = None
    if eval_cfg.get("quality", {}).get("enabled", True):
        quality_results = run_quality_evaluation(
            test_manifest=test_manifest,
            eval_cfg=eval_cfg,
            output_dir=str(Path(output_dir) / "quality"),
            batch_size=eval_cfg.get("test_data", {}).get("batch_size", 16),
            device=eval_cfg.get("detector", {}).get("device", "cuda")
        )
    
    # Generate visualization and report
    if eval_cfg.get("visualization", {}).get("generate_plots", True):
        report_path = generate_evaluation_report(
            detection_results=detection_results,
            quality_results=quality_results,
            output_dir=output_dir,
            eval_cfg=eval_cfg
        )
        print(f"Evaluation report generated: {report_path}")
    
    # Combine all results
    full_report = {
        "detection": detection_results,
        "quality": quality_results,
        "config": {
            "eval_config": eval_cfg,
            "watermark_config": watermark_cfg,
            "diffusion_config": diffusion_cfg
        },
        "output_dir": output_dir
    }
    
    # Save full report
    import json
    report_path = Path(output_dir) / "full_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    
    logger.finalize(status="completed")
    
    return full_report
