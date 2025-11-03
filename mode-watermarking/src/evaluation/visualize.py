"""
Visualization utilities for evaluation results.

Generates ROC/PR curves, score distributions, quality histograms,
and comprehensive evaluation reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None


def plot_roc_curve(
    fpr_values: np.ndarray,
    tpr_values: np.ndarray,
    auc: float,
    save_path: str,
    title: str = "ROC Curve",
    label: Optional[str] = None
) -> None:
    """
    Plot ROC curve with AUC annotation.
    
    Args:
        fpr_values: False Positive Rate values
        tpr_values: True Positive Rate values
        auc: Area Under Curve value
        save_path: Path to save plot
        title: Plot title
        label: Optional label for legend
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping ROC plot")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, label=label or f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_pr_curve(
    precision_values: np.ndarray,
    recall_values: np.ndarray,
    auc_pr: float,
    save_path: str,
    title: str = "Precision-Recall Curve",
    label: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision_values: Precision values
        recall_values: Recall values
        auc_pr: Area Under PR Curve
        save_path: Path to save plot
        title: Plot title
        label: Optional label for legend
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping PR plot")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, label=label or f"AUC-PR = {auc_pr:.3f}", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {save_path}")


def plot_score_distributions(
    watermarked_scores: np.ndarray,
    unwatermarked_scores: np.ndarray,
    threshold: Optional[float] = None,
    save_path: str,
    title: str = "Detection Score Distributions"
) -> None:
    """
    Plot histogram of S-scores for watermarked vs unwatermarked images.
    
    Args:
        watermarked_scores: Detection scores for watermarked images
        unwatermarked_scores: Detection scores for unwatermarked images
        threshold: Optional threshold line to plot
        save_path: Path to save plot
        title: Plot title
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping score distribution plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(
        unwatermarked_scores,
        bins=50,
        alpha=0.7,
        label='Unwatermarked',
        color='red',
        edgecolor='black'
    )
    plt.hist(
        watermarked_scores,
        bins=50,
        alpha=0.7,
        label='Watermarked',
        color='blue',
        edgecolor='black'
    )
    
    # Plot threshold line if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    
    plt.xlabel('Detection Score (S-statistic)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved score distribution to {save_path}")


def generate_detection_heatmap(
    g_values: np.ndarray,  # [C, H, W]
    mask: Optional[np.ndarray] = None,
    save_path: str,
    title: str = "G-Value Spatial Heatmap"
) -> None:
    """
    Generate spatial heatmap of g-value magnitudes.
    
    Args:
        g_values: G-values tensor [C, H, W]
        mask: Optional spatial mask [C, H, W]
        save_path: Path to save heatmap
        title: Plot title
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping heatmap")
        return
    
    # Aggregate across channels (mean or sum)
    if g_values.ndim == 3:
        g_spatial = np.mean(np.abs(g_values), axis=0)  # [H, W]
    else:
        raise ValueError(f"Expected 3D array [C, H, W], got {g_values.shape}")
    
    # Apply mask if provided
    if mask is not None:
        if mask.ndim == 3:
            mask_spatial = np.mean(mask, axis=0)
        else:
            mask_spatial = mask
        g_spatial = g_spatial * mask_spatial
    
    plt.figure(figsize=(10, 10))
    plt.imshow(g_spatial, cmap='hot', interpolation='nearest')
    plt.colorbar(label='G-value Magnitude')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")


def plot_quality_histograms(
    quality_results: Dict[str, Any],
    save_dir: str,
    mode: Optional[str] = None
) -> None:
    """
    Plot histograms of quality metrics.
    
    Args:
        quality_results: Quality results dictionary from run_quality_evaluation
        save_dir: Directory to save plots
        mode: Optional mode to filter ("non_distortionary" or "distortionary")
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping quality histograms")
        return
    
    # Get data by mode or overall
    if mode and mode in quality_results.get("by_mode", {}):
        data = quality_results["by_mode"][mode]
    else:
        data = quality_results.get("overall", {})
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot each metric
    for metric_name, metric_data in data.items():
        if isinstance(metric_data, dict) and "values" in metric_data:
            values = np.array(metric_data["values"])
            values = values[~np.isnan(values)]  # Remove NaN
            
            if len(values) == 0:
                continue
            
            plt.figure(figsize=(8, 6))
            plt.hist(values, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel(metric_name.upper(), fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'{metric_name.upper()} Distribution', fontsize=14, fontweight='bold')
            
            # Add statistics
            mean_val = metric_data.get("mean", np.mean(values))
            std_val = metric_data.get("std", np.std(values))
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
            plt.text(0.7, 0.9, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = Path(save_dir) / f"{metric_name}_histogram.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {metric_name} histogram to {save_path}")


def plot_mode_comparison(
    detection_results: Dict[str, Any],
    quality_results: Dict[str, Any],
    save_path: str
) -> None:
    """
    Plot comparison between non-distortionary and distortionary modes.
    
    Args:
        detection_results: Detection evaluation results
        quality_results: Quality evaluation results
        save_path: Path to save comparison plot
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping mode comparison")
        return
    
    quality_by_mode = quality_results.get("by_mode", {})
    
    if len(quality_by_mode) < 2:
        print("Warning: Need at least 2 modes for comparison")
        return
    
    # Extract metrics for comparison
    metrics_to_compare = ["psnr", "ssim", "lpips"]
    modes = list(quality_by_mode.keys())
    
    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(15, 5))
    if len(metrics_to_compare) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_compare):
        mode_values = []
        mode_labels = []
        
        for mode in modes:
            if metric in quality_by_mode[mode]:
                metric_data = quality_by_mode[mode][metric]
                if isinstance(metric_data, dict) and "values" in metric_data:
                    values = np.array(metric_data["values"])
                    values = values[~np.isnan(values)]
                    mode_values.append(values)
                    mode_labels.append(mode.replace("_", " ").title())
        
        if mode_values:
            axes[idx].boxplot(mode_values, labels=mode_labels)
            axes[idx].set_ylabel(metric.upper(), fontsize=12)
            axes[idx].set_title(f'{metric.upper()} by Mode', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mode comparison to {save_path}")


def generate_evaluation_report(
    detection_results: Optional[Dict[str, Any]],
    quality_results: Optional[Dict[str, Any]],
    output_dir: str,
    eval_cfg: Optional[Dict[str, Any]] = None,
    mode: str = "non_distortionary"
) -> str:
    """
    Generate comprehensive evaluation report (markdown format).
    
    Args:
        detection_results: Detection evaluation results
        quality_results: Quality evaluation results
        output_dir: Output directory
        eval_cfg: Evaluation configuration (optional)
        mode: Evaluation mode
    
    Returns:
        Path to generated report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / "evaluation_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Watermark Detection & Quality Evaluation Report\n\n")
        f.write(f"Generated for mode: **{mode}**\n\n")
        f.write("---\n\n")
        
        # Detection Results Section
        if detection_results:
            f.write("## Detection Results\n\n")
            
            metrics = detection_results.get("metrics", {})
            f.write("### Detection Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |\n")
            f.write(f"| Precision | {metrics.get('precision', 'N/A'):.4f} |\n")
            f.write(f"| Recall | {metrics.get('recall', 'N/A'):.4f} |\n")
            f.write(f"| F1 Score | {metrics.get('f1_score', 'N/A'):.4f} |\n")
            f.write(f"| Threshold | {metrics.get('threshold', 'N/A'):.4f} |\n")
            f.write("\n")
            
            calibration = detection_results.get("calibration", {})
            f.write("### Calibration Results\n\n")
            f.write(f"- **TPR@1%FPR**: {calibration.get('tpr_at_target_fpr', 'N/A'):.4f}\n")
            f.write(f"- **AUC-ROC**: {calibration.get('auc_roc', 'N/A'):.4f}\n")
            f.write(f"- **AUC-PR**: {calibration.get('auc_pr', 'N/A'):.4f}\n")
            f.write("\n")
            
            # Plot ROC and PR curves
            if eval_cfg and eval_cfg.get("visualization", {}).get("generate_plots", True):
                roc_curve_data = calibration.get("roc_curve")
                pr_curve_data = calibration.get("pr_curve")
                
                if roc_curve_data:
                    plot_roc_curve(
                        np.array(roc_curve_data[0]),
                        np.array(roc_curve_data[1]),
                        calibration.get("auc_roc", 0.0),
                        str(output_path / "roc_curve.png"),
                        title="ROC Curve"
                    )
                    f.write("![ROC Curve](roc_curve.png)\n\n")
                
                if pr_curve_data:
                    plot_pr_curve(
                        np.array(pr_curve_data[0]),
                        np.array(pr_curve_data[1]),
                        calibration.get("auc_pr", 0.0),
                        str(output_path / "pr_curve.png"),
                        title="Precision-Recall Curve"
                    )
                    f.write("![PR Curve](pr_curve.png)\n\n")
                
                # Score distributions
                if "s_scores" in detection_results and "labels" in detection_results:
                    s_scores = np.array(detection_results["s_scores"])
                    labels = np.array(detection_results["labels"])
                    wm_scores = s_scores[labels == 1]
                    uwm_scores = s_scores[labels == 0]
                    
                    plot_score_distributions(
                        wm_scores,
                        uwm_scores,
                        threshold=metrics.get("threshold"),
                        save_path=str(output_path / "score_distributions.png"),
                        title="Detection Score Distributions"
                    )
                    f.write("![Score Distributions](score_distributions.png)\n\n")
            
            f.write("---\n\n")
        
        # Quality Results Section
        if quality_results:
            f.write("## Quality Metrics\n\n")
            
            overall = quality_results.get("overall", {})
            f.write("### Overall Statistics\n\n")
            f.write("| Metric | Mean | Std |\n")
            f.write("|--------|------|-----|\n")
            
            for metric_name, metric_data in overall.items():
                if isinstance(metric_data, dict):
                    mean_val = metric_data.get("mean", "N/A")
                    std_val = metric_data.get("std", "N/A")
                    if isinstance(mean_val, float):
                        f.write(f"| {metric_name.upper()} | {mean_val:.4f} | {std_val:.4f} |\n")
                    else:
                        f.write(f"| {metric_name.upper()} | {mean_val} | {std_val} |\n")
            f.write("\n")
            
            # By mode comparison
            by_mode = quality_results.get("by_mode", {})
            if len(by_mode) > 1:
                f.write("### Mode Comparison\n\n")
                plot_mode_comparison(
                    detection_results or {},
                    quality_results,
                    str(output_path / "mode_comparison.png")
                )
                f.write("![Mode Comparison](mode_comparison.png)\n\n")
            
            # Quality histograms
            if eval_cfg and eval_cfg.get("visualization", {}).get("generate_plots", True):
                plot_quality_histograms(
                    quality_results,
                    str(output_path / "quality_histograms"),
                    mode=mode
                )
                f.write("### Quality Metric Distributions\n\n")
                f.write("See `quality_histograms/` directory for individual metric histograms.\n\n")
            
            f.write("---\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        if detection_results:
            calibration = detection_results.get("calibration", {})
            tpr_at_fpr = calibration.get("tpr_at_target_fpr", 0.0)
            
            if tpr_at_fpr > 0.9:
                f.write("- ✅ Excellent detection performance (TPR@1%FPR > 90%)\n")
            elif tpr_at_fpr > 0.7:
                f.write("- ⚠️ Good detection performance, consider fine-tuning watermark strength\n")
            else:
                f.write("- ❌ Detection performance below target. Review watermark embedding strategy.\n")
        
        if quality_results:
            overall = quality_results.get("overall", {})
            psnr_mean = overall.get("psnr", {}).get("mean", 0.0) if isinstance(overall.get("psnr"), dict) else 0.0
            
            if psnr_mean > 30.0:
                f.write("- ✅ High image quality (PSNR > 30dB)\n")
            elif psnr_mean > 25.0:
                f.write("- ⚠️ Acceptable image quality, may benefit from non-distortionary mode\n")
            else:
                f.write("- ❌ Image quality below target. Consider reducing watermark strength.\n")
    
    print(f"Evaluation report generated: {report_path}")
    return str(report_path)
