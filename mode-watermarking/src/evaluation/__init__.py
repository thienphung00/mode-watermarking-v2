# Evaluation utilities for detection and quality metrics

from .quality_metrics import (
    compute_quality_metrics,
    batch_compute_quality_metrics,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_fid,
    compute_clip_similarity,
)
from .eval import (
    run_detection_evaluation,
    run_quality_evaluation,
    run_full_evaluation,
)
from .visualize import (
    plot_roc_curve,
    plot_pr_curve,
    plot_score_distributions,
    generate_detection_heatmap,
    plot_quality_histograms,
    plot_mode_comparison,
    generate_evaluation_report,
)

__all__ = [
    # Quality metrics
    "compute_quality_metrics",
    "batch_compute_quality_metrics",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_fid",
    "compute_clip_similarity",
    # Evaluation runners
    "run_detection_evaluation",
    "run_quality_evaluation",
    "run_full_evaluation",
    # Visualization
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_score_distributions",
    "generate_detection_heatmap",
    "plot_quality_histograms",
    "plot_mode_comparison",
    "generate_evaluation_report",
]
