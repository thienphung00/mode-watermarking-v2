# Detection utilities for watermark recovery and statistical analysis

from .recovery import recover_g_values, batch_recover_g_values
from .correlate import (
    compute_s_statistic,
    batch_compute_s_statistics,
    compute_correlation_statistics,
)
from .calibrate import (
    calibrate_thresholds,
    compute_detection_metrics,
    tpr_at_fpr,
)

__all__ = [
    # Recovery
    "recover_g_values",
    "batch_recover_g_values",
    # Correlation
    "compute_s_statistic",
    "batch_compute_s_statistics",
    "compute_correlation_statistics",
    # Calibration
    "calibrate_thresholds",
    "compute_detection_metrics",
    "tpr_at_fpr",
]
