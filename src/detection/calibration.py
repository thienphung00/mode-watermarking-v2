"""
Threshold calibration for watermark detection.

Provides routines to calibrate detection thresholds from empirical data.
While the SynthID S-statistic has known theoretical distribution under H0,
empirical calibration can account for:
- Model-specific biases
- Distribution shifts from image processing
- Practical FPR/TPR targets

Calibration approaches:
1. Theoretical: Use z_{1-α} from N(0,1) distribution
2. Empirical: Use percentile of unwatermarked S-statistics
3. ROC-based: Find threshold achieving target TPR at FPR constraint
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""
    
    threshold: float
    target_fpr: float
    achieved_fpr: float  # Actual FPR on calibration data
    achieved_tpr: float  # Actual TPR on calibration data
    method: str
    n_watermarked: int
    n_unwatermarked: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "achieved_fpr": self.achieved_fpr,
            "achieved_tpr": self.achieved_tpr,
            "method": self.method,
            "n_watermarked": self.n_watermarked,
            "n_unwatermarked": self.n_unwatermarked,
        }


def calibrate_theoretical(target_fpr: float = 0.01) -> float:
    """
    Compute theoretical threshold for target FPR.
    
    Under H0, S ~ N(0, 1), so:
        threshold = z_{1 - FPR} = Φ^{-1}(1 - FPR)
    
    This is the recommended approach when you trust the
    theoretical null distribution.
    
    Args:
        target_fpr: Target false positive rate (default 1%)
        
    Returns:
        Threshold value
        
    Example:
        >>> threshold = calibrate_theoretical(0.01)  # 1% FPR
        >>> print(f"Threshold: {threshold:.3f}")  # ~2.326
    """
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    
    return float(stats.norm.ppf(1 - target_fpr))


def calibrate_empirical(
    s_stats_unwatermarked: np.ndarray,
    target_fpr: float = 0.01,
) -> CalibrationResult:
    """
    Calibrate threshold from empirical unwatermarked data.
    
    Sets threshold at the (1 - FPR) quantile of the unwatermarked
    S-statistic distribution.
    
    Args:
        s_stats_unwatermarked: S-statistics from unwatermarked images
        target_fpr: Target false positive rate
        
    Returns:
        CalibrationResult with threshold and metrics
    """
    s_sorted = np.sort(s_stats_unwatermarked)
    n = len(s_sorted)
    
    # Find threshold at (1 - target_fpr) quantile
    idx = int(np.ceil((1 - target_fpr) * n)) - 1
    idx = max(0, min(idx, n - 1))
    threshold = float(s_sorted[idx])
    
    # Compute actual FPR
    achieved_fpr = float(np.mean(s_stats_unwatermarked > threshold))
    
    return CalibrationResult(
        threshold=threshold,
        target_fpr=target_fpr,
        achieved_fpr=achieved_fpr,
        achieved_tpr=0.0,  # Unknown without watermarked data
        method="empirical",
        n_watermarked=0,
        n_unwatermarked=n,
    )


def calibrate_from_labeled_data(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    target_fpr: float = 0.01,
) -> CalibrationResult:
    """
    Calibrate threshold from labeled watermarked/unwatermarked data.
    
    Sets threshold to achieve target FPR on unwatermarked data
    and reports achieved TPR on watermarked data.
    
    Args:
        s_stats_watermarked: S-statistics from watermarked images
        s_stats_unwatermarked: S-statistics from unwatermarked images
        target_fpr: Target false positive rate
        
    Returns:
        CalibrationResult with threshold and metrics
    """
    # Calibrate on unwatermarked
    s_sorted = np.sort(s_stats_unwatermarked)
    n_unwm = len(s_sorted)
    n_wm = len(s_stats_watermarked)
    
    # Find threshold
    idx = int(np.ceil((1 - target_fpr) * n_unwm)) - 1
    idx = max(0, min(idx, n_unwm - 1))
    threshold = float(s_sorted[idx])
    
    # Compute achieved metrics
    achieved_fpr = float(np.mean(s_stats_unwatermarked > threshold))
    achieved_tpr = float(np.mean(s_stats_watermarked > threshold))
    
    return CalibrationResult(
        threshold=threshold,
        target_fpr=target_fpr,
        achieved_fpr=achieved_fpr,
        achieved_tpr=achieved_tpr,
        method="labeled",
        n_watermarked=n_wm,
        n_unwatermarked=n_unwm,
    )


def compute_roc_curve(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    n_thresholds: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve for detection.
    
    Args:
        s_stats_watermarked: S-statistics from watermarked images
        s_stats_unwatermarked: S-statistics from unwatermarked images
        n_thresholds: Number of threshold points
        
    Returns:
        Dictionary with 'fpr', 'tpr', 'thresholds', 'auc'
    """
    # Get threshold range
    all_stats = np.concatenate([s_stats_watermarked, s_stats_unwatermarked])
    thresholds = np.linspace(all_stats.min() - 1, all_stats.max() + 1, n_thresholds)
    
    # Compute TPR and FPR at each threshold
    tpr_values = []
    fpr_values = []
    
    for thresh in thresholds:
        tpr = np.mean(s_stats_watermarked > thresh)
        fpr = np.mean(s_stats_unwatermarked > thresh)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    tpr_arr = np.array(tpr_values)
    fpr_arr = np.array(fpr_values)
    
    # Compute AUC using trapezoidal rule
    # Sort by FPR (decreasing) for proper integration
    sort_idx = np.argsort(fpr_arr)[::-1]
    fpr_sorted = fpr_arr[sort_idx]
    tpr_sorted = tpr_arr[sort_idx]
    
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
    
    return {
        "fpr": fpr_arr,
        "tpr": tpr_arr,
        "thresholds": thresholds,
        "auc": auc,
    }


def find_threshold_at_fpr(
    fpr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float = 0.01,
) -> Tuple[float, float, int]:
    """
    Select threshold closest to a target FPR from ROC outputs.

    This is evaluation/calibration-only logic and does not change score computation.

    Args:
        fpr: ROC false positive rates (same length as thresholds)
        thresholds: ROC thresholds (same length as fpr)
        target_fpr: Desired false positive rate (default 1%)

    Returns:
        (selected_threshold, achieved_fpr, index)

    Notes:
        - This selects the threshold whose FPR is closest to target_fpr.
        - If ROC never reaches target_fpr exactly (discrete scores), this yields the
          closest achievable operating point.
    """
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")

    fpr_arr = np.asarray(fpr, dtype=float)
    thr_arr = np.asarray(thresholds, dtype=float)

    if fpr_arr.ndim != 1 or thr_arr.ndim != 1:
        raise ValueError("fpr and thresholds must be 1D arrays")
    if len(fpr_arr) != len(thr_arr):
        raise ValueError(
            f"fpr and thresholds must have same length, got {len(fpr_arr)} and {len(thr_arr)}"
        )
    if len(fpr_arr) == 0:
        raise ValueError("Empty ROC arrays: cannot select threshold")
    if not np.all(np.isfinite(fpr_arr)) or not np.all(np.isfinite(thr_arr)):
        raise ValueError("Non-finite values found in fpr/thresholds")
    if np.any(fpr_arr < 0.0) or np.any(fpr_arr > 1.0):
        raise ValueError("ROC FPR values must be within [0, 1]")

    idx = int(np.argmin(np.abs(fpr_arr - float(target_fpr))))
    selected_threshold = float(thr_arr[idx])
    achieved_fpr = float(fpr_arr[idx])

    # Sanity checks
    assert 0.0 <= achieved_fpr <= 1.0, "Achieved FPR must be within [0, 1]"

    return selected_threshold, achieved_fpr, idx


def find_threshold_at_fpr_from_labeled_data(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    target_fpr: float,
) -> Tuple[float, float]:
    """
    Find threshold and corresponding TPR at target FPR using labeled-data calibration.
    
    Args:
        s_stats_watermarked: Watermarked S-statistics
        s_stats_unwatermarked: Unwatermarked S-statistics
        target_fpr: Target FPR
        
    Returns:
        (threshold, achieved_tpr) tuple
    """
    result = calibrate_from_labeled_data(
        s_stats_watermarked,
        s_stats_unwatermarked,
        target_fpr,
    )
    return result.threshold, result.achieved_tpr


def find_threshold_at_tpr(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    target_tpr: float,
) -> Tuple[float, float]:
    """
    Find threshold and corresponding FPR at target TPR.
    
    Args:
        s_stats_watermarked: Watermarked S-statistics
        s_stats_unwatermarked: Unwatermarked S-statistics
        target_tpr: Target TPR
        
    Returns:
        (threshold, achieved_fpr) tuple
    """
    # Find threshold from watermarked data
    s_sorted = np.sort(s_stats_watermarked)[::-1]  # Descending
    n = len(s_sorted)
    
    idx = int(np.ceil(target_tpr * n)) - 1
    idx = max(0, min(idx, n - 1))
    threshold = float(s_sorted[idx])
    
    # Compute achieved FPR
    achieved_fpr = float(np.mean(s_stats_unwatermarked > threshold))
    
    return threshold, achieved_fpr


def compute_detection_metrics_at_threshold(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute detection metrics at a specific threshold.
    
    Args:
        s_stats_watermarked: Watermarked S-statistics
        s_stats_unwatermarked: Unwatermarked S-statistics
        threshold: Detection threshold
        
    Returns:
        Dictionary of metrics
    """
    # Predictions
    pred_wm = s_stats_watermarked > threshold
    pred_unwm = s_stats_unwatermarked > threshold
    
    # Counts
    tp = np.sum(pred_wm)
    fn = np.sum(~pred_wm)
    fp = np.sum(pred_unwm)
    tn = np.sum(~pred_unwm)
    
    # Rates
    n_wm = len(s_stats_watermarked)
    n_unwm = len(s_stats_unwatermarked)
    
    tpr = tp / n_wm if n_wm > 0 else 0.0
    fpr = fp / n_unwm if n_unwm > 0 else 0.0
    fnr = fn / n_wm if n_wm > 0 else 0.0
    tnr = tn / n_unwm if n_unwm > 0 else 0.0
    
    # Precision and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / (n_wm + n_unwm)
    
    return {
        "threshold": threshold,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================================
# Calibration Summary
# ============================================================================


def summarize_calibration(
    s_stats_watermarked: np.ndarray,
    s_stats_unwatermarked: np.ndarray,
    target_fprs: List[float] = [0.001, 0.01, 0.05],
) -> Dict:
    """
    Generate comprehensive calibration summary.
    
    Args:
        s_stats_watermarked: Watermarked S-statistics
        s_stats_unwatermarked: Unwatermarked S-statistics
        target_fprs: List of target FPR values
        
    Returns:
        Comprehensive summary dictionary
    """
    summary = {
        "sample_sizes": {
            "watermarked": len(s_stats_watermarked),
            "unwatermarked": len(s_stats_unwatermarked),
        },
        "s_statistic_distributions": {
            "watermarked": {
                "mean": float(np.mean(s_stats_watermarked)),
                "std": float(np.std(s_stats_watermarked)),
                "min": float(np.min(s_stats_watermarked)),
                "max": float(np.max(s_stats_watermarked)),
                "median": float(np.median(s_stats_watermarked)),
            },
            "unwatermarked": {
                "mean": float(np.mean(s_stats_unwatermarked)),
                "std": float(np.std(s_stats_unwatermarked)),
                "min": float(np.min(s_stats_unwatermarked)),
                "max": float(np.max(s_stats_unwatermarked)),
                "median": float(np.median(s_stats_unwatermarked)),
            },
        },
        "thresholds_by_fpr": {},
    }
    
    # Compute ROC
    roc = compute_roc_curve(s_stats_watermarked, s_stats_unwatermarked)
    summary["auc"] = roc["auc"]
    
    # Compute thresholds at target FPRs
    for fpr in target_fprs:
        result = calibrate_from_labeled_data(
            s_stats_watermarked,
            s_stats_unwatermarked,
            fpr,
        )
        summary["thresholds_by_fpr"][f"fpr_{fpr}"] = result.to_dict()
    
    # Add theoretical thresholds for comparison
    summary["theoretical_thresholds"] = {
        f"fpr_{fpr}": calibrate_theoretical(fpr)
        for fpr in target_fprs
    }
    
    return summary

