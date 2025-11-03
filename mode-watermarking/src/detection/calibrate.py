"""
Statistical calibration and threshold estimation for watermark detection.

Calibrates detection thresholds based on S-statistic distributions,
computes TPR@FPR, AUC-ROC, AUC-PR, and statistical significance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc as compute_auc,
    roc_auc_score
)


def calibrate_thresholds(
    s_scores: np.ndarray,  # [num_samples] detection scores
    labels: np.ndarray,    # [num_samples] binary labels (0=unwatermarked, 1=watermarked)
    target_fpr: float = 0.01,  # Target FPR (e.g., 1%)
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Calibrate detection thresholds and compute metrics.
    
    Args:
        s_scores: Detection scores for all samples
        labels: Ground truth labels (0 or 1)
        target_fpr: Target false positive rate (default: 0.01 = 1%)
        significance_level: Statistical significance level (default: 0.05)
    
    Returns:
        Dictionary with:
        - threshold: Optimal threshold at target_fpr
        - tpr_at_target_fpr: TPR@1%FPR
        - fpr_values: FPR at various thresholds
        - tpr_values: TPR at various thresholds
        - auc_roc: Area Under ROC Curve
        - auc_pr: Area Under PR Curve
        - precision_at_target_fpr: Precision at target FPR threshold
        - precision_values: Precision at various thresholds
        - recall_values: Recall at various thresholds
        - roc_curve: Tuple of (fpr, tpr) arrays
        - pr_curve: Tuple of (precision, recall) arrays
        - statistical_significance: Dict with p-values, confidence intervals
    """
    if len(s_scores) != len(labels):
        raise ValueError(f"Mismatch: {len(s_scores)} scores vs {len(labels)} labels")
    
    # Separate scores by label
    watermarked_scores = s_scores[labels == 1]
    unwatermarked_scores = s_scores[labels == 0]
    
    if len(watermarked_scores) == 0 or len(unwatermarked_scores) == 0:
        raise ValueError("Both watermarked and unwatermarked samples required")
    
    # Compute ROC curve
    fpr_values, tpr_values, roc_thresholds = roc_curve(labels, s_scores)
    
    # Compute PR curve
    precision_values, recall_values, pr_thresholds = precision_recall_curve(labels, s_scores)
    
    # Compute AUC
    auc_roc = roc_auc_score(labels, s_scores)
    auc_pr = compute_auc(recall_values, precision_values)
    
    # Find threshold at target FPR
    # Find index where FPR is closest to target_fpr
    fpr_diff = np.abs(fpr_values - target_fpr)
    target_idx = np.argmin(fpr_diff)
    
    threshold = float(roc_thresholds[target_idx]) if target_idx < len(roc_thresholds) else float(roc_thresholds[-1])
    tpr_at_target_fpr = float(tpr_values[target_idx])
    
    # Find precision at target FPR threshold
    # Map threshold to PR curve
    if threshold in pr_thresholds:
        pr_idx = np.where(pr_thresholds == threshold)[0]
        if len(pr_idx) > 0:
            precision_at_target_fpr = float(precision_values[pr_idx[0]])
        else:
            # Interpolate
            precision_at_target_fpr = np.interp(threshold, pr_thresholds[::-1], precision_values[::-1])
    else:
        # Interpolate precision at target threshold
        precision_at_target_fpr = np.interp(threshold, pr_thresholds[::-1], precision_values[::-1])
    
    # Statistical significance tests
    # Mann-Whitney U test (non-parametric test for score distributions)
    statistic, p_value = stats.mannwhitneyu(
        watermarked_scores,
        unwatermarked_scores,
        alternative='greater'  # Test if watermarked scores are greater
    )
    
    # Confidence intervals for TPR@FPR using bootstrap
    # Simplified: use percentile method
    n_bootstrap = 1000
    bootstrap_tprs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(len(s_scores), size=len(s_scores), replace=True)
        boot_scores = s_scores[boot_indices]
        boot_labels = labels[boot_indices]
        
        try:
            boot_fpr, boot_tpr, boot_thresh = roc_curve(boot_labels, boot_scores)
            boot_fpr_diff = np.abs(boot_fpr - target_fpr)
            boot_target_idx = np.argmin(boot_fpr_diff)
            if boot_target_idx < len(boot_tpr):
                bootstrap_tprs.append(boot_tpr[boot_target_idx])
        except:
            continue
    
    if len(bootstrap_tprs) > 0:
        ci_lower = float(np.percentile(bootstrap_tprs, 2.5))
        ci_upper = float(np.percentile(bootstrap_tprs, 97.5))
    else:
        ci_lower = ci_upper = tpr_at_target_fpr
    
    statistical_significance = {
        "mann_whitney_u_statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < significance_level,
        "tpr_ci_lower": ci_lower,
        "tpr_ci_upper": ci_upper,
        "confidence_level": 0.95
    }
    
    return {
        "threshold": threshold,
        "tpr_at_target_fpr": tpr_at_target_fpr,
        "fpr_values": fpr_values.tolist(),
        "tpr_values": tpr_values.tolist(),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "precision_at_target_fpr": float(precision_at_target_fpr),
        "precision_values": precision_values.tolist(),
        "recall_values": recall_values.tolist(),
        "roc_curve": (fpr_values.tolist(), tpr_values.tolist()),
        "pr_curve": (precision_values.tolist(), recall_values.tolist()),
        "roc_thresholds": roc_thresholds.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
        "statistical_significance": statistical_significance,
        "target_fpr": target_fpr
    }


def compute_detection_metrics(
    s_scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,  # If None, calibrate automatically
    target_fpr: float = 0.01
) -> Dict[str, float]:
    """
    Compute detection metrics at operating threshold.
    
    Args:
        s_scores: Detection scores
        labels: Ground truth labels
        threshold: Operating threshold (if None, calibrate from target_fpr)
        target_fpr: Target FPR for threshold calibration
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix components
    """
    if threshold is None:
        # Calibrate threshold
        calibration = calibrate_thresholds(s_scores, labels, target_fpr=target_fpr)
        threshold = calibration["threshold"]
    
    # Apply threshold
    predictions = (s_scores >= threshold).astype(int)
    
    # Compute confusion matrix
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Compute metrics
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold": float(threshold)
    }


def tpr_at_fpr(
    s_scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.01
) -> float:
    """
    Compute TPR at a specific FPR threshold.
    
    Args:
        s_scores: Detection scores
        labels: Ground truth labels
        target_fpr: Target false positive rate
    
    Returns:
        True positive rate at target FPR
    """
    calibration = calibrate_thresholds(s_scores, labels, target_fpr=target_fpr)
    return calibration["tpr_at_target_fpr"]
