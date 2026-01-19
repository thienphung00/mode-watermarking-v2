"""
Quality metrics for watermark evaluation.

OPTIONAL module that does NOT affect watermark detection S-statistic.
These metrics measure image quality degradation from watermarking,
not watermark presence.

Metrics included:
    - PSNR: Peak Signal-to-Noise Ratio
    - SSIM: Structural Similarity Index
    - LPIPS: Learned Perceptual Image Patch Similarity (requires torch)
    - FID: Fréchet Inception Distance (requires torch + inception)
    - CLIP: CLIP-based similarity (requires transformers)

These are quality metrics for evaluating watermark invisibility,
separate from the detection pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

# Quality metrics from scikit-image
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class QualityMetrics:
    """
    Computes image quality metrics for watermark evaluation.
    
    These metrics measure the visual quality impact of watermarking.
    They are OPTIONAL and do not affect watermark detection.
    
    Example:
        >>> metrics = QualityMetrics()
        >>> psnr = metrics.compute_psnr(watermarked, original)
        >>> ssim = metrics.compute_ssim(watermarked, original)
    """
    
    @staticmethod
    def compute_psnr(
        image1: np.ndarray,
        image2: np.ndarray,
        data_range: Optional[float] = None,
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Higher PSNR indicates less distortion. Typical good values:
        - >40 dB: Excellent (nearly invisible changes)
        - 30-40 dB: Good
        - <30 dB: Visible distortion
        
        Args:
            image1: First image (watermarked)
            image2: Second image (original)
            data_range: Data range (auto-detected if None)
            
        Returns:
            PSNR value in dB
        """
        if not HAS_SKIMAGE:
            return _compute_psnr_numpy(image1, image2, data_range)
        
        if data_range is None:
            data_range = _infer_data_range(image1, image2)
        
        return float(peak_signal_noise_ratio(image1, image2, data_range=data_range))
    
    @staticmethod
    def compute_ssim(
        image1: np.ndarray,
        image2: np.ndarray,
        data_range: Optional[float] = None,
    ) -> float:
        """
        Compute Structural Similarity Index.
        
        SSIM measures perceptual similarity. Range [0, 1]:
        - >0.95: Nearly identical
        - 0.90-0.95: Very similar
        - <0.90: Noticeable differences
        
        Args:
            image1: First image (watermarked)
            image2: Second image (original)
            data_range: Data range
            
        Returns:
            SSIM value (0-1)
        """
        if not HAS_SKIMAGE:
            raise ImportError("SSIM requires scikit-image: pip install scikit-image")
        
        if data_range is None:
            data_range = _infer_data_range(image1, image2)
        
        if image1.ndim == 3:
            # Multi-channel
            return float(
                structural_similarity(
                    image1, image2, data_range=data_range, channel_axis=-1
                )
            )
        else:
            # Single channel
            return float(structural_similarity(image1, image2, data_range=data_range))
    
    def compute_all(
        self,
        image_watermarked: np.ndarray,
        image_original: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all available quality metrics.
        
        Args:
            image_watermarked: Watermarked image
            image_original: Original image
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        # PSNR
        try:
            results["psnr"] = self.compute_psnr(image_watermarked, image_original)
        except Exception as e:
            results["psnr_error"] = str(e)
        
        # SSIM
        try:
            results["ssim"] = self.compute_ssim(image_watermarked, image_original)
        except Exception as e:
            results["ssim_error"] = str(e)
        
        return results
    
    def batch_compute(
        self,
        images_watermarked: List[np.ndarray],
        images_original: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute quality metrics for batch of image pairs.
        
        Args:
            images_watermarked: List of watermarked images
            images_original: List of original images
            
        Returns:
            Dictionary of average metrics
        """
        if len(images_watermarked) != len(images_original):
            raise ValueError("Image lists must have same length")
        
        psnr_values = []
        ssim_values = []
        
        for img_wm, img_orig in zip(images_watermarked, images_original):
            psnr = self.compute_psnr(img_wm, img_orig)
            ssim = self.compute_ssim(img_wm, img_orig)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        return {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "n_samples": len(psnr_values),
        }


# ============================================================================
# Detection Metrics (Binary Classification)
# ============================================================================


def compute_detection_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute detection metrics for binary classification.
    
    Args:
        predictions: Binary predictions (0 or 1)
        labels: Ground truth labels (0 or 1)
        
    Returns:
        Dictionary of metrics
    """
    predictions = np.asarray(predictions).astype(int)
    labels = np.asarray(labels).astype(int)
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    # Metrics
    total = len(labels)
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)  # TPR
    specificity = tn / max(tn + fp, 1)  # TNR
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tpr": float(recall),
        "fpr": float(fp / max(tn + fp, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns:
        2x2 array: [[TN, FP], [FN, TP]]
    """
    predictions = np.asarray(predictions).astype(int)
    labels = np.asarray(labels).astype(int)
    
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    return np.array([[tn, fp], [fn, tp]])


# ============================================================================
# Helper Functions
# ============================================================================


def _infer_data_range(image1: np.ndarray, image2: np.ndarray) -> float:
    """Infer data range from images."""
    max_val = max(image1.max(), image2.max())
    
    if max_val <= 1.0:
        return 1.0
    elif max_val <= 255:
        return 255.0
    else:
        return float(max_val)


def _compute_psnr_numpy(
    image1: np.ndarray,
    image2: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """Compute PSNR using pure numpy (fallback)."""
    if data_range is None:
        data_range = _infer_data_range(image1, image2)
    
    mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def load_image_as_array(
    path: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load image and convert to numpy array.
    
    Args:
        path: Path to image
        normalize: If True, normalize to [0, 1]
        
    Returns:
        Image as numpy array
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    
    if normalize:
        arr = arr / 255.0
    
    return arr
