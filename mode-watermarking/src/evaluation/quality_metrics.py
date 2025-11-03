"""
Image quality metrics for watermark evaluation.

Computes PSNR, SSIM, LPIPS, FID, and CLIP similarity between
watermarked and original images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Optional imports
try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    lpips = None

try:
    import torch
    import torchvision.transforms as transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    transforms = None

try:
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    CLIPModel = None
    CLIPProcessor = None


def compute_psnr(
    watermarked_image: Union[Image.Image, np.ndarray],
    original_image: Union[Image.Image, np.ndarray]
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        watermarked_image: Watermarked image
        original_image: Original image
    
    Returns:
        PSNR value in dB
    """
    # Convert to numpy arrays
    if isinstance(watermarked_image, Image.Image):
        wm_array = np.array(watermarked_image)
    else:
        wm_array = watermarked_image
    
    if isinstance(original_image, Image.Image):
        orig_array = np.array(original_image)
    else:
        orig_array = original_image
    
    # Ensure same shape
    if wm_array.shape != orig_array.shape:
        raise ValueError(f"Image shape mismatch: {wm_array.shape} vs {orig_array.shape}")
    
    # Normalize to [0, 1] if in [0, 255]
    if wm_array.max() > 1.0:
        wm_array = wm_array.astype(np.float64) / 255.0
    if orig_array.max() > 1.0:
        orig_array = orig_array.astype(np.float64) / 255.0
    
    # Compute PSNR
    psnr_value = peak_signal_noise_ratio(orig_array, wm_array, data_range=1.0)
    return float(psnr_value)


def compute_ssim(
    watermarked_image: Union[Image.Image, np.ndarray],
    original_image: Union[Image.Image, np.ndarray]
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        watermarked_image: Watermarked image
        original_image: Original image
    
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    # Convert to numpy arrays
    if isinstance(watermarked_image, Image.Image):
        wm_array = np.array(watermarked_image)
    else:
        wm_array = watermarked_image
    
    if isinstance(original_image, Image.Image):
        orig_array = np.array(original_image)
    else:
        orig_array = original_image
    
    # Ensure same shape
    if wm_array.shape != orig_array.shape:
        raise ValueError(f"Image shape mismatch: {wm_array.shape} vs {orig_array.shape}")
    
    # Normalize to [0, 1] if needed
    if wm_array.max() > 1.0:
        wm_array = wm_array.astype(np.float64) / 255.0
    if orig_array.max() > 1.0:
        orig_array = orig_array.astype(np.float64) / 255.0
    
    # Compute SSIM
    if wm_array.ndim == 3:
        # Color image
        ssim_value = structural_similarity(
            orig_array, wm_array, channel_axis=2, data_range=1.0
        )
    else:
        # Grayscale
        ssim_value = structural_similarity(
            orig_array, wm_array, data_range=1.0
        )
    
    return float(ssim_value)


def compute_lpips(
    watermarked_image: Union[Image.Image, np.ndarray],
    original_image: Union[Image.Image, np.ndarray],
    network: str = "alex",
    device: str = "cuda"
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        watermarked_image: Watermarked image
        original_image: Original image
        network: LPIPS network ("alex", "vgg", or "squeeze")
        device: Device for computation
    
    Returns:
        LPIPS value (lower is better, typically [0, 1])
    """
    if not _LPIPS_AVAILABLE:
        raise ImportError("lpips library not available. Install with: pip install lpips")
    
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required for LPIPS computation")
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=network).to(device)
    
    # Convert to PIL if needed
    if isinstance(watermarked_image, np.ndarray):
        wm_image = Image.fromarray(watermarked_image)
    else:
        wm_image = watermarked_image
    
    if isinstance(original_image, np.ndarray):
        orig_image = Image.fromarray(original_image)
    else:
        orig_image = original_image
    
    # Convert to tensor and normalize to [-1, 1]
    to_tensor = transforms.ToTensor()
    wm_tensor = to_tensor(wm_image).unsqueeze(0).to(device) * 2.0 - 1.0
    orig_tensor = to_tensor(orig_image).unsqueeze(0).to(device) * 2.0 - 1.0
    
    # Compute LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(orig_tensor, wm_tensor).item()
    
    return float(lpips_value)


def compute_fid(
    watermarked_images: List[Union[Image.Image, np.ndarray]],
    original_images: List[Union[Image.Image, np.ndarray]],
    device: str = "cuda"
) -> float:
    """
    Compute Fréchet Inception Distance (FID).
    
    Note: FID requires batch computation and uses Inception v3 features.
    
    Args:
        watermarked_images: List of watermarked images
        original_images: List of original images
        device: Device for computation
    
    Returns:
        FID value (lower is better)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required for FID computation")
    
    try:
        from pytorch_fid import fid_score
    except ImportError:
        raise ImportError(
            "pytorch-fid not available. Install with: pip install pytorch-fid"
        )
    
    # Save images to temporary directories
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wm_dir = Path(tmpdir) / "watermarked"
        orig_dir = Path(tmpdir) / "original"
        wm_dir.mkdir()
        orig_dir.mkdir()
        
        # Save images
        for i, (wm_img, orig_img) in enumerate(zip(watermarked_images, original_images)):
            if isinstance(wm_img, np.ndarray):
                wm_img = Image.fromarray(wm_img)
            if isinstance(orig_img, np.ndarray):
                orig_img = Image.fromarray(orig_img)
            
            wm_img.save(wm_dir / f"{i}.png")
            orig_img.save(orig_dir / f"{i}.png")
        
        # Compute FID
        fid_value = fid_score.calculate_fid_given_paths(
            [str(wm_dir), str(orig_dir)],
            batch_size=50,
            device=device,
            dims=2048
        )
    
    return float(fid_value)


def compute_clip_similarity(
    watermarked_image: Union[Image.Image, np.ndarray],
    original_image: Union[Image.Image, np.ndarray],
    device: str = "cuda"
) -> float:
    """
    Compute CLIP similarity score.
    
    Args:
        watermarked_image: Watermarked image
        original_image: Original image
        device: Device for computation
    
    Returns:
        CLIP similarity score in [0, 1] (higher is better)
    """
    if not _CLIP_AVAILABLE:
        raise ImportError(
            "transformers library not available. Install with: pip install transformers"
        )
    
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required for CLIP computation")
    
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Convert to PIL if needed
    if isinstance(watermarked_image, np.ndarray):
        wm_image = Image.fromarray(watermarked_image)
    else:
        wm_image = watermarked_image
    
    if isinstance(original_image, np.ndarray):
        orig_image = Image.fromarray(original_image)
    else:
        orig_image = original_image
    
    # Process images
    inputs = processor(images=[orig_image, wm_image], return_tensors="pt", padding=True).to(device)
    
    # Get image features
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        image_features[0:1], image_features[1:2]
    ).item()
    
    return float(similarity)


def compute_quality_metrics(
    watermarked_image: Union[Image.Image, np.ndarray, torch.Tensor],
    original_image: Union[Image.Image, np.ndarray, torch.Tensor],
    metrics: List[str] = ["psnr", "ssim", "lpips", "fid", "clip_similarity"],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute image quality metrics between watermarked and original.
    
    Args:
        watermarked_image: Watermarked image
        original_image: Original image
        metrics: List of metrics to compute
        device: Device for GPU-accelerated metrics
    
    Returns:
        Dictionary with metric values (NaN for unavailable metrics)
    """
    results = {}
    
    # PSNR
    if "psnr" in metrics:
        try:
            results["psnr"] = compute_psnr(watermarked_image, original_image)
        except Exception as e:
            print(f"Error computing PSNR: {e}")
            results["psnr"] = float("nan")
    
    # SSIM
    if "ssim" in metrics:
        try:
            results["ssim"] = compute_ssim(watermarked_image, original_image)
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            results["ssim"] = float("nan")
    
    # LPIPS
    if "lpips" in metrics:
        try:
            results["lpips"] = compute_lpips(watermarked_image, original_image, device=device)
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            results["lpips"] = float("nan")
    
    # FID (requires batch, skip for single image)
    if "fid" in metrics:
        results["fid"] = float("nan")  # FID requires batch computation
    
    # CLIP similarity
    if "clip_similarity" in metrics:
        try:
            results["clip_similarity"] = compute_clip_similarity(
                watermarked_image, original_image, device=device
            )
        except Exception as e:
            print(f"Error computing CLIP similarity: {e}")
            results["clip_similarity"] = float("nan")
    
    return results


def batch_compute_quality_metrics(
    watermarked_images: List[Union[Image.Image, np.ndarray]],
    original_images: List[Union[Image.Image, np.ndarray]],
    metrics: List[str] = ["psnr", "ssim", "lpips", "fid", "clip_similarity"],
    batch_size: int = 16,
    device: str = "cuda"
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Batch compute quality metrics.
    
    Args:
        watermarked_images: List of watermarked images
        original_images: List of original images
        metrics: List of metrics to compute
        batch_size: Batch size for GPU-accelerated metrics
        device: Device for computation
    
    Returns:
        Dictionary with:
        - "psnr": np.ndarray [num_images]
        - "ssim": np.ndarray [num_images]
        - "lpips": np.ndarray [num_images]
        - "fid": float (single value for entire batch)
        - "clip_similarity": np.ndarray [num_images]
    """
    if len(watermarked_images) != len(original_images):
        raise ValueError(
            f"Mismatch: {len(watermarked_images)} watermarked vs {len(original_images)} original"
        )
    
    results = {}
    
    # PSNR (per-image)
    if "psnr" in metrics:
        psnr_values = []
        for wm_img, orig_img in zip(watermarked_images, original_images):
            try:
                psnr_val = compute_psnr(wm_img, orig_img)
                psnr_values.append(psnr_val)
            except Exception as e:
                print(f"Error computing PSNR: {e}")
                psnr_values.append(float("nan"))
        results["psnr"] = np.array(psnr_values)
    
    # SSIM (per-image)
    if "ssim" in metrics:
        ssim_values = []
        for wm_img, orig_img in zip(watermarked_images, original_images):
            try:
                ssim_val = compute_ssim(wm_img, orig_img)
                ssim_values.append(ssim_val)
            except Exception as e:
                print(f"Error computing SSIM: {e}")
                ssim_values.append(float("nan"))
        results["ssim"] = np.array(ssim_values)
    
    # LPIPS (per-image, can be batched)
    if "lpips" in metrics:
        lpips_values = []
        if _LPIPS_AVAILABLE and _TORCH_AVAILABLE:
            # Batch processing for LPIPS
            loss_fn = lpips.LPIPS(net="alex").to(device)
            to_tensor = transforms.ToTensor()
            
            for i in range(0, len(watermarked_images), batch_size):
                batch_wm = watermarked_images[i:i+batch_size]
                batch_orig = original_images[i:i+batch_size]
                
                # Convert batch to tensors
                wm_tensors = []
                orig_tensors = []
                for wm_img, orig_img in zip(batch_wm, batch_orig):
                    if isinstance(wm_img, np.ndarray):
                        wm_img = Image.fromarray(wm_img)
                    if isinstance(orig_img, np.ndarray):
                        orig_img = Image.fromarray(orig_img)
                    
                    wm_tensor = to_tensor(wm_img).unsqueeze(0).to(device) * 2.0 - 1.0
                    orig_tensor = to_tensor(orig_img).unsqueeze(0).to(device) * 2.0 - 1.0
                    wm_tensors.append(wm_tensor)
                    orig_tensors.append(orig_tensor)
                
                wm_batch = torch.cat(wm_tensors, dim=0)
                orig_batch = torch.cat(orig_tensors, dim=0)
                
                # Compute LPIPS
                with torch.no_grad():
                    lpips_batch = loss_fn(orig_batch, wm_batch).cpu().numpy()
                    lpips_values.extend(lpips_batch.tolist())
        else:
            # Fallback to per-image
            for wm_img, orig_img in zip(watermarked_images, original_images):
                try:
                    lpips_val = compute_lpips(wm_img, orig_img, device=device)
                    lpips_values.append(lpips_val)
                except Exception as e:
                    print(f"Error computing LPIPS: {e}")
                    lpips_values.append(float("nan"))
        results["lpips"] = np.array(lpips_values)
    
    # FID (single value for entire batch)
    if "fid" in metrics:
        try:
            fid_value = compute_fid(watermarked_images, original_images, device=device)
            results["fid"] = fid_value
        except Exception as e:
            print(f"Error computing FID: {e}")
            results["fid"] = float("nan")
    
    # CLIP similarity (per-image)
    if "clip_similarity" in metrics:
        clip_values = []
        for wm_img, orig_img in zip(watermarked_images, original_images):
            try:
                clip_val = compute_clip_similarity(wm_img, orig_img, device=device)
                clip_values.append(clip_val)
            except Exception as e:
                print(f"Error computing CLIP similarity: {e}")
                clip_values.append(float("nan"))
        results["clip_similarity"] = np.array(clip_values)
    
    return results
