"""
Detection utilities for image and latent conversion.

This module provides detection-specific utilities for:
- Image ↔ Latent conversion (VAE encoding/decoding)
- DDIM inversion helpers
- Fast latent extraction
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image


__all__ = [
    "encode_image_to_latent",
    "decode_latent_to_image",
    "invert_latent_ddim",
    "extract_latent",
]


def encode_image_to_latent(
    image: Union[Image.Image, str, torch.Tensor, np.ndarray],
    vae: Any,
    device: str = "cuda",
    vae_scaling_factor: float = 0.18215,
    allow_resize: bool = True,
) -> torch.Tensor:
    """
    Encode image to latent space z_0 using VAE (fast path, no inversion).
    
    This utility converts images to latent representations for detection.
    
    Args:
        image: Input image (PIL Image, path, tensor, or numpy array)
        vae: VAE encoder module from Stable Diffusion
        device: Device for computation
        vae_scaling_factor: VAE scaling factor (default: 0.18215 for SD 1.x)
        allow_resize: If True, resize images to 512x512. If False, require exact 512x512.
            Set to False for hybrid/full_inversion detection modes.
    
    Returns:
        Latent tensor z_0 [1, C, H, W] on specified device
    
    Example:
        >>> latent = encode_image_to_latent(image, vae, device="cuda")
        >>> print(latent.shape)  # [1, 4, 64, 64]
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        tensor = _pil_to_tensor(image, allow_resize=allow_resize)
    elif isinstance(image, np.ndarray):
        tensor = _numpy_to_tensor(image)
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Ensure correct shape and device
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device=device, dtype=vae.dtype)
    
    # VAE encode
    with torch.no_grad():
        latent_dist = vae.encode(tensor).latent_dist
        z_0 = latent_dist.sample()
        z_0 = z_0 * vae_scaling_factor
    
    return z_0


def decode_latent_to_image(
    latent: torch.Tensor,
    vae: Any,
    device: str = "cuda",
    vae_scaling_factor: float = 0.18215,
) -> Image.Image:
    """
    Decode latent tensor to PIL Image using VAE decoder.
    
    Args:
        latent: Latent tensor [1, C, H, W] or [C, H, W]
        vae: VAE decoder module from Stable Diffusion
        device: Device for computation
        vae_scaling_factor: VAE scaling factor (default: 0.18215)
    
    Returns:
        PIL Image in RGB mode
    
    Example:
        >>> image = decode_latent_to_image(latent, vae, device="cuda")
    """
    # Ensure correct shape and device
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    latent = latent.to(device=device, dtype=vae.dtype)
    
    # Scale latent
    latent = latent / vae_scaling_factor
    
    # VAE decode
    with torch.no_grad():
        decoded = vae.decode(latent).sample
    
    # Convert to PIL Image
    decoded = decoded.squeeze(0).cpu().permute(1, 2, 0)
    decoded = (decoded + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    decoded = torch.clamp(decoded, 0, 1)
    decoded_np = (decoded.numpy() * 255).astype(np.uint8)
    
    return Image.fromarray(decoded_np, mode="RGB")


def invert_latent_ddim(
    image: Union[Image.Image, str, torch.Tensor],
    pipeline: Any,
    num_inference_steps: int = 50,
    prompt: str = "",
    guidance_scale: float = 1.0,
    return_intermediates: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
    """
    Invert image to recover initial latent z_T using DDIM inversion.
    
    This utility provides a simpler interface to DDIMInverter for detection.
    
    Args:
        image: Input image (PIL Image, path, or tensor)
        pipeline: Stable Diffusion pipeline with VAE, UNet, and scheduler
        num_inference_steps: Number of inversion steps
        prompt: Text prompt for guided inversion (optional)
        guidance_scale: Guidance scale (1.0 = unconditional)
        return_intermediates: If True, also return intermediate latents
    
    Returns:
        z_T: Recovered initial latent tensor [1, 4, H/8, W/8]
        intermediates: (optional) List of intermediate latents
    
    Example:
        >>> z_T = invert_latent_ddim(image, pipeline, num_inference_steps=50)
    """
    from .inversion import DDIMInverter
    
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Create inverter and invert
    inverter = DDIMInverter(pipeline)
    return inverter.invert(
        image,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        guidance_scale=guidance_scale,
        return_intermediates=return_intermediates,
    )


def extract_latent(
    image: Union[Image.Image, str],
    vae: Any,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Fast VAE-only latent extraction (alias for encode_image_to_latent).
    
    This is a convenience function that provides a clear name for the
    fast path of extracting latents without full DDIM inversion.
    
    Args:
        image: Input image (PIL Image or path)
        vae: VAE encoder module
        device: Device for computation
    
    Returns:
        Latent tensor z_0 [1, C, H, W]
    
    Example:
        >>> latent = extract_latent(image, vae, device="cuda")
    """
    return encode_image_to_latent(image, vae, device=device)


# ============================================================================
# Internal helper functions
# ============================================================================


def _pil_to_tensor(image: Image.Image, allow_resize: bool = True) -> torch.Tensor:
    """
    Convert PIL image to tensor [-1, 1].
    
    Args:
        image: PIL Image
        allow_resize: If True, resize to 512x512. If False, require exact 512x512.
        
    Returns:
        Tensor in range [-1, 1]
        
    Raises:
        ValueError: If allow_resize=False and image is not 512x512
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Handle resizing
    if image.size != (512, 512):
        if allow_resize:
            # Fast-only mode: allow resizing for convenience
            image = image.resize((512, 512), Image.LANCZOS)
        else:
            # Hybrid/full_inversion mode: require exact 512x512 for mathematical correctness
            raise ValueError(
                f"Image must be exactly 512x512 for hybrid/full_inversion detection modes. "
                f"Got {image.size[0]}x{image.size[1]}. "
                f"Resizing is disabled to ensure DDIM inversion correctness. "
                f"Please resize the image before detection."
            )
    
    # Convert to tensor [C, H, W] in range [-1, 1]
    arr = np.array(image).astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    
    return tensor


def _numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to tensor [-1, 1]."""
    if arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    arr = 2.0 * arr - 1.0
    
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] != 3:
        # Assume grayscale, convert to RGB
        arr = np.stack([arr[:, :, 0]] * 3, axis=-1)
    
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor

