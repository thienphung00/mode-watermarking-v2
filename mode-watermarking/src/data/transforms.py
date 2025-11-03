"""
Image transformations for watermark detector training.
Handles pixel-to-latent conversion, normalization, and resizing.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


class ResizeTransform:
    """
    Resize image to target size for SD compatibility.
    
    Args:
        size: Target size (height, width) or single int for square
        interpolation: PIL interpolation method (default: LANCZOS)
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]] = 512, interpolation: int = Image.LANCZOS):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: PIL Image or numpy array [H, W, C] uint8
        
        Returns:
            PIL Image resized to target size
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        return image.resize(self.size, resample=self.interpolation)


class ImageNormalizeTransform:
    """
    Normalize image to [-1, 1] range (SD preprocessing standard).
    
    Forward:
        Input: PIL Image or np.ndarray [H, W, C] uint8
        Output: torch.Tensor [C, H, W] float32, range [-1, 1]
    """
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Normalize image to [-1, 1] range.
        
        Args:
            image: PIL Image or numpy array [H, W, C] uint8
        
        Returns:
            torch.Tensor [C, H, W] float32 in range [-1, 1]
        """
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)
        else:
            image = image.astype(np.float32)
        
        # Normalize from [0, 255] to [-1, 1]
        image = (image / 127.5) - 1.0
        
        # Convert [H, W, C] to [C, H, W]
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()


class ImageToLatentTransform:
    """
    Transform image to latent representation using VAE encoder.
    
    Args:
        vae_encoder: VAE encoder model (from Stable Diffusion)
        normalize: Whether to normalize latents (default: True)
        scale_factor: VAE scale factor (default: 0.18215)
        device: Device to run VAE on (default: "cuda")
    
    Forward:
        Input: PIL Image or np.ndarray [H, W, C] uint8
        Output: torch.Tensor [C, H, W] float32 (latent)
    """
    
    def __init__(
        self,
        vae_encoder: Any,
        normalize: bool = True,
        scale_factor: float = 0.18215,
        device: str = "cuda"
    ):
        self.vae_encoder = vae_encoder
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.device = device
        
        # Normalize transform for preprocessing
        self._normalize_transform = ImageNormalizeTransform()
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Encode image to latent using VAE encoder.
        
        Args:
            image: PIL Image or numpy array [H, W, C] uint8
        
        Returns:
            torch.Tensor [C, H, W] float32 (latent representation)
        """
        if self.vae_encoder is None:
            raise ValueError("VAE encoder is required for ImageToLatentTransform")
        
        # Normalize image to [-1, 1] and convert to tensor [C, H, W]
        if self.normalize:
            img_tensor = self._normalize_transform(image)
        else:
            if isinstance(image, Image.Image):
                img_array = np.array(image, dtype=np.float32) / 255.0
            else:
                img_array = image.astype(np.float32) / 255.0
            
            if len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            
            img_tensor = torch.from_numpy(img_array).float()
        
        # Add batch dimension: [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Encode to latent
        # Handle both diffusers VAE (pipeline.vae) and direct encoder
        with torch.no_grad():
            if hasattr(self.vae_encoder, 'encode'):
                # Diffusers VAE: encode returns a distribution
                posterior = self.vae_encoder.encode(img_tensor)
                if hasattr(posterior, 'sample'):
                    latent = posterior.sample() * self.scale_factor
                elif hasattr(posterior, 'mode'):
                    latent = posterior.mode() * self.scale_factor
                else:
                    latent = posterior * self.scale_factor
            else:
                # Direct encoder call
                latent = self.vae_encoder(img_tensor)
                if hasattr(latent, 'sample'):
                    latent = latent.sample * self.scale_factor
                else:
                    latent = latent * self.scale_factor
        
        # Remove batch dimension: [C, H, W]
        latent = latent.squeeze(0)
        
        return latent.cpu()


class LatentToImageTransform:
    """
    Transform latent back to image using VAE decoder (for visualization).
    
    Args:
        vae_decoder: VAE decoder model
        scale_factor: VAE scale factor (default: 0.18215)
        device: Device to run VAE on (default: "cuda")
    
    Forward:
        Input: torch.Tensor [C, H, W] float32 (latent)
        Output: PIL Image [H, W, C] uint8
    """
    
    def __init__(
        self,
        vae_decoder: Any,
        scale_factor: float = 0.18215,
        device: str = "cuda"
    ):
        self.vae_decoder = vae_decoder
        self.scale_factor = scale_factor
        self.device = device
    
    def __call__(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to image using VAE decoder.
        
        Args:
            latent: torch.Tensor [C, H, W] float32 (latent representation)
        
        Returns:
            PIL Image [H, W, C] uint8
        """
        if self.vae_decoder is None:
            raise ValueError("VAE decoder is required for LatentToImageTransform")
        
        # Scale latent
        latent = latent / self.scale_factor
        
        # Add batch dimension: [1, C, H, W]
        latent = latent.unsqueeze(0).to(self.device)
        
        # Decode to image
        # Handle both diffusers VAE decoder and direct decoder
        with torch.no_grad():
            if hasattr(self.vae_decoder, 'decode'):
                # Diffusers VAE decoder
                image = self.vae_decoder.decode(latent)
            else:
                # Direct decoder call
                image = self.vae_decoder(latent)
        
        # Remove batch dimension and normalize: [C, H, W] -> [H, W, C]
        image = image.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        
        # Convert to numpy: [C, H, W] -> [H, W, C]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.0).astype(np.uint8)
        
        return Image.fromarray(image)


class ComposeTransforms:
    """
    Compose multiple transforms sequentially.
    Similar to torchvision.transforms.Compose but with custom types.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize transform composition.
        
        Args:
            transforms: List of transform callables to apply sequentially
        """
        self.transforms = transforms
    
    def __call__(self, sample: Any) -> Any:
        """
        Apply transforms sequentially.
        
        Args:
            sample: Input sample (image, latent, etc.)
        
        Returns:
            Transformed sample
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def __repr__(self) -> str:
        """String representation of composed transforms."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    ' + str(t)
        format_string += '\n)'
        return format_string


def get_detector_transforms(
    detector_type: str,
    vae_encoder: Optional[Any] = None,
    mode: str = "train",
    image_size: Union[int, Tuple[int, int]] = 512,
    scale_factor: float = 0.18215,
    device: str = "cuda"
) -> Callable:
    """
    Get transform pipeline for detector type.
    
    Args:
        detector_type: "unet" (latent input) or "bayesian" (g-values, future)
        vae_encoder: VAE encoder for latent transforms
        mode: "train" or "val" (for data augmentation)
        image_size: Target image size (default: 512)
        scale_factor: VAE scale factor (default: 0.18215)
        device: Device for VAE operations (default: "cuda")
    
    Returns:
        Transform callable that takes PIL Image or np.ndarray and returns tensor
    """
    transforms = []
    
    # Step 1: Resize to target size (required for all inputs)
    transforms.append(ResizeTransform(size=image_size))
    
    if detector_type == "unet":
        # For UNet detector: resize -> normalize -> encode to latent
        if vae_encoder is not None:
            transforms.append(ImageToLatentTransform(
                vae_encoder=vae_encoder,
                normalize=True,
                scale_factor=scale_factor,
                device=device
            ))
        else:
            # Fallback to normalized pixel values if VAE not available
            transforms.append(ImageNormalizeTransform())
    elif detector_type == "bayesian":
        # For Bayesian detector: currently return normalized pixels
        # Future: add g-value recovery transform in Stage 5
        transforms.append(ImageNormalizeTransform())
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}. Use 'unet' or 'bayesian'")
    
    # Future: Add data augmentation transforms based on mode
    # if mode == "train" and augmentation_enabled:
    #     transforms.insert(-1, RandomHorizontalFlip(p=0.5))
    #     transforms.insert(-1, ColorJitter(...))
    
    return ComposeTransforms(transforms)
