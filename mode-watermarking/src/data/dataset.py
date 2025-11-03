"""
PyTorch Dataset classes for watermark detector training.
Loads images and metadata from manifest files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.io import ImageIO, ManifestIO, AssetStore


def load_split_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate manifest file.
    
    Args:
        manifest_path: Path to JSON manifest file
        
    Returns:
        List of manifest entries (one per image)
    
    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest format is invalid
    """
    manifest_path_obj = Path(manifest_path)
    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Load JSON manifest
    manifest_data = ManifestIO.read_json(manifest_path)
    
    # Validate format
    if not isinstance(manifest_data, list):
        raise ValueError(f"Manifest must be a list of entries, got {type(manifest_data)}")
    
    # Validate each entry has required fields
    required_fields = ["image_path"]
    for i, entry in enumerate(manifest_data):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {i} must be a dictionary")
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Manifest entry {i} missing required field: {field}")
    
    return manifest_data


def extract_label_from_manifest(entry: Dict[str, Any], label_key: Optional[str] = None) -> int:
    """
    Extract label from manifest entry.
    
    Label extraction priority:
    1. Explicit 'label' field in entry
    2. label_key if provided
    3. Inferred from image_path (contains "watermarked" or "unwatermarked")
    4. Inferred from metadata['mode'] field
    
    Args:
        entry: Manifest entry dictionary
        label_key: Optional key to extract label from
    
    Returns:
        Label (0=unwatermarked, 1=watermarked)
    """
    # Priority 1: Explicit label field
    if "label" in entry:
        label = entry["label"]
        if isinstance(label, (int, bool)):
            return int(label)
        if isinstance(label, str):
            return 1 if label.lower() in ["watermarked", "1", "true"] else 0
    
    # Priority 2: label_key
    if label_key and label_key in entry:
        label = entry[label_key]
        return 1 if label else 0
    
    # Priority 3: Infer from image_path
    image_path = entry.get("image_path", "")
    if "watermarked" in image_path.lower():
        return 1
    if "unwatermarked" in image_path.lower():
        return 0
    
    # Priority 4: Infer from metadata mode
    metadata = entry.get("metadata", {})
    if isinstance(metadata, dict):
        mode = metadata.get("mode", "")
        if "distortion" in mode.lower() or "watermark" in mode.lower():
            return 1
    
    # Default: assume unwatermarked if unclear
    return 0


class WatermarkDataset(Dataset):
    """
    Base dataset for watermark detector training.
    Loads images and labels from manifest files.
    
    Returns:
        Dict with keys:
        - 'image': torch.Tensor [C, H, W]
        - 'label': int (0=unwatermarked, 1=watermarked)
        - 'metadata': Dict with manifest fields
    """
    
    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
        label_key: Optional[str] = None,
        image_root: Optional[str] = None
    ):
        """
        Initialize watermark dataset.
        
        Args:
            manifest_path: Path to JSON manifest file
            transform: Optional transform to apply to images
            label_key: Key in manifest to extract label (optional)
            image_root: Root directory for resolving image paths (optional)
        """
        self.manifest_path = manifest_path
        self.transform = transform
        self.label_key = label_key
        self.image_root = image_root
        
        # Load manifest
        self.entries = load_split_manifest(manifest_path)
        
        # Initialize asset store for path resolution
        if image_root:
            self.asset_store = AssetStore(image_root)
        else:
            self.asset_store = None
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.entries)
    
    def _resolve_image_path(self, image_path: str) -> str:
        """Resolve image path relative to image_root if provided."""
        if self.asset_store:
            # If image_path is relative, resolve relative to image_root
            if not Path(image_path).is_absolute():
                return self.asset_store.resolve(image_path)
        return image_path
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return one sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'image', 'label', and 'metadata' keys
        """
        entry = self.entries[idx]
        
        # Resolve image path
        image_path = self._resolve_image_path(entry["image_path"])
        
        # Load image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load as PIL Image first (more flexible for transforms)
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image_array = np.array(image, dtype=np.float32) / 255.0
            if len(image_array.shape) == 3:
                image_array = np.transpose(image_array, (2, 0, 1))
            image = torch.from_numpy(image_array).float()
        
        # Extract label
        label = extract_label_from_manifest(entry, self.label_key)
        
        # Extract metadata (exclude image_path and label for cleaner metadata)
        metadata = {k: v for k, v in entry.items() if k not in ["image_path", "label"]}
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": metadata
        }


class LatentWatermarkDataset(WatermarkDataset):
    """
    Dataset that returns latent representations for UNet detector.
    Uses VAE encoder to convert images → latents.
    
    Args:
        vae_encoder: VAE encoder model (from Stable Diffusion)
        latent_shape: Expected latent shape [C, H, W] (default: [4, 64, 64])
        All other args same as WatermarkDataset
    """
    
    def __init__(
        self,
        manifest_path: str,
        vae_encoder: Optional[Any] = None,
        latent_shape: Tuple[int, int, int] = (4, 64, 64),
        transform: Optional[Callable] = None,
        image_root: Optional[str] = None,
        label_key: Optional[str] = None
    ):
        """
        Initialize latent watermark dataset.
        
        Args:
            manifest_path: Path to JSON manifest file
            vae_encoder: VAE encoder model for latent conversion
            latent_shape: Expected latent shape [C, H, W]
            transform: Transform to apply (should include VAE encoding)
            image_root: Root directory for resolving image paths
            label_key: Key in manifest to extract label
        """
        super().__init__(
            manifest_path=manifest_path,
            transform=transform,
            label_key=label_key,
            image_root=image_root
        )
        self.vae_encoder = vae_encoder
        self.latent_shape = latent_shape
        
        # If transform is provided, it should handle VAE encoding
        # Otherwise, we'll encode in __getitem__ if vae_encoder is available
        if transform is None and vae_encoder is not None:
            from .transforms import ImageToLatentTransform, ResizeTransform, ComposeTransforms
            
            # Create transform pipeline: resize -> encode to latent
            self.transform = ComposeTransforms([
                ResizeTransform(size=512),  # SD standard size
                ImageToLatentTransform(
                    vae_encoder=vae_encoder,
                    normalize=True,
                    scale_factor=0.18215
                )
            ])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return one sample with latent representation.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'image' (latent), 'label', and 'metadata' keys
        """
        sample = super().__getitem__(idx)
        
        # Verify latent shape if transform was applied
        if self.transform and hasattr(sample["image"], "shape"):
            expected_shape = self.latent_shape
            actual_shape = sample["image"].shape
            if actual_shape != expected_shape:
                print(f"Warning: Expected latent shape {expected_shape}, got {actual_shape}")
        
        return sample


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    dataset_class: Type[WatermarkDataset],
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        dataset_class: Dataset class to instantiate (WatermarkDataset or LatentWatermarkDataset)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes (default: 0 for debugging)
        shuffle_train: Whether to shuffle training data (default: True)
        shuffle_val: Whether to shuffle validation data (default: False)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        **dataset_kwargs: Additional arguments for dataset constructor
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create training dataset
    train_dataset = dataset_class(manifest_path=train_manifest, **dataset_kwargs)
    
    # Create validation dataset
    val_dataset = dataset_class(manifest_path=val_manifest, **dataset_kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Keep all samples even if last batch is smaller
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
