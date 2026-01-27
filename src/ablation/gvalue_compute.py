"""
G-value computation for ablation experiments.

Reusable functions for computing g-values from latents with caching.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from ..core.config import AppConfig
from ..core.key_utils import UNWATERMARKED_DUMMY_KEY
from ..detection.inversion import DDIMInverter
from ..detection.g_values import compute_g_values, g_field_config_to_dict
from .latent_cache import LatentCache

logger = logging.getLogger(__name__)


def compute_g_values_for_family(
    config: AppConfig,
    manifest: List[Dict[str, Any]],
    manifest_dir: Path,
    master_key: str,
    device: str,
    latent_cache: Optional[LatentCache] = None,
    inverter: Optional[DDIMInverter] = None,
    pipeline: Optional[Any] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute g-values for all samples in manifest.
    
    Args:
        config: Watermark configuration
        manifest: List of manifest entries (with image_path, key_id, etc.)
        manifest_dir: Base directory for resolving image paths
        master_key: Master key for PRF
        device: Device to run on
        latent_cache: Optional latent cache instance
        inverter: Optional pre-created inverter (will create if None)
        pipeline: Optional pre-created pipeline (will create if None)
        
    Returns:
        Tuple of (g_values, mask, metadata):
            - g_values: Tensor [N, D] where N is num samples, D is g-value dimension
            - mask: Tensor [D] binary mask for valid positions
            - metadata: Dictionary with stats (N_eff, etc.)
    """
    from ..engine.pipeline import create_pipeline
    
    # Create pipeline and inverter if needed
    if pipeline is None:
        pipeline = create_pipeline(config.diffusion, device=device)
    if inverter is None:
        inverter = DDIMInverter(pipeline, device=device)
    
    # Create latent cache if needed
    if latent_cache is None:
        latent_cache = LatentCache(
            cache_dir=manifest_dir / "latents",
            model_id=config.diffusion.model_id,
            num_inversion_steps=25,
        )
    
    # Get g-field config
    g_field_config = g_field_config_to_dict(config.watermark.algorithm_params.g_field)
    
    # Process all samples
    g_list = []
    mask_list = []
    N_eff_list = []
    S_list = []
    
    logger.info(f"Computing g-values for {len(manifest)} samples...")
    
    for entry in tqdm(manifest, desc="Computing g-values"):
        image_path = manifest_dir / entry["image_path"]
        key_id = entry.get("key_id")
        
        # Use dummy key for clean samples
        computation_key = key_id if key_id is not None else UNWATERMARKED_DUMMY_KEY
        
        # Get latent (from cache or compute)
        latent_T = latent_cache.get(image_path, inverter, device=device)
        
        # Compute g-values
        g, mask = compute_g_values(
            latent_T,
            computation_key,
            master_key,
            return_mask=True,
            g_field_config=g_field_config,
            latent_type="zT",
        )  # g: [1, N] or [N], mask: [1, N] or [N]
        
        # Ensure 1D
        if g.dim() > 1:
            g = g.flatten()
        if mask is not None and mask.dim() > 1:
            mask = mask.flatten()
        
        # Apply mask: select only valid positions
        if mask is not None:
            g_valid = g[mask > 0.5]  # [N_eff]
            N_eff = int((mask > 0.5).sum().item())
        else:
            g_valid = g  # [N]
            N_eff = int(g.numel())
        
        # g is binary {0, 1} from compute_g_values
        g_binary = (g_valid > 0).float()  # [N_eff]
        
        # Compute S = sum(g_valid)
        S = float(g_binary.sum().item())
        
        g_list.append(g_binary.cpu().numpy())
        if mask is not None:
            # Store mask once (should be same for all samples)
            mask_valid = mask[mask > 0.5].cpu().numpy()
            mask_list.append(mask_valid)
        N_eff_list.append(N_eff)
        S_list.append(S)
    
    # Stack g-values
    g_array = np.stack(g_list)  # [N, N_eff]
    
    # Get mask (use first one, should be same for all)
    if mask_list:
        mask_array = mask_list[0]  # [N_eff]
    else:
        # No mask: all positions valid
        mask_array = np.ones(g_array.shape[1], dtype=np.float32)
    
    # Convert to tensors
    g_tensor = torch.from_numpy(g_array).float()  # [N, N_eff]
    mask_tensor = torch.from_numpy(mask_array).float()  # [N_eff]
    
    # Compute metadata
    metadata = {
        "num_samples": len(manifest),
        "N_eff": int(N_eff),
        "N_eff_mean": float(np.mean(N_eff_list)),
        "N_eff_std": float(np.std(N_eff_list)),
        "S_mean": float(np.mean(S_list)),
        "S_std": float(np.std(S_list)),
        "g_shape": list(g_tensor.shape),
        "mask_shape": list(mask_tensor.shape),
    }
    
    return g_tensor, mask_tensor, metadata

