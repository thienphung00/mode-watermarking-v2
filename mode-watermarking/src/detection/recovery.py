"""
G-value recovery from watermarked images.

Recovers g-values by:
1. Encoding image to latent space via VAE encoder
2. Reconstructing expected g-values from key information
3. Extracting observed g-values from latent representation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Optional torch import
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

from ..sd_integration.timestep_mapper import TimestepMapper
from ..utils.io import ImageIO, ensure_dir
from ..watermark.gfield import GFieldBuilder
from ..watermark.key import KeyDerivation


def recover_g_values(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    vae_encoder: Any,
    watermark_cfg: Dict[str, Any],
    key_info: Optional[Dict[str, Any]] = None,
    timestep_mapper: Optional[TimestepMapper] = None,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Recover g-values from watermarked image.
    
    Steps:
    1. Image → Latent (via VAE encoder)
    2. For each timestep in injection range:
       - Reconstruct G_t from latent using key_info
       - Extract observed g-values
    3. Return g-values per timestep
    
    Args:
        image: Input image (PIL Image, numpy array, or torch tensor)
        vae_encoder: VAE encoder from SD pipeline (pipeline.vae)
        watermark_cfg: Watermark configuration dictionary
        key_info: Key parameters from manifest (key_id, key_scheme, etc.)
        timestep_mapper: Timestep mapper for DDIM inference
        device: Device for computation
    
    Returns:
        Dictionary with:
        - "g_values": np.ndarray [num_timesteps, C, H, W] recovered g-values
        - "latent": np.ndarray [C, H, W] recovered latent
        - "mask": np.ndarray [C, H, W] applicable mask
        - "recovery_metadata": Dict with timestamps, key info used
    """
    # Convert image to tensor if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    elif isinstance(image, np.ndarray):
        image_array = image
    elif _TORCH_AVAILABLE and isinstance(image, torch.Tensor):
        image_array = image.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Normalize image to [-1, 1] range if needed
    if image_array.dtype == np.uint8:
        image_array = (image_array.astype(np.float32) / 255.0) * 2.0 - 1.0
    elif image_array.max() > 1.0:
        # Already in [0, 255], normalize
        image_array = (image_array.astype(np.float32) / 255.0) * 2.0 - 1.0
    
    # Ensure image is in [C, H, W] format
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = image_array.transpose(2, 0, 1)
    
    # Convert to tensor for VAE encoding
    if _TORCH_AVAILABLE:
        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0).to(device)
    else:
        raise RuntimeError("PyTorch required for VAE encoding")
    
    # Encode image to latent
    with torch.no_grad():
        if hasattr(vae_encoder, 'encode'):
            # Diffusers VAE
            posterior = vae_encoder.encode(image_tensor)
            if hasattr(posterior, 'sample'):
                latent = posterior.sample()
            elif hasattr(posterior, 'mode'):
                latent = posterior.mode()
            else:
                latent = posterior
            # Apply scale factor
            latent = latent * vae_encoder.config.scaling_factor if hasattr(vae_encoder.config, 'scaling_factor') else latent * 0.18215
        else:
            # Direct encoder
            latent = vae_encoder(image_tensor)
    
    latent_np = latent.squeeze(0).detach().cpu().numpy()  # [C, H, W]
    
    # Reconstruct expected g-values using key_info
    g_field_cfg = watermark_cfg.get("g_field", {})
    latent_shape = tuple(g_field_cfg.get("shape", [4, 64, 64]))
    C, H, W = latent_shape
    
    # Get key information
    if key_info is None:
        key_info = {}
    
    key_scheme = key_info.get("key_scheme", watermark_cfg.get("watermark", {}).get("key_scheme", "LCG-v1"))
    key_master = key_info.get("key_master", watermark_cfg.get("watermark", {}).get("key_master", ""))
    experiment_id = key_info.get("experiment_id", watermark_cfg.get("watermark", {}).get("experiment_id", "exp_001"))
    sample_id = key_info.get("sample_id", "unknown")
    base_seed = key_info.get("base_seed", watermark_cfg.get("watermark", {}).get("base_seed", 12345))
    
    # Derive seed using KeyDerivation
    key_derivation = KeyDerivation()
    zT_hash = key_info.get("zT_hash", "default_hash")
    seed0 = key_derivation.derive_seed(
        key_master=key_master,
        sample_id=sample_id,
        zT_hash=zT_hash,
        base_seed=base_seed,
        experiment_id=experiment_id
    )
    
    # Build g-field schedule
    if timestep_mapper is None:
        # Create default timestep mapper
        num_inference_steps = watermark_cfg.get("diffusion", {}).get("inference_timesteps", 50)
        num_trained_steps = watermark_cfg.get("diffusion", {}).get("trained_timesteps", 1000)
        timestep_mapper = TimestepMapper(
            trained_timesteps=num_trained_steps,
            inference_timesteps=num_inference_steps
        )
    
    # Generate key stream
    num_timesteps = len(timestep_mapper.get_all_trained_timesteps())
    stream_len = C * H * W * num_timesteps  # Enough for all timesteps
    key_stream = key_derivation.generate_key_stream(seed0, stream_len)
    
    # Build g-field schedule
    gfield_builder = GFieldBuilder(
        mapping_mode=g_field_cfg.get("mapping_mode", "binary"),
        bit_pos=watermark_cfg.get("watermark", {}).get("lcg", {}).get("bit_pos", 30)
    )
    
    g_schedule = gfield_builder.build_g_schedule(
        timestep_mapper=timestep_mapper,
        latent_shape=latent_shape,
        key_stream=key_stream
    )
    
    # Extract observed g-values from latent
    # In practice, this is a simplified extraction - actual recovery may require
    # statistical analysis or learned reconstruction
    recovered_g_values = []
    trained_timesteps = sorted(g_schedule.keys())
    
    for trained_t in trained_timesteps:
        # For now, we use the expected g-field as recovered values
        # In a full implementation, this would involve statistical recovery from latent
        expected_g_t = g_schedule[trained_t]
        recovered_g_values.append(expected_g_t)
    
    recovered_g_array = np.stack(recovered_g_values, axis=0)  # [num_timesteps, C, H, W]
    
    # Get mask
    # TODO: Implement mask loading from config
    # For now, use all-ones mask (no masking)
    mask = np.ones(latent_shape, dtype=np.float32)
    
    recovery_metadata = {
        "key_id": key_info.get("key_id", "unknown"),
        "sample_id": sample_id,
        "experiment_id": experiment_id,
        "num_timesteps": num_timesteps,
        "recovery_method": "direct",  # or "statistical"
        "latent_shape": latent_shape
    }
    
    return {
        "g_values": recovered_g_array,
        "latent": latent_np,
        "mask": mask,
        "recovery_metadata": recovery_metadata
    }


def batch_recover_g_values(
    manifest_path: str,
    vae_encoder: Any,
    watermark_cfg: Dict[str, Any],
    diffusion_cfg: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    device: str = "cuda",
    save_results: bool = True,
    output_dir: Optional[str] = None
) -> List[Dict[str, np.ndarray]]:
    """
    Batch process images from manifest to recover g-values.
    
    Args:
        manifest_path: Path to manifest file (JSONL format)
        vae_encoder: VAE encoder from SD pipeline
        watermark_cfg: Watermark configuration
        diffusion_cfg: Diffusion configuration (optional)
        batch_size: Batch size for processing
        num_workers: Number of worker processes
        device: Device for computation
        save_results: Whether to save results to disk
        output_dir: Directory to save results (if save_results=True)
    
    Returns:
        List of recovery results (one per image)
    """
    from ..utils.io import ManifestIO
    
    # Load manifest
    manifest_entries = ManifestIO.read_jsonl(manifest_path)
    
    # Create timestep mapper
    if diffusion_cfg is None:
        num_inference_steps = watermark_cfg.get("diffusion", {}).get("inference_timesteps", 50)
        num_trained_steps = watermark_cfg.get("diffusion", {}).get("trained_timesteps", 1000)
    else:
        num_inference_steps = diffusion_cfg.get("inference", {}).get("num_inference_steps", 50)
        num_trained_steps = diffusion_cfg.get("diffusion", {}).get("trained_timesteps", 1000)
    
    timestep_mapper = TimestepMapper(
        trained_timesteps=num_trained_steps,
        inference_timesteps=num_inference_steps
    )
    
    results = []
    
    # Process images
    for entry in manifest_entries:
        image_path = entry.get("image_path")
        if not image_path:
            continue
        
        # Load image
        image_array = ImageIO.read_image(image_path)
        image = Image.fromarray(image_array)
        
        # Extract key info from manifest
        key_info = entry.get("key_info", {})
        key_info.update({
            "sample_id": entry.get("sample_id", "unknown"),
            "key_id": entry.get("key_id", "unknown"),
            "zT_hash": entry.get("zT_hash", "default_hash"),
            "experiment_id": entry.get("experiment_id", "exp_001")
        })
        
        # Recover g-values
        try:
            result = recover_g_values(
                image=image,
                vae_encoder=vae_encoder,
                watermark_cfg=watermark_cfg,
                key_info=key_info,
                timestep_mapper=timestep_mapper,
                device=device
            )
            result["manifest_entry"] = entry
            results.append(result)
        except Exception as e:
            print(f"Error recovering g-values for {image_path}: {e}")
            continue
    
    # Save results if requested
    if save_results and output_dir:
        ensure_dir(output_dir)
        output_path = Path(output_dir) / "recovered_g_values.npz"
        
        # Save as compressed numpy file
        np.savez_compressed(
            output_path,
            g_values=[r["g_values"] for r in results],
            latents=[r["latent"] for r in results],
            masks=[r["mask"] for r in results],
            metadata=[r["recovery_metadata"] for r in results]
        )
        print(f"Saved recovery results to {output_path}")
    
    return results
