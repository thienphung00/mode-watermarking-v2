#!/usr/bin/env python3
"""
Post-hoc manifest patching: attach deterministic positional mask to existing manifests.

This script generates a deterministic mask based on g-field configuration and patches
all manifest entries to include a reference to the mask file.

The mask is:
- Deterministic (depends only on g-field config and geometry)
- Saved once per dataset as mask.pt
- Referenced by all manifest entries

Usage:
    python scripts/patch_manifest_with_mask.py \
        --manifest outputs/train/manifest.jsonl \
        --g-field-config configs/experiments/seedbias.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.algorithms.g_field import GFieldGenerator
from src.core.config import AppConfig, GFieldConfig


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load manifest.jsonl file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    entries = []
    with open(manifest_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} of {manifest_path}: {e}")
    
    if not entries:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    
    return entries


def save_manifest(manifest_path: Path, entries: List[Dict[str, Any]]) -> None:
    """Save manifest.jsonl file."""
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")


def load_g_field_config(config_path: Optional[Path]) -> GFieldConfig:
    """
    Load g-field configuration from config file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        GFieldConfig instance
    """
    if config_path is None:
        # Use defaults matching typical SD setup
        return GFieldConfig(
            shape=(4, 64, 64),
            domain="frequency",
            frequency_mode="bandpass",
            low_freq_cutoff=0.05,
        )
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load full config and extract g-field config
    config = AppConfig.from_yaml(str(config_path))
    
    if isinstance(config.watermark, dict):
        # Handle dict format (shouldn't happen with Pydantic, but be safe)
        g_field_dict = config.watermark.get("algorithm_params", {}).get("g_field", {})
        return GFieldConfig(**g_field_dict)
    else:
        # Pydantic model format
        if hasattr(config.watermark, "algorithm_params") and hasattr(config.watermark.algorithm_params, "g_field"):
            return config.watermark.algorithm_params.g_field
        else:
            raise ValueError(
                f"Config file {config_path} does not contain g_field configuration. "
                "Please provide a valid watermark config with g_field settings."
            )


def create_generator_from_config(
    g_field_config: GFieldConfig,
    config_path: Optional[Path] = None,
) -> GFieldGenerator:
    """
    Create GFieldGenerator from GFieldConfig.
    
    This function delegates all mask geometry logic to GFieldGenerator,
    which is the single source of truth for mask semantics.
    
    Args:
        g_field_config: G-field configuration
        config_path: Optional config path to extract high_freq_cutoff if needed
    
    Returns:
        GFieldGenerator instance configured with the g-field settings
    """
    # Extract normalization settings from config
    normalize_dict = g_field_config.normalize if isinstance(g_field_config.normalize, dict) else {}
    normalize_zero_mean = normalize_dict.get("zero_mean_per_timestep", True) or normalize_dict.get("zero_mean_per_channel", True)
    normalize_unit_variance = normalize_dict.get("unit_variance", False)
    
    # Build generator kwargs
    generator_kwargs = {
        "mapping_mode": g_field_config.mapping_mode,
        "domain": g_field_config.domain,
        "frequency_mode": g_field_config.frequency_mode,
        "low_freq_cutoff": g_field_config.low_freq_cutoff,
        "normalize_zero_mean": normalize_zero_mean,
        "normalize_unit_variance": normalize_unit_variance,
    }
    
    # Add optional fields
    if g_field_config.continuous_range is not None:
        generator_kwargs["continuous_range"] = g_field_config.continuous_range
    
    # For bandpass mode, GFieldGenerator requires high_freq_cutoff
    # Extract it from seed_bias config if available (let generator handle validation)
    if g_field_config.frequency_mode.lower() == "bandpass" and config_path is not None:
        try:
            config = AppConfig.from_yaml(str(config_path))
            if hasattr(config.watermark, "algorithm_params") and hasattr(config.watermark.algorithm_params, "seed_bias"):
                if config.watermark.algorithm_params.seed_bias is not None:
                    if hasattr(config.watermark.algorithm_params.seed_bias, "high_freq_cutoff"):
                        generator_kwargs["high_freq_cutoff"] = config.watermark.algorithm_params.seed_bias.high_freq_cutoff
        except Exception:
            # If extraction fails, let GFieldGenerator handle the error
            pass
    
    return GFieldGenerator(**generator_kwargs)


def generate_deterministic_mask(
    g_field_config: GFieldConfig,
    config_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Generate deterministic positional mask using GFieldGenerator.
    
    This function delegates all mask geometry logic to GFieldGenerator,
    which is the single source of truth for mask semantics.
    
    The mask is deterministic and depends only on:
    - g-field shape (C, H, W)
    - domain (spatial/frequency)
    - frequency mode and cutoffs (if frequency domain)
    
    Args:
        g_field_config: G-field configuration
        config_path: Optional config path (used to extract high_freq_cutoff if needed)
    
    Returns:
        Binary mask tensor [N] where N = C * H * W, values in {0, 1}
    """
    C, H, W = g_field_config.shape
    
    # Create generator from config (delegates all geometry logic)
    generator = create_generator_from_config(g_field_config, config_path)
    
    # Generate dummy G-field with mask (seeds don't matter for mask geometry)
    # Use zeros as dummy seeds - mask is independent of seed values
    num_elements = C * H * W
    dummy_seeds = [0] * num_elements
    
    # Call generate_g_field with return_mask=True to get mask from generator
    _, mask_2d = generator.generate_g_field(
        shape=(C, H, W),
        seeds=dummy_seeds,
        return_mask=True,
    )
    
    # mask_2d is [H, W] from generator
    # Broadcast to [C, H, W] (same mask for all channels)
    mask_3d = np.broadcast_to(mask_2d[np.newaxis, :, :], (C, H, W)).copy()
    
    # Flatten to 1D [N] where N = C * H * W
    mask_flat = mask_3d.flatten()
    
    # Convert to binary {0, 1} and then to torch tensor
    mask_binary = (mask_flat > 0.5).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_binary)
    
    return mask_tensor


def compute_mask_checksum(mask: torch.Tensor) -> str:
    """Compute deterministic checksum of mask for validation."""
    mask_np = mask.numpy()
    mask_bytes = mask_np.tobytes()
    return hashlib.sha256(mask_bytes).hexdigest()[:16]


def patch_manifest(
    manifest_path: Path,
    g_field_config: Optional[GFieldConfig] = None,
    config_path: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """
    Patch manifest to include mask field.
    
    This function delegates all mask generation to GFieldGenerator,
    which is the single source of truth for mask semantics.
    
    Args:
        manifest_path: Path to manifest.jsonl file
        g_field_config: Optional GFieldConfig (if None, will load from config_path)
        config_path: Optional path to config file (if g_field_config is None)
        overwrite: If True, overwrite existing mask field
    """
    print(f"=" * 60)
    print(f"Patching manifest: {manifest_path}")
    print(f"=" * 60)
    
    # Load manifest
    entries = load_manifest(manifest_path)
    print(f"Loaded {len(entries)} manifest entries")
    
    # Check if mask already exists
    has_mask = any("mask" in entry for entry in entries)
    if has_mask and not overwrite:
        raise ValueError(
            f"Manifest already contains 'mask' field. "
            "Use --overwrite to replace existing mask references."
        )
    
    # Load g-field config
    if g_field_config is None:
        print(f"Loading g-field config from: {config_path}")
        g_field_config = load_g_field_config(config_path)
    else:
        print("Using provided g-field config")
    
    print(f"G-field config:")
    print(f"  Shape: {g_field_config.shape}")
    print(f"  Domain: {g_field_config.domain}")
    if g_field_config.domain == "frequency":
        print(f"  Frequency mode: {g_field_config.frequency_mode}")
        print(f"  Low freq cutoff: {g_field_config.low_freq_cutoff}")
    
    # Generate deterministic mask using GFieldGenerator (single source of truth)
    print("\nGenerating deterministic mask using GFieldGenerator...")
    mask = generate_deterministic_mask(g_field_config, config_path)
    
    # Validate mask
    N = mask.shape[0]
    expected_N = g_field_config.shape[0] * g_field_config.shape[1] * g_field_config.shape[2]
    if N != expected_N:
        raise ValueError(
            f"Mask length {N} does not match expected {expected_N} "
            f"(C={g_field_config.shape[0]}, H={g_field_config.shape[1]}, W={g_field_config.shape[2]})"
        )
    
    # Compute checksum for validation
    mask_checksum = compute_mask_checksum(mask)
    num_valid = int(mask.sum().item())
    
    print(f"Mask generated:")
    print(f"  Shape: [{N}] (flattened from {g_field_config.shape})")
    print(f"  Valid positions: {num_valid} / {N} ({100.0 * num_valid / N:.1f}%)")
    print(f"  Checksum: {mask_checksum}")
    
    # Save mask
    manifest_dir = manifest_path.parent
    mask_path = manifest_dir / "mask.pt"
    torch.save(mask, mask_path)
    print(f"\n✓ Saved mask to: {mask_path}")
    
    # Patch manifest entries
    print(f"\nPatching {len(entries)} manifest entries...")
    patched_count = 0
    for entry in entries:
        if "mask" in entry and not overwrite:
            continue  # Skip if already has mask and not overwriting
        entry["mask"] = "mask.pt"
        patched_count += 1
    
    # Save patched manifest
    save_manifest(manifest_path, entries)
    print(f"✓ Patched {patched_count} entries")
    print(f"✓ Saved updated manifest to: {manifest_path}")
    
    print(f"\n" + "=" * 60)
    print(f"✅ Manifest patching complete!")
    print(f"=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Patch manifest with deterministic positional mask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.jsonl file",
    )
    
    parser.add_argument(
        "--g-field-config",
        type=str,
        default=None,
        help="Path to YAML config file containing g-field configuration",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask field if present",
    )
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    config_path = Path(args.g_field_config) if args.g_field_config else None
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    if args.g_field_config and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    patch_manifest(
        manifest_path=manifest_path,
        config_path=config_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

