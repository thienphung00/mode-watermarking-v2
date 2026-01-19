#!/usr/bin/env python3
"""
Phase 0: Group ablation configs into detector families.

This script:
1. Loads all YAML configs from a directory
2. Extracts family signatures based on detector geometry (mapping_mode, g_field, mask)
3. Groups configs into families (watermark strength differences do NOT create new families)
4. Saves families with signature.json and configs.json

Family signature includes:
- mapping_mode
- g_field geometry (domain, frequency_mode, cutoffs, normalization)
- mask behavior (mode, band, cutoff_freq, bandwidth_fraction)

Watermark strength (lambda_strength, mask_strength) is EXCLUDED from signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

from src.core.config import AppConfig, WatermarkedConfig, extract_detector_geometry_signature, normalize_geometry_signature


def extract_family_signature(config: AppConfig) -> Dict[str, Any]:
    """
    Extract family signature from config using authoritative schema.
    
    This function uses the authoritative geometry schema defined in
    src.core.config.extract_detector_geometry_signature() to ensure
    completeness and consistency.
    
    Family signature includes ALL detector geometry parameters:
    - mapping_mode
    - All g_field parameters (grid size, bounds, normalization, cutoff, transforms, etc.)
    - All mask parameters (shape, smoothing, padding, scaling, thresholds, etc.)
    
    Watermark strength (lambda_strength, mask_strength) is EXCLUDED.
    
    Args:
        config: AppConfig instance
        
    Returns:
        Normalized signature dictionary
        
    Raises:
        ValueError: If config is not watermarked or if any required field is missing
    """
    if config.watermark.mode != "watermarked":
        raise ValueError(f"Expected watermarked config, got {config.watermark.mode}")
    
    if not isinstance(config.watermark, WatermarkedConfig):
        raise ValueError(f"Expected WatermarkedConfig, got {type(config.watermark)}")
    
    # Use authoritative schema from config module
    signature = extract_detector_geometry_signature(config.watermark)
    
    return signature


# normalize_signature is now imported from src.core.config


def hash_signature(signature: Dict[str, Any]) -> str:
    """
    Hash signature to get deterministic family ID.
    
    Args:
        signature: Normalized signature dictionary
        
    Returns:
        Family ID string (e.g., "family_001")
    """
    # Convert to JSON string with sorted keys
    json_str = json.dumps(signature, sort_keys=True, separators=(',', ':'))
    
    # Hash
    hash_obj = hashlib.md5(json_str.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:8]
    
    # Convert to family ID (deterministic based on hash)
    # Use first 8 hex digits to create a numeric ID
    family_num = int(hash_hex, 16) % 1000
    family_id = f"family_{family_num:03d}"
    
    return family_id


def group_configs_into_families(configs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Group configs into families based on detector geometry.
    
    Args:
        configs_dir: Directory containing YAML config files
        
    Returns:
        Dictionary mapping family_id to family data:
        {
            "family_001": {
                "signature": {...},
                "configs": [path1, path2, ...]
            },
            ...
        }
    """
    families: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "signature": None,
        "configs": []
    })
    
    # Load all configs
    config_files = sorted(configs_dir.glob("*.yaml"))
    
    if not config_files:
        raise ValueError(f"No YAML config files found in {configs_dir}")
    
    print(f"Loading {len(config_files)} config files...")
    
    for config_path in config_files:
        try:
            config = AppConfig.from_yaml(str(config_path))
            
            # Extract signature using authoritative schema
            signature = extract_family_signature(config)
            normalized = normalize_geometry_signature(signature)
            family_id = hash_signature(normalized)
            
            # Add to family
            if families[family_id]["signature"] is None:
                families[family_id]["signature"] = normalized
            else:
                # CRITICAL VALIDATION: Verify signature matches (should be byte-identical)
                existing_sig = families[family_id]["signature"]
                if existing_sig != normalized:
                    # Convert to JSON for readable error message
                    import json
                    existing_json = json.dumps(existing_sig, sort_keys=True, indent=2)
                    new_json = json.dumps(normalized, sort_keys=True, indent=2)
                    raise ValueError(
                        f"❌ SIGNATURE MISMATCH for family {family_id}:\n"
                        f"This indicates a bug in signature extraction or config inconsistency.\n"
                        f"Config: {config_path.name}\n"
                        f"Existing signature:\n{existing_json}\n"
                        f"New signature:\n{new_json}\n"
                        f"Signatures must be byte-identical for configs in the same family."
                    )
            
            families[family_id]["configs"].append(str(config_path.relative_to(configs_dir)))
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to process {config_path.name}: {e}")
            continue
    
    # Convert defaultdict to regular dict
    return dict(families)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Group ablation configs into detector families",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Group existing configs into families
  python scripts/generate_ablation_configs.py \\
    --configs-dir experiments/watermark_ablation/configs \\
    --output-dir experiments/watermark_ablation/families
        """
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/configs"),
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/watermark_ablation/families"),
        help="Output directory for family groupings",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {args.configs_dir}")
    
    # Group configs into families
    families = group_configs_into_families(args.configs_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save families
    print(f"\nGrouping configs into {len(families)} families...")
    
    for family_id, family_data in sorted(families.items()):
        family_dir = args.output_dir / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
                
        # Save signature (normalized, canonical JSON format)
        signature_path = family_dir / "signature.json"
        with open(signature_path, "w") as f:
            json.dump(family_data["signature"], f, indent=2, sort_keys=True)
        
        # Log signature for auditability
        signature_json = json.dumps(family_data["signature"], sort_keys=True, indent=2)
        print(f"✓ {family_id}: {len(family_data['configs'])} configs")
        print(f"    Signature (normalized):\n{signature_json}")
        print(f"    Signature file: {signature_path}")
                
        # Save configs list
        configs_path = family_dir / "configs.json"
        with open(configs_path, "w") as f:
            json.dump(family_data["configs"], f, indent=2)
        print(f"    Configs file: {configs_path}")
    
    print(f"\n✓ Grouped {sum(len(f['configs']) for f in families.values())} configs into {len(families)} families")
    print(f"✓ Families saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

