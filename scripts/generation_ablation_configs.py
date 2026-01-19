#!/usr/bin/env python3
"""
Phase 0: Generate detector family groupings from watermark configs.

This script:
1. Scans a directory of watermark configs
2. Groups configs by detector family (based on shared detector geometry)
3. Emits a families.json manifest file

Usage:
    python scripts/generation_ablation_configs.py \
        --configs-dir experiments/watermark_ablation/configs \
        --output experiments/watermark_ablation/families.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from src.core.config import AppConfig
from src.ablation.family_signature import compute_family_signature, compute_family_id
from scripts.utils import setup_logging


logger = setup_logging()


def group_configs_by_family(configs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Group configs by detector family.
    
    Args:
        configs_dir: Directory containing config YAML files
        
    Returns:
        Dictionary mapping family_id to family info:
        {
            "family_id": {
                "signature": {...},
                "configs": ["config1.yaml", "config2.yaml", ...]
            },
            ...
        }
    """
    families: Dict[str, Dict[str, Any]] = {}
    
    # Find all config files
    config_files = sorted(configs_dir.glob("*.yaml"))
    logger.info(f"Found {len(config_files)} config files")
    
    for config_path in config_files:
        try:
            # Load config
            config = AppConfig.from_yaml(str(config_path))
            
            # Compute family signature and ID
            signature = compute_family_signature(config)
            family_id = compute_family_id(signature)
            
            # Add to family
            if family_id not in families:
                families[family_id] = {
                    "signature": signature,
                    "configs": [],
                }
            
            families[family_id]["configs"].append(config_path.name)
            logger.debug(f"{config_path.name} -> family {family_id}")
            
        except Exception as e:
            logger.warning(f"Failed to process {config_path.name}: {e}")
            continue
    
    # Sort configs within each family
    for family_id in families:
        families[family_id]["configs"].sort()
    
    return families


def select_representative_config(configs: List[str]) -> str:
    """
    Select representative config for a family.
    
    Prefers medium-strength config if available (e.g., contains "medium"),
    otherwise first config in list.
    
    Args:
        configs: List of config filenames
        
    Returns:
        Selected config filename
    """
    # Prefer medium-strength config
    for config in configs:
        if "medium" in config.lower():
            return config
    
    # Otherwise first config
    return configs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Group watermark configs by detector family"
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        required=True,
        help="Directory containing config YAML files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for families.json",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {args.configs_dir}")
    
    # Group configs
    logger.info("Grouping configs by detector family...")
    families = group_configs_by_family(args.configs_dir)
    
    logger.info(f"Found {len(families)} detector families")
    
    # Select representative configs
    for family_id, family_data in families.items():
        representative = select_representative_config(family_data["configs"])
        family_data["representative_config"] = representative
        logger.info(
            f"Family {family_id}: {len(family_data['configs'])} configs, "
            f"representative: {representative}"
        )
    
    # Save manifest
    output_data = {
        "families": families,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Saved families manifest to: {args.output}")


if __name__ == "__main__":
    main()

