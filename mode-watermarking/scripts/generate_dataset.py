#!/usr/bin/env python3
"""
Dataset generation script.

Python CLI for generating watermarked/unwatermarked datasets.
This script can be run directly or called from shell scripts.

Usage:
    python scripts/generate_dataset.py --mode both --prompts-file data/coco/prompts_train.txt
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run CLI
from src.cli.generate import main

if __name__ == "__main__":
    main()

