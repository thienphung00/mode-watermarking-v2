#!/usr/bin/env python3
"""
Download COCO image-caption dataset using kagglehub.
"""
import kagglehub
from pathlib import Path
import sys


def main():
    print("Downloading COCO image caption dataset...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("nikhil7280/coco-image-caption")
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # Optionally show what's in the directory
        dataset_path = Path(path)
        if dataset_path.exists():
            print(f"\nContents of {dataset_path}:")
            for item in list(dataset_path.iterdir())[:10]:  # Show first 10 items
                print(f"  - {item.name}")
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nMake sure you have kagglehub installed:")
        print("  pip install kagglehub")
        sys.exit(1)


if __name__ == "__main__":
    main()

