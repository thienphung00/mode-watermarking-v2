#!/usr/bin/env python3
"""
Extract COCO captions into separate txt files for train, val, and test splits.

Setup:
- 2014 Data: Used for Training and Validation
- 2017 Data: Used for Testing

Outputs:
- data/coco/prompts_train.txt
- data/coco/prompts_val.txt
- data/coco/prompts_test.txt
"""

import json
from pathlib import Path
import sys


def load_coco_annotations(json_path):
    """Load COCO annotation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_captions(annotations):
    """Extract captions from COCO annotations."""
    # Build image_id to captions mapping
    image_to_captions = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_to_captions:
            image_to_captions[image_id] = []
        image_to_captions[image_id].append(caption)
    
    # Extract one caption per image (first one)
    captions = []
    for image_id in sorted(image_to_captions.keys()):
        captions.append(image_to_captions[image_id][0])
    
    return captions


def save_captions_to_file(captions, output_path):
    """Save captions to a text file, one per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for caption in captions:
            f.write(caption.strip() + '\n')
    print(f"✓ Saved {len(captions)} captions to {output_path}")


def main():
    """Main function to extract COCO captions."""
    # Paths
    base_dir = Path(__file__).parent.parent
    annot_data_dir = base_dir / "data" / "coco" / "annotations"
    output_dir = base_dir / "data" / "coco"
    
    # Look for COCO annotation files in data/coco/annotations directory
    # Typical COCO structure:
    # - train2014: captions_train2014.json
    # - val2014: captions_val2014.json  
    # - train2017: captions_train2017.json
    # - val2017: captions_val2017.json
    
    annot_files = {
        'train2014': annot_data_dir / 'captions_train2014.json',
        'val2014': annot_data_dir / 'captions_val2014.json',
        'train2017': annot_data_dir / 'captions_train2017.json',
        'val2017': annot_data_dir / 'captions_val2017.json',
    }
    
    # Check which files exist
    print("Looking for COCO annotation files...")
    found_files = {}
    for split, path in annot_files.items():
        if path.exists():
            found_files[split] = path
            print(f"  ✓ Found: {path}")
        else:
            print(f"  ✗ Missing: {path}")
    
    if not found_files:
        print("\n✗ No COCO annotation files found!")
        print(f"Expected files in: {annot_data_dir}")
        print("\nPlease download the COCO dataset using:")
        print("  python scripts/download_coco.py")
        sys.exit(1)
    
    # Process captions based on available files
    # Setup: 2014 Data for train/val, 2017 Data for test
    
    # Train: 2014 train data
    if 'train2014' in found_files:
        print("\nExtracting training captions (2014)...")
        train_annots = load_coco_annotations(found_files['train2014'])
        train_captions = extract_captions(train_annots)
        save_captions_to_file(train_captions, output_dir / "prompts_train.txt")
    else:
        print("\n⚠ Warning: train2014 not found, skipping training set")
    
    # Val: 2014 val data
    if 'val2014' in found_files:
        print("\nExtracting validation captions (2014)...")
        val_annots = load_coco_annotations(found_files['val2014'])
        val_captions = extract_captions(val_annots)
        save_captions_to_file(val_captions, output_dir / "prompts_val.txt")
    else:
        print("\n⚠ Warning: val2014 not found, skipping validation set")
    
    # Test: 2017 val data (standard practice to use val2017 as test)
    if 'val2017' in found_files:
        print("\nExtracting test captions (2017)...")
        test_annots = load_coco_annotations(found_files['val2017'])
        test_captions = extract_captions(test_annots)
        save_captions_to_file(test_captions, output_dir / "prompts_test.txt")
    else:
        # Fallback to train2017 if val2017 not available
        if 'train2017' in found_files:
            print("\nExtracting test captions (2017)...")
            test_annots = load_coco_annotations(found_files['train2017'])
            test_captions = extract_captions(test_annots)
            save_captions_to_file(test_captions, output_dir / "prompts_test.txt")
        else:
            print("\n⚠ Warning: 2017 data not found, skipping test set")
    
    print("\n✓ Caption extraction complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

