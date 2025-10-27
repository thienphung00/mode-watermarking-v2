#!/usr/bin/env python3
"""
Main entry point for Mode Watermarking package.

This module provides a simple CLI interface for watermarking operations.
"""

import argparse
import sys
from pathlib import Path

from mode_watermarking import (
    WatermarkEmbedder, WatermarkDetector, WatermarkConfig,
    validate_watermark_key
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Mode Watermarking CLI')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed watermark')
    embed_parser.add_argument('--key', required=True, help='Watermark key')
    embed_parser.add_argument('--model-id', required=True, help='Model ID')
    embed_parser.add_argument('--technique', default='multi_temporal', 
                             choices=['multi_temporal', 'late_stage', 'random_step'],
                             help='Embedding technique')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark')
    detect_parser.add_argument('--key', required=True, help='Watermark key')
    detect_parser.add_argument('--model-id', help='Model ID')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    
    args = parser.parse_args()
    
    if args.command == 'embed':
        print(f"Embedding watermark with technique: {args.technique}")
        print(f"Key: {args.key[:10]}...")
        print(f"Model ID: {args.model_id}")
        
        # Validate key
        try:
            validate_watermark_key(args.key)
            print("✅ Key validation passed")
        except ValueError as e:
            print(f"❌ Key validation failed: {e}")
            return 1
            
    elif args.command == 'detect':
        print(f"Detecting watermark")
        print(f"Key: {args.key[:10]}...")
        if args.model_id:
            print(f"Model ID: {args.model_id}")
            
    elif args.command == 'config':
        config = WatermarkConfig()
        print("Default Configuration:")
        print(f"  Scales: {config.scales}")
        print(f"  Temporal Windows: {config.temporal_windows}")
        print(f"  Spatial Strengths: {config.spatial_strengths}")
        
    else:
        parser.print_help()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
