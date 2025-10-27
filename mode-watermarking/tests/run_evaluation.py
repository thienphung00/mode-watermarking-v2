#!/usr/bin/env python3
"""
Simple test runner for embedding method evaluation.

This script provides a streamlined interface for running embedding evaluations
with different configurations and generating reports.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_embedding_evaluation import (
    EvaluationConfig, EmbeddingMethodEvaluator, run_comprehensive_evaluation
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Evaluate watermark embedding methods')
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=1000,
        help='Number of test samples (default: 1000)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device to use for evaluation (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation with minimal samples'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Adjust configuration for quick mode
    if args.quick:
        args.samples = 100
        args.batch_size = 16
        logger.info("Running in quick mode with reduced samples")
    
    # Create configuration
    config = EvaluationConfig(
        num_samples=args.samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=Path(args.output_dir)
    )
    
    logger.info(f"Starting evaluation with {config.num_samples} samples")
    logger.info(f"Output directory: {config.output_dir}")
    
    try:
        # Run comprehensive evaluation
        report = run_comprehensive_evaluation()
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {config.output_dir}")
        
        # Print key results
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        if 'summary' in report:
            for method, stats in report['summary'].items():
                print(f"\n{method.upper()}:")
                print(f"  PSNR: {stats['avg_psnr']:.2f} dB")
                print(f"  SSIM: {stats['avg_ssim']:.3f}")
                print(f"  LPIPS: {stats['avg_lpips']:.4f}")
                print(f"  FID: {stats['avg_fid']:.2f}")
                print(f"  CLIP Similarity: {stats['avg_clip_similarity']:.3f}")
        
        if 'recommendations' in report:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
