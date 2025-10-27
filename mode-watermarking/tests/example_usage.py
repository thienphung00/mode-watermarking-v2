#!/usr/bin/env python3
"""
Example usage of the embedding evaluation system.

This script demonstrates how to use the evaluation framework
to compare different embedding methods.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_embedding_evaluation import (
    EvaluationConfig, EmbeddingMethodEvaluator, QualityMetrics
)
from mode_watermarking import WatermarkEmbedder, WatermarkConfig

def simple_evaluation_example():
    """Simple example of evaluating embedding methods."""
    print("🔬 Embedding Method Evaluation Example")
    print("=" * 50)
    
    # Create a small test configuration
    config = EvaluationConfig(
        num_samples=100,  # Small for demo
        batch_size=16,
        output_dir=Path("example_results")
    )
    
    # Initialize evaluator
    evaluator = EmbeddingMethodEvaluator(config)
    
    # Generate test data
    print("📊 Generating test dataset...")
    original_images, _ = evaluator.generate_test_dataset()
    
    # Test different embedding methods
    methods = {
        'multi_temporal': {
            'technique': 'multi_temporal',
            'noise_scale': 0.01
        },
        'late_stage': {
            'technique': 'late_stage', 
            'noise_scale': 0.015
        },
        'random_step': {
            'technique': 'random_step',
            'noise_scale': 0.012
        }
    }
    
    results = {}
    
    for method_name, method_config in methods.items():
        print(f"\n🔍 Evaluating {method_name}...")
        
        # Generate watermarked images
        watermarked_images = evaluator._generate_watermarked_images(original_images, method_config)
        
        # Evaluate quality
        result = evaluator.evaluate_embedding_method(
            method_name,
            method_config,
            original_images,
            watermarked_images
        )
        
        results[method_name] = result
        
        # Print results
        metrics = result.metrics
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.3f}")
        print(f"  LPIPS: {metrics.lpips:.4f}")
        print(f"  FID: {metrics.fid:.2f}")
        print(f"  CLIP Similarity: {metrics.clip_similarity:.3f}")
        print(f"  Processing Time: {result.performance_stats['processing_time']:.2f}s")
    
    # Compare methods
    print("\n📈 Method Comparison:")
    print("-" * 30)
    
    best_psnr = max(results.items(), key=lambda x: x[1].metrics.psnr)
    best_ssim = max(results.items(), key=lambda x: x[1].metrics.ssim)
    best_speed = min(results.items(), key=lambda x: x[1].performance_stats['processing_time'])
    
    print(f"Best PSNR: {best_psnr[0]} ({best_psnr[1].metrics.psnr:.2f} dB)")
    print(f"Best SSIM: {best_ssim[0]} ({best_ssim[1].metrics.ssim:.3f})")
    print(f"Fastest: {best_speed[0]} ({best_speed[1].performance_stats['processing_time']:.2f}s)")
    
    # Generate report
    print("\n📋 Generating report...")
    report = evaluator.generate_report()
    
    print(f"✅ Evaluation complete! Results saved to: {config.output_dir}")
    return results

def parameter_sweep_example():
    """Example of parameter sweeping for robustness-quality trade-off."""
    print("\n🔬 Parameter Sweep Example")
    print("=" * 50)
    
    config = EvaluationConfig(
        num_samples=200,
        batch_size=16,
        output_dir=Path("sweep_results")
    )
    
    evaluator = EmbeddingMethodEvaluator(config)
    
    # Define parameter ranges
    parameter_ranges = {
        'noise_scale': [0.005, 0.01, 0.02, 0.05]
    }
    
    print("🔄 Running parameter sweep...")
    sweep_results = evaluator.parameter_sweep_evaluation(
        'multi_temporal',
        {'technique': 'multi_temporal'},
        parameter_ranges
    )
    
    # Analyze results
    print("\n📊 Parameter Sweep Results:")
    print("-" * 40)
    
    for result in sweep_results:
        noise_scale = result.config.get('noise_scale', 'unknown')
        metrics = result.metrics
        print(f"Noise Scale {noise_scale}:")
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.3f}")
        print(f"  FID: {metrics.fid:.2f}")
        print()
    
    return sweep_results

if __name__ == "__main__":
    print("🚀 Starting Embedding Evaluation Examples")
    print("=" * 60)
    
    try:
        # Run simple evaluation
        results = simple_evaluation_example()
        
        # Run parameter sweep
        sweep_results = parameter_sweep_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated result directories")
        print("2. Modify parameters in the examples")
        print("3. Run full evaluation with: python run_evaluation.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
