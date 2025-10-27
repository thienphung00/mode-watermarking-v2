# Embedding Method Evaluation Tests

This directory contains comprehensive tests for evaluating different watermark embedding methods with focus on image quality metrics, security, and performance.

## Files

- `test_embedding_evaluation.py` - Main evaluation suite with comprehensive metrics
- `run_evaluation.py` - Simple test runner with command-line interface
- `requirements_test.txt` - Test dependencies
- `README.md` - This file

## Metrics Implemented

### Image Quality Metrics
- **PSNR** (Peak Signal-to-Noise Ratio) - Pixel-level fidelity
- **SSIM** (Structural Similarity Index) - Perceptual structure preservation
- **LPIPS** (Learned Perceptual Image Patch Similarity) - Human-perceived differences
- **FID** (Fréchet Inception Distance) - Distributional similarity
- **CLIP Similarity** - High-level semantic preservation

### Performance Metrics
- Processing time per image
- Memory usage
- Throughput (images per second)

## Usage

### Quick Start
```bash
# Install test dependencies
pip install -r requirements_test.txt

# Run quick evaluation
python run_evaluation.py --quick --verbose

# Run full evaluation
python run_evaluation.py --samples 10000 --batch-size 64
```

### Command Line Options
```bash
python run_evaluation.py --help
```

Options:
- `--samples N` - Number of test samples (default: 1000)
- `--batch-size N` - Batch size for processing (default: 32)
- `--output-dir DIR` - Output directory for results
- `--device DEVICE` - Device to use (auto, cuda, cpu, mps)
- `--verbose` - Enable verbose logging
- `--quick` - Run with minimal samples for testing

### Programmatic Usage
```python
from tests.test_embedding_evaluation import (
    EvaluationConfig, EmbeddingMethodEvaluator
)

# Create configuration
config = EvaluationConfig(
    num_samples=1000,
    batch_size=32,
    output_dir=Path("results")
)

# Run evaluation
evaluator = EmbeddingMethodEvaluator(config)
report = evaluator.generate_report()
```

## Evaluation Methods

### 1. Multi-Temporal Embedding
- Embeds watermarks throughout the sampling process
- Uses temporal windows with different weights
- Balanced approach for robust watermarking

### 2. Late-Stage Embedding
- Embeds watermarks only in final timesteps
- Minimal interference with early generation
- Stronger signal in final stages

### 3. Random-Step Embedding
- Embeds watermarks at random timesteps
- Adds unpredictability for security
- Uses seeded random generation

## Parameter Sweeping

The evaluation includes parameter sweeping to analyze robustness-quality trade-offs:

```python
parameter_ranges = {
    'noise_scale': [0.005, 0.01, 0.02, 0.05, 0.1],
    'strength_multiplier': [0.5, 1.0, 1.5, 2.0],
    'embedding_probability': [0.1, 0.3, 0.5, 0.7]
}
```

## Output

Results are saved to the specified output directory:
- `evaluation_report.json` - Comprehensive results
- `evaluation.log` - Detailed logging
- Plots and visualizations (if enabled)

## Quality Thresholds

Default quality thresholds:
- PSNR ≥ 20.0 dB
- SSIM ≥ 0.8
- FID ≤ 10.0
- CLIP Similarity ≥ 0.9

## Statistical Analysis

The evaluation includes:
- Confidence intervals (95% default)
- Statistical significance testing
- Distribution analysis
- Performance benchmarking

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA support (optional)
- 8GB+ RAM recommended
- GPU recommended for large evaluations

## Notes

- The current implementation uses synthetic data for demonstration
- For production use, integrate with actual diffusion models
- Adjust quality thresholds based on your requirements
- Consider computational resources for large-scale evaluations
