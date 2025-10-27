# GCP Deployment Guide

## Quick Start

1. **Upload the codebase to your GCP instance:**
   ```bash
   # Copy the entire mode-watermarking directory to your GCP instance
   scp -r mode-watermarking/ user@your-gcp-instance:~/
   ```

2. **Install dependencies:**
   ```bash
   # On your GCP instance
   cd mode-watermarking
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import mode_watermarking; print('✅ Installation successful!')"
   ```

## Alternative: Package Installation

1. **Build package locally:**
   ```bash
   ./deploy_gcp.sh
   ```

2. **Upload and install:**
   ```bash
   scp dist/*.whl user@your-gcp-instance:~/
   # On GCP instance:
   pip install *.whl
   ```

## GCP-Specific Considerations

### GPU Support
- Install CUDA-enabled PyTorch: `pip install torch[cuda] torchvision[cuda]`
- Ensure your GCP instance has GPU access enabled

### Memory Requirements
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM for large models
- GPU memory: 8GB+ VRAM recommended

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU support
pip install torch[cuda] torchvision[cuda]
```

## Testing on GCP

```bash
# Quick test
python -c "from mode_watermarking.core.embedding import WatermarkEmbedder; print('✅ Core modules working')"

# Run evaluation tests (if needed)
cd tests
pip install -r requirements_test.txt
python run_evaluation.py --quick
```

## Troubleshooting

- **CUDA issues**: Ensure GPU drivers are installed on GCP instance
- **Memory errors**: Reduce batch sizes or use CPU-only mode
- **Import errors**: Verify all dependencies are installed correctly
