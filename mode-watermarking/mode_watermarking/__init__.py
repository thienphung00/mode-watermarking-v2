"""
Mode Watermarking - Production-Ready Image Watermarking for Diffusion Models

A streamlined watermarking system for latent diffusion models optimized for GCP deployment.
Provides core embedding, detection, and key management capabilities.
"""

__version__ = "0.1.0"
__author__ = "Mode Watermarking Team"

# Core watermarking functionality
from .core.embedding import (
    WatermarkEmbedder,
    MultiTemporalNoiseModifier,
    LateStageNoiseModifier,
    RandomStepNoiseModifier,
)

from .core.detection import (
    WatermarkDetector,
    DetectionResult,
)

# Utilities
from .utils.utils import (
    WatermarkConfig,
    keyed_hash,
    validate_watermark_key,
)

from .utils.key_manager import (
    WatermarkKeyManager,
)

# Essential exports only
__all__ = [
    # Core classes
    "WatermarkEmbedder",
    "WatermarkDetector", 
    "WatermarkKeyManager",
    
    # Embedding techniques
    "MultiTemporalNoiseModifier",
    "LateStageNoiseModifier", 
    "RandomStepNoiseModifier",
    
    # Configuration and utilities
    "WatermarkConfig",
    "DetectionResult",
    "keyed_hash",
    "validate_watermark_key",
]