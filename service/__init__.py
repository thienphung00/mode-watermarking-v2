"""
Watermark Service Package.

This package contains a production-shaped GPU-backed watermarking system.

Components:
- service.api: Public API service (CPU)
- service.gpu: GPU worker (private, GPU-backed)

Usage:
    # Start API service
    python -m service.api.main
    
    # Start GPU worker
    python -m service.gpu.main
"""

__version__ = "1.0.0"
