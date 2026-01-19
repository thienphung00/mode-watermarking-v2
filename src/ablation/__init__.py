"""
Shared modules for watermark ablation experiments.

This package provides reusable components for:
- Detector family grouping
- Dataset generation
- G-value computation
- Latent caching
"""

from .family_signature import compute_family_signature, compute_family_id
from .dataset_generation import generate_ablation_dataset
from .gvalue_compute import compute_g_values_for_family
from .latent_cache import LatentCache

__all__ = [
    "compute_family_signature",
    "compute_family_id",
    "generate_ablation_dataset",
    "compute_g_values_for_family",
    "LatentCache",
]

