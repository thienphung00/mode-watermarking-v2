"""Engine components for Torch/Diffusers integration."""

from .pipeline import (
    create_pipeline,
    apply_watermark_hook,
    generate_with_watermark,
    batch_generate_with_watermark,
    compute_zT_hash_batch,
    prepare_latents_with_hashes,
    prepare_initial_latents,
    compute_zT_hash,
)
from .hooks import WatermarkHook, TimestepMapper
from .trainer import DetectorTrainer
from .strategy_factory import create_strategy_from_config, create_per_sample_strategy
from .sampling_utils import (
    get_text_embeddings,
    custom_ddim_sample,
    extract_latents_from_pipeline_result,
    ddim_invert,
)

__all__ = [
    # Pipeline creation
    "create_pipeline",
    "apply_watermark_hook",
    # Generation with zT_hash integration
    "generate_with_watermark",
    "batch_generate_with_watermark",
    "compute_zT_hash_batch",
    "prepare_latents_with_hashes",
    # Latent utilities
    "prepare_initial_latents",
    "compute_zT_hash",
    # Hooks
    "WatermarkHook",
    "TimestepMapper",
    # Strategy
    "create_strategy_from_config",
    "create_per_sample_strategy",
    # Training
    "DetectorTrainer",
    # Sampling utilities
    "get_text_embeddings",
    "custom_ddim_sample",
    "extract_latents_from_pipeline_result",
    "ddim_invert",
]

