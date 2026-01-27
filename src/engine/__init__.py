"""Engine components for Torch/Diffusers integration.

Uses lazy imports to avoid loading heavy diffusers dependencies when only
lightweight components (strategies, hooks) are needed.
"""

# Lazy import mapping: name -> (module_name, attribute_name or None for whole module)
_LAZY_IMPORTS = {
    # Pipeline (requires diffusers)
    "create_pipeline": (".pipeline", "create_pipeline"),
    "apply_watermark_hook": (".pipeline", "apply_watermark_hook"),
    "generate_with_watermark": (".pipeline", "generate_with_watermark"),
    "batch_generate_with_watermark": (".pipeline", "batch_generate_with_watermark"),
    "compute_zT_hash_batch": (".pipeline", "compute_zT_hash_batch"),
    "prepare_latents_with_hashes": (".pipeline", "prepare_latents_with_hashes"),
    "prepare_initial_latents": (".pipeline", "prepare_initial_latents"),
    "compute_zT_hash": (".pipeline", "compute_zT_hash"),
    # Hooks (lightweight)
    "WatermarkHook": (".hooks", "WatermarkHook"),
    "TimestepMapper": (".hooks", "TimestepMapper"),
    # Trainer (may require diffusers)
    "DetectorTrainer": (".trainer", "DetectorTrainer"),
    # Strategy factory (lightweight)
    "create_strategy_from_config": (".strategy_factory", "create_strategy_from_config"),
    "create_per_sample_strategy": (".strategy_factory", "create_per_sample_strategy"),
    # Sampling utilities (requires diffusers)
    "get_text_embeddings": (".sampling_utils", "get_text_embeddings"),
    "custom_ddim_sample": (".sampling_utils", "custom_ddim_sample"),
    "extract_latents_from_pipeline_result": (".sampling_utils", "extract_latents_from_pipeline_result"),
    "ddim_invert": (".sampling_utils", "ddim_invert"),
}


def __getattr__(name: str):
    """Lazy import for engine components."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy imports."""
    return list(_LAZY_IMPORTS.keys()) + ["__all__"]


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

