# Mode Watermarking Package V2

__version__ = "2.0.0"

# Core exports
from .core import (
    AppConfig,
    ModeType,
    WatermarkedConfig,
    UnwatermarkedConfig,
    DiffusionConfig,
    TrainingConfig,
    AlgorithmParams,
    KeySettings,
    WatermarkStrategy,
    NullStrategy,
    LatentInjectionStrategy,
    ExperimentContext,
)

# Algorithm exports
from .algorithms import (
    GFieldGenerator,
    MaskGenerator,
    AlphaScheduler,
)

# Engine exports are lazy-loaded to avoid importing heavy ML dependencies
# when only core/algorithm components are needed (e.g., in unit tests)
def __getattr__(name):
    """Lazy import for engine components."""
    _engine_exports = {
        "create_pipeline",
        "generate_with_watermark", 
        "apply_watermark_hook",
    }
    if name in _engine_exports:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Core
    "AppConfig",
    "ModeType", 
    "WatermarkedConfig",
    "UnwatermarkedConfig",
    "DiffusionConfig",
    "TrainingConfig",
    "AlgorithmParams",
    "KeySettings",
    "WatermarkStrategy",
    "NullStrategy",
    "LatentInjectionStrategy",
    "ExperimentContext",
    # Algorithms
    "GFieldGenerator",
    "MaskGenerator",
    "AlphaScheduler",
    # Engine (lazy-loaded)
    "create_pipeline",
    "generate_with_watermark",
    "apply_watermark_hook",
]
