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

# Engine exports
from .engine import (
    create_pipeline,
    generate_with_watermark,
    apply_watermark_hook,
)

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
    # Engine
    "create_pipeline",
    "generate_with_watermark",
    "apply_watermark_hook",
]
