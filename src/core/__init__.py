"""Core abstractions and type definitions for mode watermarking."""

from .config import (
    AppConfig,
    ModeType,
    PRFConfig,
    WatermarkedConfig,
    UnwatermarkedConfig,
    DiffusionConfig,
    TrainingConfig,
    AlgorithmParams,
    KeySettings,
)
from .interfaces import (
    WatermarkStrategy,
    NullStrategy,
    LatentInjectionStrategy,
)
from .context import ExperimentContext
from .metadata_schema import (
    MinimalWatermarkMetadata,
    StandardWatermarkMetadata,
    ExtendedWatermarkMetadata,
    SecureMetadataValidator,
    create_metadata_from_generation_result,
    serialize_metadata,
    deserialize_metadata,
)

__all__ = [
    # Config
    "AppConfig",
    "ModeType",
    "PRFConfig",
    "WatermarkedConfig",
    "UnwatermarkedConfig",
    "DiffusionConfig",
    "TrainingConfig",
    "AlgorithmParams",
    "KeySettings",
    # Interfaces
    "WatermarkStrategy",
    "NullStrategy",
    "LatentInjectionStrategy",
    # Context
    "ExperimentContext",
    # Metadata Schema (Security-Safe)
    "MinimalWatermarkMetadata",
    "StandardWatermarkMetadata",
    "ExtendedWatermarkMetadata",
    "SecureMetadataValidator",
    "create_metadata_from_generation_result",
    "serialize_metadata",
    "deserialize_metadata",
]

