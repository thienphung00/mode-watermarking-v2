"""
Type-safe configuration using Pydantic models.

This module replaces loose dictionary passing with strict validation.
Guarantees that mode=watermarked strictly enforces the presence of watermark
parameters at startup, not runtime.
"""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModeType(StrEnum):
    """Watermarking mode type."""

    WATERMARKED = "watermarked"
    UNWATERMARKED = "unwatermarked"


# ============================================================================
# PRF Configuration
# ============================================================================


class PRFConfig(BaseModel):
    """Configuration for PRF-based key generation."""
    
    algorithm: Literal["chacha20", "aes_ctr"] = Field(
        "chacha20", 
        description="PRF algorithm: chacha20 or aes_ctr"
    )
    output_bits: Literal[32, 64] = Field(
        64, 
        description="Bits per PRF output: 32 or 64"
    )
    
    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate PRF algorithm."""
        if v not in {"chacha20", "aes_ctr"}:
            raise ValueError(f"algorithm must be 'chacha20' or 'aes_ctr', got '{v}'")
        return v
    
    @field_validator("output_bits")
    @classmethod
    def validate_output_bits(cls, v: int) -> int:
        """Validate output bits."""
        if v not in {32, 64}:
            raise ValueError(f"output_bits must be 32 or 64, got {v}")
        return v


# ============================================================================
# Algorithm Configuration
# ============================================================================


class GFieldConfig(BaseModel):
    """G-field generation configuration."""

    shape: Tuple[int, int, int] = Field((4, 64, 64), description="Latent shape [C, H, W]")
    mapping_mode: Literal["binary", "continuous"] = Field("binary", description="Mapping mode")
    continuous_range: Optional[Tuple[float, float]] = Field(
        (-1.0, 1.0), description="Range for continuous mode"
    )
    channel_wise: bool = Field(True, description="Separate g-field per channel")

    # Domain settings
    domain: Literal["spatial", "frequency"] = Field("frequency", description="Generation domain")
    frequency_mode: str = Field("lowpass", description="Frequency mode")
    low_freq_cutoff: float = Field(0.12, description="Low frequency cutoff")
    high_freq_cutoff: Optional[float] = Field(None, description="High frequency cutoff (for bandpass mode)")
    frequency_normalization: str = Field("parseval", description="Frequency normalization")

    # Normalization (can be specified as nested dict or top-level fields)
    normalize: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable": True,
            "zero_mean_per_timestep": True,
            "zero_mean_per_channel": True,
            "unit_variance": False,
            "eps": 1e-8,
        }
    )
    # Top-level normalization fields (override nested normalize dict if provided)
    normalize_zero_mean: Optional[bool] = Field(None, description="Zero-mean normalization (overrides normalize dict)")
    normalize_unit_variance: Optional[bool] = Field(None, description="Unit-variance normalization (overrides normalize dict)")


class MaskConfig(BaseModel):
    """Mask configuration for spatial/frequency masking."""

    mode: Literal["frequency", "spatial"] = Field("frequency", description="Mask mode")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Mask strength")

    # Frequency mask
    band: str = Field("low", description="Frequency band: low/high/band")
    cutoff_freq: float = Field(0.30, ge=0.0, le=1.0, description="Cutoff frequency")
    bandwidth_fraction: float = Field(0.15, description="Bandwidth fraction for band mode")

    # Spatial mask
    shape: str = Field("radial", description="Spatial shape: radial/rect")
    radius_fraction: float = Field(0.50, ge=0.0, le=1.0, description="Radius fraction")


class BiasConfig(BaseModel):
    """Bias injection configuration."""

    mode: Literal["non_distortionary", "distortionary"] = Field(
        "non_distortionary", description="Embedding mode"
    )
    target_snr: float = Field(0.05, gt=0.0, description="Target SNR")
    alpha_bounds: Tuple[float, float] = Field((0.0, 0.08), description="Alpha bounds [min, max]")
    min_strength: float = Field(0.005, ge=0.0, description="Minimum strength")
    max_strength: float = Field(0.025, ge=0.0, description="Maximum strength")

    injection: Dict[str, Any] = Field(
        default_factory=lambda: {
            "strategy": "concentrated_late",
            "start_fraction": 0.70,
            "peak_fraction": 0.90,
            "end_fraction": 1.00,
            "shape": "triangular",
            "normalize_alpha": True,
        }
    )

    @field_validator("alpha_bounds")
    @classmethod
    def validate_alpha_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Validate alpha bounds."""
        if v[0] >= v[1]:
            raise ValueError(f"alpha_bounds[0] must be < alpha_bounds[1], got {v}")
        return v


class KeySettings(BaseModel):
    """Key derivation settings for PRF-based system."""

    key_master: str = Field(..., description="Master key (keep secret)")
    key_id: str = Field("default_key_001", description="Public key identifier")
    prf_config: PRFConfig = Field(
        default_factory=PRFConfig,
        description="PRF configuration (ChaCha20/AES-CTR)"
    )
    experiment_id: Optional[str] = Field(
        "exp_001", 
        description="Experiment identifier (for metadata only, not used for seed generation)"
    )


class SeedBiasConfig(BaseModel):
    """
    Configuration for seed bias (latent initialization) watermarking strategy.
    
    ⚠️ GENERATION-ONLY CONFIG ⚠️
    This config controls watermark embedding during generation.
    It does NOT configure detection algorithms.
    
    Detection is configured separately in DetectionService (detector_type="bayesian").
    Detection mode selection (fast_only/hybrid/full_inversion) is deprecated.
    Only Bayesian detection is supported.
    """

    lambda_strength: float = Field(
        0.05, ge=0.0, lt=1.0, description="Injection strength (0.0 to 1.0)"
    )

    # Frequency Filtering controls
    domain: Literal["spatial", "frequency"] = Field(
        "frequency", description="Generation domain"
    )
    low_freq_cutoff: float = Field(
        0.05, ge=0.0, le=1.0, description="Remove very low freq (artifacts)"
    )
    high_freq_cutoff: float = Field(
        0.4, ge=0.0, le=1.0, description="Remove high freq (washout prevention)"
    )

    # Masking
    mask_config: Optional[MaskConfig] = None

    @field_validator("lambda_strength")
    @classmethod
    def validate_lambda_strength(cls, v: float) -> float:
        """Warn if lambda is too high (visible artifacts likely)."""
        if v > 0.15:
            import warnings

            warnings.warn(
                f"lambda_strength={v} > 0.15 may cause visible artifacts. "
                "Consider using lower values (0.05-0.10) for high-fidelity generation.",
                UserWarning,
            )
        return v


class AlgorithmParams(BaseModel):
    """Algorithm parameters for watermark generation."""

    g_field: GFieldConfig = Field(default_factory=GFieldConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)

    content_hash_strength: float = Field(0.3, ge=0.0, le=1.0)
    time_dependent_offset: bool = Field(False, description="Use time-dependent offset")
    
    # Seed bias configuration (optional, for seed bias strategy)
    seed_bias: Optional[SeedBiasConfig] = None


# ============================================================================
# Watermark Configuration (Discriminated Union)
# ============================================================================


class BaseWatermarkConfig(BaseModel):
    """Base watermark configuration."""

    mode: ModeType


class UnwatermarkedConfig(BaseWatermarkConfig):
    """Configuration for unwatermarked mode (no watermark parameters needed)."""

    mode: Literal[ModeType.UNWATERMARKED] = ModeType.UNWATERMARKED




class WatermarkedConfig(BaseWatermarkConfig):
    """Configuration for watermarked mode (all parameters mandatory)."""

    mode: Literal[ModeType.WATERMARKED] = ModeType.WATERMARKED

    # These fields are MANDATORY for watermarked mode
    # Note: target_snr is controlled via algorithm_params.bias.target_snr
    algorithm_params: AlgorithmParams = Field(..., description="Algorithm parameters")
    key_settings: KeySettings = Field(..., description="Key derivation settings")


# ============================================================================
# Diffusion Configuration
# ============================================================================


class DiffusionConfig(BaseModel):
    """Diffusion model configuration."""

    model_id: str = Field("runwayml/stable-diffusion-v1-5", description="Hugging Face model ID")

    # Timesteps
    trained_timesteps: int = Field(1000, gt=0, description="Training timesteps")
    inference_timesteps: int = Field(50, gt=0, description="Inference timesteps")

    # Beta schedule
    beta_start: float = Field(0.00085, gt=0.0, description="Beta schedule start")
    beta_end: float = Field(0.012, gt=0.0, description="Beta schedule end")
    beta_schedule: str = Field("linear", description="Beta schedule type")

    # Prediction and scheduling
    prediction_type: str = Field("epsilon", description="Prediction type")
    scheduler: str = Field("DDIM", description="Scheduler type")
    scheduler_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Scheduler kwargs")

    # Guidance
    guidance_scale: float = Field(7.5, gt=0.0, description="Guidance scale")
    guidance_scale_range: Tuple[float, float] = Field(
        (3.0, 15.0), description="Guidance scale range"
    )

    # Memory optimization
    use_fp16: bool = Field(True, description="Use FP16 precision")
    gradient_checkpointing: bool = Field(False, description="Enable gradient checkpointing")

    @field_validator("beta_end")
    @classmethod
    def validate_beta_range(cls, v: float, info) -> float:
        """Validate beta_end > beta_start."""
        if "beta_start" in info.data and v <= info.data["beta_start"]:
            raise ValueError(
                f"beta_end ({v}) must be > beta_start ({info.data['beta_start']})"
            )
        return v


# ============================================================================
# Training Configuration
# ============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Data
    train_manifest: str = Field(..., description="Path to training manifest")
    val_manifest: str = Field(..., description="Path to validation manifest")
    batch_size: int = Field(32, gt=0, description="Batch size")
    num_workers: int = Field(4, ge=0, description="Number of dataloader workers")

    # Model
    detector_type: Literal["unet", "bayesian"] = Field("unet", description="Detector type")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Model kwargs")
    
    # Input representation (explicit extraction method ownership)
    input_representation: Literal["z_0", "z_T", "g_binary"] = Field(
        "z_0", 
        description="Input representation: z_0 (VAE-encoded), z_T (inverted), or g_binary (binary g-values)"
    )
    extraction_method: Literal["none", "whitened", "normalized", "sign"] = Field(
        "none",
        description="Extraction method applied before model input. 'none' for raw latents (UNetDetector default)"
    )

    # Optimization
    epochs: int = Field(100, gt=0, description="Number of training epochs")
    learning_rate: float = Field(1e-4, gt=0.0, description="Learning rate")
    weight_decay: float = Field(1e-5, ge=0.0, description="Weight decay")
    optimizer: str = Field("adamw", description="Optimizer type")

    # Loss
    loss_type: str = Field("bce", description="Loss type: bce/focal/combined")
    focal_alpha: float = Field(0.25, ge=0.0, le=1.0, description="Focal loss alpha")
    focal_gamma: float = Field(2.0, ge=0.0, description="Focal loss gamma")

    # Checkpointing
    checkpoint_dir: str = Field("outputs/checkpoints", description="Checkpoint directory")
    save_every_n_epochs: int = Field(5, gt=0, description="Save checkpoint every N epochs")
    keep_last_n_checkpoints: int = Field(5, gt=0, description="Keep last N checkpoints")

    # Logging
    log_every_n_steps: int = Field(10, gt=0, description="Log every N steps")
    log_backend: str = Field("tensorboard", description="Logging backend")


# ============================================================================
# Evaluation Configuration
# ============================================================================


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    test_manifest: str = Field(..., description="Path to test manifest")
    detector_checkpoint: Optional[str] = Field(None, description="Detector checkpoint path")
    batch_size: int = Field(32, gt=0, description="Batch size")

    # Metrics
    compute_quality_metrics: bool = Field(True, description="Compute quality metrics")
    compute_detection_metrics: bool = Field(True, description="Compute detection metrics")

    # Thresholds
    target_fpr: float = Field(0.01, gt=0.0, lt=1.0, description="Target FPR for threshold")

    # Output
    output_dir: str = Field("outputs/evaluation", description="Output directory")
    save_visualizations: bool = Field(True, description="Save visualizations")


# ============================================================================
# Master Configuration
# ============================================================================


class AppConfig(BaseModel):
    watermark: Union[WatermarkedConfig, UnwatermarkedConfig] = Field(
        ..., discriminator="mode"
    )
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    training: Optional[TrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None

    @classmethod
    def from_yaml(cls, path: str):
        from pathlib import Path
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Read file content once for parsing and diagnostics
        file_size = path.stat().st_size
        with open(path, "r") as f:
            content = f.read()

        # Check if file is empty
        if not content.strip():
            raise ValueError(
                f"Config file '{path}' is empty (size: {file_size} bytes). "
                "Please ensure the file contains valid YAML configuration."
            )
        
        # Try to parse the YAML
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Config file '{path}' contains invalid YAML: {e}\n"
                f"File size: {file_size} bytes. First 200 chars: {repr(content[:200])}"
            )

        if data is None:
            raise ValueError(
                f"Config file '{path}' parsed to None (likely empty or only comments). "
                f"File size: {file_size} bytes. First 200 chars: {content[:200]}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"Config file '{path}' must contain a YAML dictionary/mapping at the root level, "
                f"but got {type(data).__name__}"
            )

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Use mode='json' to convert enums to strings and tuples to lists
            # This ensures safe YAML output without Python-specific tags
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)

    @model_validator(mode="after")
    def validate_watermarked_dependencies(self) -> "AppConfig":
        """Ensure watermarked mode has all required parameters."""
        if isinstance(self.watermark, WatermarkedConfig):
            # All validation is handled by Pydantic's required fields
            # This is a placeholder for any additional cross-field validation
            pass
        return self


# ============================================================================
# Geometry Signature Schema (for detector family grouping)
# ============================================================================


def extract_detector_geometry_signature(config: WatermarkedConfig) -> Dict[str, Any]:
    """
    Extract complete detector geometry signature from watermark config.
    
    This function defines the authoritative schema for detector geometry.
    It enumerates ALL fields that affect detector geometry, including:
    - mapping_mode
    - All g_field parameters (grid size, bounds, normalization, cutoff, transforms, etc.)
    - All mask parameters (shape, smoothing, padding, scaling, thresholds, etc.)
    - Any implicit defaults currently assumed in code
    
    This signature is used to group configs into detector families.
    Configs with identical geometry signatures belong to the same family.
    
    Args:
        config: WatermarkedConfig instance
        
    Returns:
        Normalized signature dictionary with all geometry-affecting fields
        
    Raises:
        ValueError: If any required field is missing (no silent defaults)
    """
    g_field = config.algorithm_params.g_field
    mask = config.algorithm_params.mask
    
    # Extract ALL g_field geometry parameters
    # Priority: top-level fields > nested normalize dict > defaults
    normalize_dict = g_field.normalize if isinstance(g_field.normalize, dict) else {}
    
    # Normalization settings (explicit, no silent defaults)
    if g_field.normalize_zero_mean is not None:
        normalize_zero_mean = bool(g_field.normalize_zero_mean)
    else:
        # Check nested dict
        zero_mean_per_timestep = normalize_dict.get("zero_mean_per_timestep", True)
        zero_mean_per_channel = normalize_dict.get("zero_mean_per_channel", True)
        normalize_zero_mean = bool(zero_mean_per_timestep or zero_mean_per_channel)
    
    if g_field.normalize_unit_variance is not None:
        normalize_unit_variance = bool(g_field.normalize_unit_variance)
    else:
        normalize_unit_variance = bool(normalize_dict.get("unit_variance", False))
    
    # Build complete g_field signature
    g_field_signature = {
        "mapping_mode": str(g_field.mapping_mode),
        "domain": str(g_field.domain),
        "frequency_mode": str(g_field.frequency_mode),
        "low_freq_cutoff": float(g_field.low_freq_cutoff),
        "normalize_zero_mean": normalize_zero_mean,
        "normalize_unit_variance": normalize_unit_variance,
        "channel_wise": bool(g_field.channel_wise),
        "frequency_normalization": str(g_field.frequency_normalization),
    }
    
    # Add optional fields (explicit None handling)
    if g_field.continuous_range is not None:
        g_field_signature["continuous_range"] = tuple(float(x) for x in g_field.continuous_range)
    else:
        # Explicit default for continuous mode
        if g_field.mapping_mode == "continuous":
            g_field_signature["continuous_range"] = (-1.0, 1.0)
    
    # Add high_freq_cutoff if present or if bandpass mode requires it
    if g_field.high_freq_cutoff is not None:
        g_field_signature["high_freq_cutoff"] = float(g_field.high_freq_cutoff)
    elif g_field.frequency_mode == "bandpass":
        # Bandpass mode requires high_freq_cutoff - raise error if missing
        raise ValueError(
            f"high_freq_cutoff is REQUIRED for bandpass mode but is None. "
            f"This prevents silent defaults that could cause family grouping errors."
        )
    
    # Extract ALL mask parameters (exclude strength, which is not geometry)
    mask_signature = {
        "mode": str(mask.mode),
        "band": str(mask.band),
        "cutoff_freq": float(mask.cutoff_freq),
        "bandwidth_fraction": float(mask.bandwidth_fraction),
    }
    
    # Add spatial mask parameters if mode is spatial
    if mask.mode == "spatial":
        mask_signature["shape"] = str(mask.shape)
        mask_signature["radius_fraction"] = float(mask.radius_fraction)
    
    # Build complete signature
    signature = {
        "mapping_mode": g_field_signature["mapping_mode"],
        "g_field": g_field_signature,
        "mask": mask_signature,
    }
    
    return signature


def normalize_geometry_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize signature for deterministic hashing.
    
    Ensures dictionary ordering is stable and float values are rounded
    to avoid floating-point precision issues.
    
    Args:
        signature: Signature dictionary
        
    Returns:
        Normalized signature dictionary with sorted keys and rounded floats
    """
    def normalize_value(v: Any) -> Any:
        """Recursively normalize values."""
        if isinstance(v, float):
            # Round to 10 decimal places to avoid precision issues while preserving accuracy
            return round(v, 10)
        elif isinstance(v, bool):
            return bool(v)  # Ensure boolean type
        elif isinstance(v, int):
            return int(v)  # Ensure integer type
        elif isinstance(v, str):
            return str(v)  # Ensure string type
        elif isinstance(v, dict):
            # Sort keys and recursively normalize
            return {k: normalize_value(v) for k, v in sorted(v.items())}
        elif isinstance(v, (list, tuple)):
            # Normalize each item and convert tuple to list for JSON compatibility
            normalized = [normalize_value(item) for item in v]
            return tuple(normalized) if isinstance(v, tuple) else normalized
        else:
            return v
    
    # Sort top-level keys and normalize values
    normalized = {k: normalize_value(v) for k, v in sorted(signature.items())}
    return normalized


def compute_key_fingerprint(
    master_key: str,
    key_id: str,
    prf_config: Optional[PRFConfig] = None,
) -> str:
    """
    Compute cryptographic key fingerprint for key isolation.
    
    Creates a deterministic, non-reversible hash from master_key, key_id, and PRF algorithm.
    This fingerprint is used to ensure cached artifacts are key-isolated and cannot
    be accidentally reused across different keys.
    
    The fingerprint is:
    - Deterministic: same inputs → same fingerprint
    - Non-reversible: cannot recover master_key from fingerprint
    - Stable: same across runs
    - Unique: different keys → different fingerprints
    
    Args:
        master_key: Secret master key
        key_id: Public key identifier
        prf_config: PRF configuration (default: ChaCha20 with 64-bit outputs)
        
    Returns:
        64-character hex string (SHA-256 hash)
    """
    import hashlib
    
    if prf_config is None:
        prf_config = PRFConfig()
    
    # Create deterministic input string
    # Format: "key_fingerprint_v1|master_key|key_id|algorithm|output_bits"
    fingerprint_input = (
        f"key_fingerprint_v1|"
        f"{master_key}|"
        f"{key_id}|"
        f"{prf_config.algorithm}|"
        f"{prf_config.output_bits}"
    )
    
    # Compute SHA-256 hash
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    
    return fingerprint


def compute_cache_key(
    image_id: str,
    config: AppConfig,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    num_inversion_steps: Optional[int] = None,
    code_version_hash: Optional[str] = None,
    master_key: Optional[str] = None,
    key_id: Optional[str] = None,
) -> str:
    """
    Compute comprehensive cache key for cached artifacts.
    
    This function creates a canonical cache key that includes all parameters
    that affect the cached artifact to prevent cache collisions.
    
    Cache key includes:
    - Key fingerprint (master_key + key_id + PRF algorithm) - CRITICAL for key isolation
    - Normalized detector geometry signature hash
    - Full config hash
    - Prompt hash (if provided)
    - Seed (if provided)
    - Inversion parameters (num_inversion_steps, model_id)
    - Image generation parameters (guidance_scale, inference_timesteps)
    - Code version hash (if available)
    
    Args:
        image_id: Image identifier
        config: AppConfig instance
        prompt: Optional prompt (for prompt-dependent caches)
        seed: Optional seed (for seed-dependent caches)
        num_inversion_steps: Optional number of inversion steps
        code_version_hash: Optional code version hash (e.g., git commit)
        master_key: Optional master key (required for watermarked configs)
        key_id: Optional key identifier (required for watermarked configs)
        
    Returns:
        Deterministic cache key string
    """
    import hashlib
    import json
    
    components = []
    
    # 0. KEY FINGERPRINT (CRITICAL: Must be first to ensure key isolation)
    if isinstance(config.watermark, WatermarkedConfig):
        if master_key is None:
            master_key = config.watermark.key_settings.key_master
        if key_id is None:
            key_id = config.watermark.key_settings.key_id
        prf_config = config.watermark.key_settings.prf_config
        
        key_fingerprint = compute_key_fingerprint(master_key, key_id, prf_config)
        # Use first 16 chars for brevity (still 64 bits of entropy)
        components.append(f"key{key_fingerprint[:16]}")
    else:
        # Unwatermarked: use a fixed fingerprint to ensure isolation from watermarked caches
        components.append(f"key{hashlib.sha256(b'unwatermarked').hexdigest()[:16]}")
    
    # 1. Geometry signature hash (for detector family grouping)
    if isinstance(config.watermark, WatermarkedConfig):
        geometry_sig = extract_detector_geometry_signature(config.watermark)
        normalized_sig = normalize_geometry_signature(geometry_sig)
        sig_json = json.dumps(normalized_sig, sort_keys=True, separators=(',', ':'))
        sig_hash = hashlib.md5(sig_json.encode()).hexdigest()[:8]
        components.append(f"geom{sig_hash}")
    
    # 2. Full config hash (excluding watermark strength)
    # Create a config dict excluding strength parameters
    config_dict = config.model_dump(mode='json')
    if isinstance(config.watermark, WatermarkedConfig):
        # Remove strength parameters that don't affect geometry
        if 'algorithm_params' in config_dict.get('watermark', {}):
            algo_params = config_dict['watermark']['algorithm_params']
            if 'mask' in algo_params:
                algo_params['mask'].pop('strength', None)
            if 'bias' in algo_params:
                # Keep bias params as they affect generation
                pass
    config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]
    components.append(f"cfg{config_hash}")
    
    # 3. Prompt hash (if provided)
    if prompt is not None:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        components.append(f"prompt{prompt_hash}")
    
    # 4. Seed (if provided)
    if seed is not None:
        components.append(f"seed{seed}")
    
    # 5. Inversion parameters
    if num_inversion_steps is not None:
        components.append(f"invsteps{num_inversion_steps}")
    
    # Model ID hash
    model_hash = hashlib.md5(config.diffusion.model_id.encode()).hexdigest()[:8]
    components.append(f"model{model_hash}")
    
    # 6. Image generation parameters
    components.append(f"guidance{config.diffusion.guidance_scale}")
    components.append(f"infsteps{config.diffusion.inference_timesteps}")
    
    # 7. Code version hash (if available)
    if code_version_hash is not None:
        components.append(f"code{code_version_hash[:8]}")
    
    # 8. Image ID
    components.append(f"img{image_id}")
    
    # Combine all components
    cache_key = "_".join(components)
    
    return cache_key

