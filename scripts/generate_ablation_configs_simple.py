#!/usr/bin/env python3
"""
Generate the 12 watermark ablation YAML configs (simple, deterministic, hard-coded).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from enum import Enum

import yaml

# Ensure repo root is on sys.path so `import src...` works when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import (
    AlgorithmParams,
    AppConfig,
    BiasConfig,
    DiffusionConfig,
    GFieldConfig,
    KeySettings,
    MaskConfig,
    PRFConfig,
    SeedBiasConfig,
    WatermarkedConfig,
)

# ============================================================================
# Hard-coded parameter grid
# ============================================================================

STRENGTHS = {
    "baseline": {
        "lambda_strength": 0.075,
        "mask_strength": 0.90,
    },
    "strong": {
        "lambda_strength": 0.10,
        "mask_strength": 0.92,
    },
}

FREQUENCY_BANDS = {
    "bandpass_low": {"low": 0.03, "high": 0.25},
    "bandpass_mid": {"low": 0.05, "high": 0.40},
    "bandpass_wide": {"low": 0.02, "high": 0.50},
}

MAPPING_MODES = ["binary", "continuous"]

# ============================================================================
# Fixed constants (hard-coded)
# ============================================================================

G_FIELD_SHAPE = (4, 64, 64)
PRF_BITS = 64
PRF_ALGO = "chacha20"

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
TRAINED_TIMESTEPS = 1000
INFERENCE_TIMESTEPS = 50
SCHEDULER = "DDIM"
PREDICTION_TYPE = "epsilon"
GUIDANCE_SCALE = 7.5
USE_FP16 = True
GRADIENT_CHECKPOINTING = False

# Mask configuration (fixed across all configs)
MASK_MODE = "frequency"
MASK_BAND = "bandpass"
MASK_CUTOFF = 0.50
MASK_BANDWIDTH_FRACTION = 0.25


def build_config(
    *,
    strength: str,
    mapping_mode: str,
    frequency_band: str,
    key_master: str,
) -> AppConfig:
    strength_params = STRENGTHS[strength]
    band = FREQUENCY_BANDS[frequency_band]

    g_field = GFieldConfig(
        shape=G_FIELD_SHAPE,
        mapping_mode=mapping_mode,
        continuous_range=(-1.0, 1.0),
        channel_wise=True,
        domain="frequency",
        frequency_mode="bandpass",
        low_freq_cutoff=band["low"],
        high_freq_cutoff=band["high"],
        frequency_normalization="parseval",
        normalize={
            "enable": True,
            "zero_mean_per_timestep": True,
            "zero_mean_per_channel": True,
            "unit_variance": False,
            "eps": 1e-8,
        },
        normalize_zero_mean=None,
        normalize_unit_variance=None,
    )

    mask = MaskConfig(
        mode=MASK_MODE,
        strength=strength_params["mask_strength"],
        band=MASK_BAND,
        cutoff_freq=MASK_CUTOFF,
        bandwidth_fraction=MASK_BANDWIDTH_FRACTION,
        shape="radial",
        radius_fraction=0.50,
    )

    bias = BiasConfig(
        mode="non_distortionary",
        target_snr=0.05,
        alpha_bounds=(0.0, 0.08),
        min_strength=0.005,
        max_strength=0.025,
        injection={
            "strategy": "concentrated_late",
            "start_fraction": 0.70,
            "peak_fraction": 0.90,
            "end_fraction": 1.00,
            "shape": "triangular",
            "normalize_alpha": True,
        },
    )

    # Seed bias domain is always frequency and reuses the same band cutoffs as g-field.
    seed_bias = SeedBiasConfig(
        lambda_strength=strength_params["lambda_strength"],
        domain="frequency",
        low_freq_cutoff=band["low"],
        high_freq_cutoff=band["high"],
        mask_config=None,
    )

    key_id = f"ablation_{strength}_{mapping_mode}_{frequency_band}"
    key_settings = KeySettings(
        key_master=key_master,
        key_id=key_id,
        prf_config=PRFConfig(algorithm=PRF_ALGO, output_bits=PRF_BITS),
        experiment_id="watermark_ablation",
    )

    watermark = WatermarkedConfig(
        mode="watermarked",
        algorithm_params=AlgorithmParams(
            g_field=g_field,
            mask=mask,
            bias=bias,
            content_hash_strength=0.3,
            time_dependent_offset=False,
            seed_bias=seed_bias,
        ),
        key_settings=key_settings,
    )

    diffusion = DiffusionConfig(
        model_id=DIFFUSION_MODEL_ID,
        trained_timesteps=TRAINED_TIMESTEPS,
        inference_timesteps=INFERENCE_TIMESTEPS,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        prediction_type=PREDICTION_TYPE,
        scheduler=SCHEDULER,
        scheduler_kwargs={},
        guidance_scale=GUIDANCE_SCALE,
        use_fp16=USE_FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return AppConfig(watermark=watermark, diffusion=diffusion, training=None, evaluation=None)


def _to_plain_yaml_obj(obj):  # type: ignore[no-untyped-def]
    """
    Convert Pydantic/Enum/tuple objects into plain JSON/YAML-serializable types.
    Hard-coded, deterministic, and avoids PyYAML python-specific tags.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_plain_yaml_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_yaml_obj(v) for v in obj]
    return obj


def write_config_yaml(config: AppConfig, out_path: Path) -> None:
    data = config.model_dump(mode="python")
    plain = _to_plain_yaml_obj(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(plain, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate watermark ablation YAML configs (simple).")
    parser.add_argument(
        "--key-master",
        type=str,
        default="<your-secret-key-here>",
        help="Master key for PRF (will be written into each config).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    output_dir = REPO_ROOT / "experiments" / "watermark_ablation" / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_filenames: list[str] = []
    written_paths: list[Path] = []

    for strength in ["baseline", "strong"]:
        for mapping_mode in MAPPING_MODES:
            for frequency_band in ["bandpass_low", "bandpass_mid", "bandpass_wide"]:
                filename = f"{strength}_{mapping_mode}_{frequency_band}.yaml"
                expected_filenames.append(filename)

                config = build_config(
                    strength=strength,
                    mapping_mode=mapping_mode,
                    frequency_band=frequency_band,
                    key_master=args.key_master,
                )

                out_path = output_dir / filename
                write_config_yaml(config, out_path)
                logger.info(str(out_path))
                written_paths.append(out_path)

    # Validation: exactly 12 files written with expected names
    expected_set = set(expected_filenames)
    written_set = {p.name for p in written_paths}
    assert len(written_paths) == 12, f"Expected to write 12 files, wrote {len(written_paths)}"
    assert written_set == expected_set, f"Filename mismatch. Expected {sorted(expected_set)}, got {sorted(written_set)}"

    # Optionally log filenames sorted lexicographically
    logger.info("")
    for name in sorted(expected_set):
        logger.info(name)
    logger.info("")
    logger.info(f"✓ Generated {len(written_paths)} configs into: {output_dir}")


if __name__ == "__main__":
    main()


