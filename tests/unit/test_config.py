"""
Unit tests for Pydantic configuration models.

Tests validation and discriminated unions.
"""
import pytest
from pydantic import ValidationError

from src.core.config import (
    AppConfig,
    WatermarkedConfig,
    UnwatermarkedConfig,
    DiffusionConfig,
    ModeType,
)


class TestWatermarkConfig:
    """Test watermark configuration validation."""

    def test_watermarked_requires_parameters(self):
        """Test that watermarked mode requires all parameters."""
        with pytest.raises(ValidationError):
            # Missing algorithm_params, key_settings
            WatermarkedConfig(mode=ModeType.WATERMARKED)

    def test_unwatermarked_minimal(self):
        """Test that unwatermarked mode needs minimal config."""
        config = UnwatermarkedConfig(mode=ModeType.UNWATERMARKED)
        assert config.mode == ModeType.UNWATERMARKED

    def test_alpha_bounds_validation(self):
        """Test that alpha bounds are validated."""
        from src.core.config import BiasConfig

        with pytest.raises(ValidationError):
            # alpha_bounds[0] >= alpha_bounds[1]
            BiasConfig(alpha_bounds=(0.08, 0.0))


class TestDiffusionConfig:
    """Test diffusion configuration validation."""

    def test_beta_range_validation(self):
        """Test that beta_end > beta_start."""
        with pytest.raises(ValidationError):
            DiffusionConfig(
                trained_timesteps=1000,
                inference_timesteps=50,
                beta_start=0.012,
                beta_end=0.00085,  # Invalid: end < start
            )

    def test_valid_config(self):
        """Test that valid config passes."""
        config = DiffusionConfig(
            trained_timesteps=1000,
            inference_timesteps=50,
            beta_start=0.00085,
            beta_end=0.012,
        )
        assert config.trained_timesteps == 1000


class TestAppConfig:
    """Test master app configuration."""

    def test_discriminated_union(self):
        """Test that discriminated union works correctly."""
        # This would be loaded from YAML in practice
        config_data = {
            "watermark": {"mode": "unwatermarked"},
            "diffusion": {
                "trained_timesteps": 1000,
                "inference_timesteps": 50,
                "beta_start": 0.00085,
                "beta_end": 0.012,
            },
        }

        config = AppConfig(**config_data)
        assert isinstance(config.watermark, UnwatermarkedConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

