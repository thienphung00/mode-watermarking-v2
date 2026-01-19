"""
Unit tests for G-field generation.

Fast, CPU-only tests that verify mathematical correctness using PRF-based seeds.
"""
import numpy as np
import pytest

from src.algorithms.g_field import GFieldGenerator
from src.detection.prf import PRFKeyDerivation, PRFConfig


class TestGFieldGenerator:
    """Test G-field generator with PRF-based seeds."""

    def test_generates_correct_shape(self):
        """Test G-field has correct shape."""
        gen = GFieldGenerator(mapping_mode="binary", domain="spatial")
        prf = PRFKeyDerivation("test_master_key_32bytes_long!", PRFConfig())
        # Generate PRF seeds for G-field
        seeds = prf.generate_seeds("test_key_id", 4*64*64*2)

        G = gen.generate_g_field(shape=(4, 64, 64), seeds=seeds)

        assert G.shape == (4, 64, 64)
        assert G.dtype == np.float32

    def test_zero_mean_normalization(self):
        """Test zero-mean normalization works."""
        gen = GFieldGenerator(
            mapping_mode="binary",
            domain="spatial",
            normalize_zero_mean=True,
            normalize_unit_variance=False,
        )
        prf = PRFKeyDerivation("test_master_key_32bytes_long!", PRFConfig())
        seeds = prf.generate_seeds("test_key_id", 4*64*64*2)

        G = gen.generate_g_field(shape=(4, 64, 64), seeds=seeds)

        mean = np.mean(G)
        assert abs(mean) < 1e-5  # Should be very close to zero

    def test_unit_variance_normalization(self):
        """Test unit-variance normalization works."""
        gen = GFieldGenerator(
            mapping_mode="binary",
            domain="spatial",
            normalize_zero_mean=True,
            normalize_unit_variance=True,
        )
        prf = PRFKeyDerivation("test_master_key_32bytes_long!", PRFConfig())
        seeds = prf.generate_seeds("test_key_id", 4*64*64*2)

        G = gen.generate_g_field(shape=(4, 64, 64), seeds=seeds)

        std = np.std(G)
        assert abs(std - 1.0) < 0.1  # Should be close to 1.0

    def test_deterministic_generation(self):
        """Test deterministic generation from same key_id."""
        gen = GFieldGenerator(mapping_mode="binary", domain="spatial")
        prf = PRFKeyDerivation("test_master_key_32bytes_long!", PRFConfig())

        seeds1 = prf.generate_seeds("test_key_id", 4*64*64*2)
        G1 = gen.generate_g_field(shape=(4, 64, 64), seeds=seeds1)

        seeds2 = prf.generate_seeds("test_key_id", 4*64*64*2)
        G2 = gen.generate_g_field(shape=(4, 64, 64), seeds=seeds2)

        np.testing.assert_array_equal(G1, G2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

