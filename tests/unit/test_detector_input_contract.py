"""
Unit tests for detector input contract validation.

Tests that detectors enforce their input contracts and fail clearly on violations.
"""
from __future__ import annotations

import pytest
import torch

from src.models.detectors import UNetDetector, BayesianDetector


class TestUNetDetectorContract:
    """Test UNetDetector input contract."""

    def test_correct_input_succeeds(self):
        """Feed correct input → forward succeeds."""
        model = UNetDetector()
        x = torch.randn(2, 4, 64, 64)  # Correct shape
        logits = model(x)
        assert logits.shape == (2, 1)

    def test_wrong_shape_fails_clearly(self):
        """Feed wrong shape → fails clearly."""
        model = UNetDetector()
        
        # Wrong number of dimensions
        with pytest.raises(AssertionError, match="Expected 4D tensor"):
            model(torch.randn(4, 64, 64))  # Missing batch dimension
        
        # Wrong spatial size
        with pytest.raises(AssertionError, match="Expected shape"):
            model(torch.randn(2, 4, 32, 32))  # Wrong spatial dimensions
        
        # Wrong channels
        with pytest.raises(AssertionError, match="Expected shape"):
            model(torch.randn(2, 3, 64, 64))  # Wrong channel count

    def test_wrong_representation_warning(self):
        """Feed wrong representation → warning or failure."""
        model = UNetDetector()
        x = torch.randn(2, 4, 64, 64)
        
        # Test with validation enabled
        # Binary values should trigger warning
        x_binary = torch.sign(x)  # Convert to ±1
        # This should work but may warn if validate=True
        logits = model(x_binary, validate=False)
        assert logits.shape == (2, 1)
        
        # With validation, should warn about statistics
        with pytest.warns(UserWarning, match="Input mean"):
            model(x_binary * 2.0, validate=True)  # Mean far from 0


class TestBayesianDetectorContract:
    """Test BayesianDetector input contract."""

    def test_correct_input_succeeds(self):
        """Feed correct input → forward succeeds."""
        model = BayesianDetector(master_key="test_key_12345")
        g_values = torch.randint(0, 2, (2, 100)).float()  # Binary g-values [B, N]
        key = "test_key_id"
        result = model(g_values, key)
        assert "score" in result
        assert "decision" in result
        assert "matches" in result
        assert "num_bits" in result
        assert result["score"].shape == (2,)
        assert result["decision"].shape == (2,)
        assert result["matches"].shape == (2,)

    def test_correct_input_with_tensor_key(self):
        """Feed correct input with tensor key → forward succeeds."""
        model = BayesianDetector()
        g_values = torch.randint(0, 2, (2, 100)).float()  # Binary g-values [B, N]
        # Use tensor as key (expected g-values)
        key = torch.randint(0, 2, (100,)).float()
        result = model(g_values, key)
        assert "score" in result
        assert result["score"].shape == (2,)

    def test_wrong_shape_fails_clearly(self):
        """Feed wrong shape → fails clearly."""
        model = BayesianDetector(master_key="test_key")
        
        # Wrong number of dimensions
        with pytest.raises(AssertionError, match="Expected 2D tensor"):
            model(torch.randint(0, 2, (100,)).float(), "key", validate=True)  # Missing batch dimension

    def test_missing_key_fails(self):
        """Feed key_id without master_key → fails clearly."""
        model = BayesianDetector()  # No master_key
        g_values = torch.randint(0, 2, (2, 100)).float()
        
        with pytest.raises(ValueError, match="master_key must be provided"):
            model(g_values, "key_id")

    def test_rademacher_format(self):
        """Test with Rademacher {-1,+1} format."""
        model = BayesianDetector(master_key="test_key")
        g_values = torch.randint(0, 2, (2, 100)).float() * 2 - 1  # Convert to {-1,+1}
        key = "test_key_id"
        result = model(g_values, key)
        assert result["score"].shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

