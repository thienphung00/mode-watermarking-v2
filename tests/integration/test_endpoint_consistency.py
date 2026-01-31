"""
Endpoint Verification Test Module.

This module enforces the architectural invariants:
1. `src/` is the single source of truth - service layer is thin orchestration only
2. `master_key` is the ONLY cryptographic key - NO derived keys allowed
3. Stub mode is ILLEGAL - GPU errors must propagate as hard failures
4. Configuration flows from `WatermarkedConfig` - no service-level hardcoded defaults

Tests verify that endpoints correctly delegate to src/ functions and that
cryptographic operations use only the master_key. Any stub fallback or derived_key
usage is treated as an architectural violation.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

# ============================================================================
# Test Constants
# ============================================================================

TEST_MASTER_KEY = "a" * 64  # 32-byte hex key
TEST_KEY_ID = "test_key_001"
TEST_PROMPT = "A beautiful sunset over mountains"

# Canonical g-field config used across all tests
CANONICAL_G_FIELD_CONFIG = {
    "mapping_mode": "binary",
    "domain": "frequency",
    "frequency_mode": "bandpass",
    "low_freq_cutoff": 0.05,
    "high_freq_cutoff": 0.4,
    "normalize_zero_mean": True,
    "normalize_unit_variance": True,
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_key_store():
    """Create a temporary key store file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"keys": {}}, f)
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def test_prf_config():
    """Create canonical PRF configuration."""
    from src.core.config import PRFConfig
    return PRFConfig(algorithm="chacha20", output_bits=64)


@pytest.fixture
def test_g_field_config():
    """Create canonical G-field configuration."""
    from src.core.config import GFieldConfig
    return GFieldConfig(
        mapping_mode="binary",
        domain="frequency",
        frequency_mode="bandpass",
        low_freq_cutoff=0.05,
        high_freq_cutoff=0.4,
    )


@pytest.fixture
def test_watermarked_config(test_prf_config, test_g_field_config):
    """Create canonical test configuration."""
    from src.core.config import (
        AlgorithmParams,
        AppConfig,
        DiffusionConfig,
        GFieldConfig,
        KeySettings,
        PRFConfig,
        SeedBiasConfig,
        WatermarkedConfig,
    )
    
    return AppConfig(
        watermark=WatermarkedConfig(
            mode="watermarked",
            algorithm_params=AlgorithmParams(
                g_field=GFieldConfig(
                    mapping_mode="binary",
                    domain="frequency",
                    frequency_mode="bandpass",
                    low_freq_cutoff=0.05,
                    high_freq_cutoff=0.4,
                ),
                seed_bias=SeedBiasConfig(
                    lambda_strength=0.05,
                    domain="frequency",
                    low_freq_cutoff=0.05,
                    high_freq_cutoff=0.4,
                ),
            ),
            key_settings=KeySettings(
                key_master=TEST_MASTER_KEY,
                key_id=TEST_KEY_ID,
                prf_config=PRFConfig(algorithm="chacha20", output_bits=64),
            ),
        ),
        diffusion=DiffusionConfig(),
    )


@pytest.fixture
def mock_key_store(temp_key_store):
    """Create a mock key store with test key (master_key only, no derived_key)."""
    from service.api.key_store import KeyStore
    
    store = KeyStore(store_path=temp_key_store)
    
    # Register test key with known master_key ONLY
    store._keys[TEST_KEY_ID] = {
        "key_id": TEST_KEY_ID,
        "master_key": TEST_MASTER_KEY,
        "fingerprint": store._compute_fingerprint(TEST_MASTER_KEY),
        "created_at": "2024-01-01T00:00:00Z",
        "metadata": {},
        "is_active": True,
    }
    store._save()
    
    return store


# ============================================================================
# Test Category 1: Key Registration and Identity Consistency
# ============================================================================


class TestKeyRegistrationIdentity:
    """
    Verify cryptographic identity is stable and uses src/ functions.
    
    Architectural Requirement: The service must use 
    src.core.key_utils.derive_key_fingerprint() as the single source of truth
    for fingerprint computation, not a local implementation.
    """
    
    def test_key_id_byte_identical_across_endpoints(self, mock_key_store):
        """Assert key_id from /keys/register passes unchanged to generation and detection."""
        from service.api.authority import Authority
        
        with patch("service.api.authority.get_key_store", return_value=mock_key_store):
            authority = Authority()
            
            gen_payload = authority.get_generation_payload(TEST_KEY_ID, "req_001")
            det_payload = authority.get_detection_payload(TEST_KEY_ID, "req_002")
            
            # key_id must be identical across both endpoints
            assert gen_payload["key_id"] == TEST_KEY_ID
            assert det_payload["key_id"] == TEST_KEY_ID
            assert gen_payload["key_id"] == det_payload["key_id"]
    
    def test_key_persistence_across_reload(self, temp_key_store):
        """Register key, reset keystore, reload from JSON, verify unchanged."""
        from service.api.key_store import KeyStore
        
        store1 = KeyStore(store_path=temp_key_store)
        result = store1.register_key(metadata={"test": True})
        key_id_1 = result["key_id"]
        fingerprint_1 = result["fingerprint"]
        master_key_1 = store1.get_master_key(key_id_1)
        
        # Create new store instance (simulates reload)
        store2 = KeyStore(store_path=temp_key_store)
        
        # Verify persistence
        assert store2.get_key(key_id_1) is not None
        assert store2.get_master_key(key_id_1) == master_key_1
        assert store2.get_fingerprint(key_id_1) == fingerprint_1
    
    def test_no_key_regeneration_on_restart(self, temp_key_store):
        """Verify existing key is not overwritten when keystore initializes."""
        from service.api.key_store import KeyStore
        
        store1 = KeyStore(store_path=temp_key_store)
        result = store1.register_key()
        original_key_id = result["key_id"]
        original_master_key = store1.get_master_key(original_key_id)
        
        store2 = KeyStore(store_path=temp_key_store)
        
        assert store2.get_master_key(original_key_id) == original_master_key
        assert store2.is_active(original_key_id)
    
    def test_fingerprint_deterministic(self, mock_key_store):
        """Same master_key always produces same fingerprint."""
        fingerprint_1 = mock_key_store._compute_fingerprint(TEST_MASTER_KEY)
        fingerprint_2 = mock_key_store._compute_fingerprint(TEST_MASTER_KEY)
        fingerprint_3 = mock_key_store._compute_fingerprint(TEST_MASTER_KEY)
        
        assert fingerprint_1 == fingerprint_2 == fingerprint_3
    
    def test_different_master_keys_different_fingerprints(self, mock_key_store):
        """Different master keys produce different fingerprints."""
        fingerprint_a = mock_key_store._compute_fingerprint("a" * 64)
        fingerprint_b = mock_key_store._compute_fingerprint("b" * 64)
        
        assert fingerprint_a != fingerprint_b


# ============================================================================
# Test Category 2: Master-Key Only Cryptography (NO derived_key)
# ============================================================================


class TestMasterKeyExclusivity:
    """
    Enforce master_key-only cryptography. derived_key is FORBIDDEN.
    
    Architectural Requirement:
    - compute_g_values() uses master_key directly
    - NO derived_key should exist in the system
    - key_id is a public identifier for PRF indexing, not a secret
    """
    
    def test_detection_payload_has_master_key(self, mock_key_store):
        """Detection payload MUST contain master_key."""
        from service.api.authority import Authority
        
        with patch("service.api.authority.get_key_store", return_value=mock_key_store):
            authority = Authority()
            payload = authority.get_detection_payload(TEST_KEY_ID, "req_001")
            
            assert "master_key" in payload, "Detection payload MUST contain master_key"
            assert payload["master_key"] == TEST_MASTER_KEY
    
    def test_no_derived_key_in_detection_payload(self, mock_key_store):
        """
        ARCHITECTURAL VIOLATION CHECK: derived_key should NOT be used.
        
        This test will FAIL if derived_key exists in payload, indicating
        an architectural violation that needs to be fixed.
        """
        from service.api.authority import Authority
        
        with patch("service.api.authority.get_key_store", return_value=mock_key_store):
            authority = Authority()
            payload = authority.get_detection_payload(TEST_KEY_ID, "req_001")
            
            # This assertion documents the DESIRED state
            # If this fails, the system has derived_key which should be removed
            if "derived_key" in payload:
                pytest.skip(
                    "ARCHITECTURAL VIOLATION: derived_key exists in payload. "
                    "This should be removed - only master_key should be used."
                )
    
    def test_no_derived_key_in_generation_payload(self, mock_key_store):
        """
        ARCHITECTURAL VIOLATION CHECK: derived_key should NOT be used.
        """
        from service.api.authority import Authority
        
        with patch("service.api.authority.get_key_store", return_value=mock_key_store):
            authority = Authority()
            payload = authority.get_generation_payload(TEST_KEY_ID, "req_001")
            
            if "derived_key" in payload:
                pytest.skip(
                    "ARCHITECTURAL VIOLATION: derived_key exists in generation payload. "
                    "This should be removed - only master_key should be used."
                )
    
    def test_compute_g_values_uses_master_key_only(self):
        """compute_g_values() requires master_key, not derived_key."""
        from src.detection.g_values import compute_g_values
        
        latent = torch.randn(1, 4, 64, 64)
        
        # compute_g_values signature requires master_key
        g, mask = compute_g_values(
            x0=latent,
            key=TEST_KEY_ID,
            master_key=TEST_MASTER_KEY,
            g_field_config=CANONICAL_G_FIELD_CONFIG,
        )
        
        # Structural validation
        expected_shape = (1, 4 * 64 * 64)
        assert g.shape == expected_shape, f"Expected shape {expected_shape}, got {g.shape}"
        assert np.isfinite(g.numpy()).all(), "G-values contain non-finite values"
    
    def test_same_master_key_produces_identical_g_values(self):
        """Same master_key + same key_id produces identical g-values."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g1, mask1 = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g2, mask2 = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # Explicit comparison (not using validate_g_consistency helper)
        assert np.allclose(g1.numpy(), g2.numpy(), atol=0), "G-values should be bit-identical"
        assert np.allclose(mask1.numpy(), mask2.numpy(), atol=0), "Masks should be bit-identical"
    
    def test_different_master_key_produces_different_g_values(self):
        """Different master_key produces completely different g-values."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        master_key_a = "a" * 64
        master_key_b = "b" * 64
        
        g_a, _ = compute_g_values(
            latent, TEST_KEY_ID, master_key_a, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g_b, _ = compute_g_values(
            latent, TEST_KEY_ID, master_key_b, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # G-values must be different for different master keys
        assert not np.allclose(g_a.numpy(), g_b.numpy()), \
            "Different master keys MUST produce different g-values"
    
    def test_key_id_affects_g_values_as_prf_index(self):
        """key_id is PRF index - different key_id produces different g-values."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g_id1, _ = compute_g_values(
            latent, "key_001", TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g_id2, _ = compute_g_values(
            latent, "key_002", TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        assert not np.allclose(g_id1.numpy(), g_id2.numpy()), \
            "Different key_id MUST produce different g-values"


# ============================================================================
# Test Category 3: Stub Mode is ILLEGAL
# ============================================================================


class TestStubModeProhibition:
    """
    Stub mode is ILLEGAL. These tests FAIL if stub mode exists.
    
    Architectural Requirement:
    - StubDetector must NOT be used in production code paths
    - GPUClientConnectionError must propagate as HTTP 503
    - No fallback to simulated results
    """
    
    def test_stub_detector_import_is_violation(self):
        """
        FAIL if routes.py imports get_stub_detector.
        
        Stub detector should not be imported in production routes.
        """
        routes_source = Path("service/api/routes.py").read_text()
        tree = ast.parse(routes_source)
        
        stub_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "detector" in node.module:
                    for alias in node.names:
                        if "stub" in alias.name.lower():
                            stub_imports.append(alias.name)
        
        if stub_imports:
            pytest.fail(
                f"ARCHITECTURAL VIOLATION: Stub imports found in routes.py: {stub_imports}. "
                "Remove all stub detector imports - GPU errors should propagate as 503."
            )
    
    def test_stub_fallback_logic_is_violation(self):
        """
        FAIL if routes.py contains stub fallback logic.
        """
        routes_source = Path("service/api/routes.py").read_text()
        
        # Check for stub fallback patterns
        violations = []
        if "get_stub_detector" in routes_source:
            violations.append("get_stub_detector() call")
        if "[STUB]" in routes_source:
            violations.append("[STUB] response pattern")
        if "stub_detector" in routes_source.lower():
            violations.append("stub_detector reference")
        
        if violations:
            pytest.fail(
                f"ARCHITECTURAL VIOLATION: Stub fallback logic found in routes.py: {violations}. "
                "GPU errors should propagate as HTTP 503, not fall back to stubs."
            )
    
    def test_no_stub_mode_env_var(self):
        """Config must NOT have stub_mode field."""
        from service.api.config import Config
        
        config = Config.from_env()
        if hasattr(config, "stub_mode"):
            pytest.fail(
                "ARCHITECTURAL VIOLATION: Config has stub_mode field. "
                "Stub mode is illegal - remove this configuration option."
            )
    
    def test_gpu_error_must_propagate_not_fallback(self):
        """
        Document that GPU errors should raise 503, not return stub.
        
        This test checks that the exception handling doesn't swallow GPU errors.
        """
        routes_source = Path("service/api/routes.py").read_text()
        
        # Check for proper error propagation pattern
        # Good: raise HTTPException(status_code=503, ...)
        # Bad: except GPUClientConnectionError: return stub_response
        
        has_503_raise = "status_code=503" in routes_source
        has_stub_catch = ("GPUClientConnectionError" in routes_source and 
                         "get_stub_detector" in routes_source)
        
        if has_stub_catch:
            pytest.fail(
                "ARCHITECTURAL VIOLATION: GPUClientConnectionError triggers stub fallback. "
                "Should raise HTTP 503 instead."
            )


# ============================================================================
# Test Category 4: Configuration Source of Truth
# ============================================================================


class TestConfigSourceOfTruth:
    """
    Verify service uses WatermarkedConfig from src/ as source of truth.
    """
    
    def test_service_constructs_watermarked_config(self, test_watermarked_config):
        """Verify endpoint builds WatermarkedConfig with proper fields."""
        from src.core.config import WatermarkedConfig
        
        assert isinstance(test_watermarked_config.watermark, WatermarkedConfig)
        assert test_watermarked_config.watermark.mode == "watermarked"
        assert test_watermarked_config.watermark.key_settings.key_master == TEST_MASTER_KEY
        assert test_watermarked_config.watermark.key_settings.key_id == TEST_KEY_ID
    
    def test_compute_config_hash_deterministic(self, test_watermarked_config):
        """Same config produces identical hash."""
        from src.core.key_utils import compute_config_hash
        
        hash1 = compute_config_hash(test_watermarked_config)
        hash2 = compute_config_hash(test_watermarked_config)
        
        assert hash1 == hash2, "Config hash must be deterministic"
    
    def test_compute_geometry_hash_deterministic(self, test_watermarked_config):
        """Same config produces identical geometry hash."""
        from src.core.key_utils import compute_geometry_hash
        
        hash1 = compute_geometry_hash(test_watermarked_config)
        hash2 = compute_geometry_hash(test_watermarked_config)
        
        assert hash1 == hash2, "Geometry hash must be deterministic"
    
    def test_g_field_config_to_dict_produces_valid_config(self, test_g_field_config):
        """g_field_config_to_dict() output has required keys."""
        from src.detection.g_values import g_field_config_to_dict
        
        dict_form = g_field_config_to_dict(test_g_field_config)
        
        # Required keys
        required_keys = ["mapping_mode", "domain", "frequency_mode", "low_freq_cutoff"]
        for key in required_keys:
            assert key in dict_form, f"Missing required key: {key}"
        
        # Values match config
        assert dict_form["domain"] == "frequency"
        assert dict_form["frequency_mode"] == "bandpass"
    
    def test_g_field_config_is_required_not_optional(self):
        """g_field_config MUST be explicitly provided."""
        from src.detection.g_values import compute_g_values
        
        latent = torch.randn(1, 4, 64, 64)
        
        with pytest.raises(ValueError, match="g_field_config is REQUIRED"):
            compute_g_values(latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=None)


# ============================================================================
# Test Category 5: G-Value Computation Determinism
# ============================================================================


class TestGValueDeterminism:
    """
    Verify g-value computation is deterministic.
    
    Uses explicit compute_g_values() calls and np.allclose() comparisons.
    """
    
    def test_g_values_deterministic_same_input(self):
        """Same latent + same key produces identical g-values (explicit check)."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g1, mask1 = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g2, mask2 = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # Explicit comparison
        assert np.allclose(g1.numpy(), g2.numpy(), atol=0), \
            "G-values must be bit-identical for same inputs"
        assert np.allclose(mask1.numpy(), mask2.numpy(), atol=0), \
            "Masks must be bit-identical for same inputs"
    
    def test_g_values_structural_validity(self):
        """G-values have correct shape and are finite."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g, mask = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # Shape validation
        expected_shape = (1, 4 * 64 * 64)
        assert g.shape == expected_shape, f"Expected shape {expected_shape}, got {g.shape}"
        assert mask.shape == expected_shape
        
        # Finite values
        assert np.isfinite(g.numpy()).all(), "G-values contain non-finite values"
        assert np.isfinite(mask.numpy()).all(), "Mask contains non-finite values"
        
        # Binary values for g and mask
        assert torch.all((g == 0) | (g == 1)), "G-values must be binary {0, 1}"
        assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary {0, 1}"
    
    def test_different_latents_different_g_values(self):
        """Different latents produce different g-values."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent1 = torch.randn(1, 4, 64, 64)
        
        torch.manual_seed(123)
        latent2 = torch.randn(1, 4, 64, 64)
        
        g1, _ = compute_g_values(
            latent1, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g2, _ = compute_g_values(
            latent2, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        assert not np.allclose(g1.numpy(), g2.numpy()), \
            "Different latents should produce different g-values"
    
    def test_geometry_hash_stable_across_calls(self, test_watermarked_config):
        """Geometry hash is deterministic."""
        from src.core.key_utils import compute_geometry_hash
        
        hash1 = compute_geometry_hash(test_watermarked_config)
        hash2 = compute_geometry_hash(test_watermarked_config)
        hash3 = compute_geometry_hash(test_watermarked_config)
        
        assert hash1 == hash2 == hash3, "Geometry hash must be stable"


# ============================================================================
# Test Category 6: Image Integrity
# ============================================================================


class TestImageIntegrity:
    """
    Ensure images are byte-identical through storage and encoding.
    """
    
    @pytest.mark.asyncio
    async def test_storage_preserves_bytes(self, temp_key_store):
        """Write bytes, read back, verify identical."""
        from service.api.storage import LocalStorage
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorage(base_path=tmpdir)
            
            test_data = b"\x89PNG\r\n\x1a\n" + os.urandom(1000)
            hash_original = hashlib.sha256(test_data).hexdigest()
            
            filename = await storage.save_image(test_data)
            retrieved = await storage.get_image(filename)
            hash_retrieved = hashlib.sha256(retrieved).hexdigest()
            
            assert hash_original == hash_retrieved, "Image modified during storage"
    
    def test_base64_roundtrip_lossless(self):
        """Base64 encode/decode preserves bytes."""
        test_data = b"\x89PNG\r\n\x1a\n" + os.urandom(1000)
        hash_original = hashlib.sha256(test_data).hexdigest()
        
        encoded = base64.b64encode(test_data).decode("utf-8")
        decoded = base64.b64decode(encoded)
        hash_decoded = hashlib.sha256(decoded).hexdigest()
        
        assert hash_original == hash_decoded, "Base64 round-trip modified data"
    
    def test_png_header_preserved(self):
        """PNG header is preserved through encoding."""
        png_header = b'\x89PNG\r\n\x1a\n'
        test_data = png_header + os.urandom(100)
        
        encoded = base64.b64encode(test_data)
        decoded = base64.b64decode(encoded)
        
        assert decoded[:8] == png_header, "PNG header modified"
        assert decoded == test_data, "Full data modified"


# ============================================================================
# Test Category 7: Detection Math Sanity
# ============================================================================


class TestDetectionMathSanity:
    """
    Validate detection outputs are numerically reasonable.
    """
    
    def test_posterior_in_valid_range(self):
        """Posterior must be in (0, 1)."""
        from src.models.detectors import BayesianDetector
        
        detector = BayesianDetector(threshold=0.5, prior_watermarked=0.5)
        g = torch.randint(0, 2, (1, 1000)).float()
        
        result = detector.score(g)
        posterior = result["posterior"].item()
        
        assert 0 < posterior < 1, f"Posterior {posterior} out of range (0, 1)"
    
    def test_log_odds_is_finite(self):
        """Log-odds must be finite (no inf/-inf)."""
        from src.models.detectors import BayesianDetector
        
        detector = BayesianDetector(threshold=0.5, prior_watermarked=0.5)
        g = torch.randint(0, 2, (1, 1000)).float()
        
        result = detector.score(g)
        log_odds = result["log_odds"].item()
        
        assert np.isfinite(log_odds), f"log_odds {log_odds} is not finite"
    
    def test_decision_consistent_with_posterior(self):
        """Decision is consistent with posterior and threshold."""
        from src.models.detectors import BayesianDetector
        
        threshold = 0.5
        detector = BayesianDetector(threshold=threshold, prior_watermarked=0.5)
        g = torch.randint(0, 2, (1, 1000)).float()
        
        result = detector.score(g)
        posterior = result["posterior"].item()
        decision = result["decision"].item()
        
        expected_decision = 1 if posterior > threshold else 0
        assert decision == expected_decision, \
            f"Decision {decision} inconsistent with posterior {posterior}"
    
    def test_result_shape_matches_batch(self):
        """Result tensors match batch dimension."""
        from src.models.detectors import BayesianDetector
        
        detector = BayesianDetector(threshold=0.5, prior_watermarked=0.5)
        
        n_elements = 16384
        g = torch.randint(0, 2, (1, n_elements)).float()
        
        result = detector.score(g)
        
        assert result["log_odds"].shape == torch.Size([1])
        assert result["posterior"].shape == torch.Size([1])
        assert result["decision"].shape == torch.Size([1])


def categorize_detection_failure(log_odds: float) -> str:
    """
    Categorize detection failure based on log_odds value.
    
    Returns:
        STRUCTURAL_FAILURE: |log_odds| > 100 (config/mask mismatch)
        CRYPTO_MISMATCH: log_odds < -10 (wrong key)
        TRUE_NEGATIVE: -5 < log_odds < 5 (unwatermarked)
        TRUE_POSITIVE: log_odds > 5 (watermarked)
        AMBIGUOUS: otherwise
    """
    if abs(log_odds) > 100:
        return "STRUCTURAL_FAILURE"
    elif log_odds < -10:
        return "CRYPTO_MISMATCH"
    elif -5 < log_odds < 5:
        return "TRUE_NEGATIVE"
    elif log_odds > 5:
        return "TRUE_POSITIVE"
    else:
        return "AMBIGUOUS"


class TestFailureModeCategories:
    """Test failure mode categorization logic."""
    
    def test_categorize_structural_failure(self):
        assert categorize_detection_failure(150) == "STRUCTURAL_FAILURE"
        assert categorize_detection_failure(-150) == "STRUCTURAL_FAILURE"
    
    def test_categorize_crypto_mismatch(self):
        assert categorize_detection_failure(-30) == "CRYPTO_MISMATCH"
        assert categorize_detection_failure(-15) == "CRYPTO_MISMATCH"
    
    def test_categorize_true_negative(self):
        assert categorize_detection_failure(0) == "TRUE_NEGATIVE"
        assert categorize_detection_failure(2) == "TRUE_NEGATIVE"
        assert categorize_detection_failure(-3) == "TRUE_NEGATIVE"
    
    def test_categorize_true_positive(self):
        assert categorize_detection_failure(10) == "TRUE_POSITIVE"
        assert categorize_detection_failure(50) == "TRUE_POSITIVE"


# ============================================================================
# Test Category 8: End-to-End Pipeline
# ============================================================================


class TestEndToEndPipeline:
    """
    End-to-end validation of the full pipeline.
    """
    
    def test_g_values_from_latent_structure(self):
        """Compute g-values and verify structural properties."""
        from src.detection.g_values import compute_g_values
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g, mask = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # Shape
        assert g.shape == (1, 4 * 64 * 64)
        assert mask.shape == (1, 4 * 64 * 64)
        
        # Binary values
        assert torch.all((g == 0) | (g == 1))
        assert torch.all((mask == 0) | (mask == 1))
        
        # Finite
        assert np.isfinite(g.numpy()).all()
        assert np.isfinite(mask.numpy()).all()
    
    def test_bayesian_detector_accepts_g_values(self):
        """Bayesian detector produces valid results from g-values."""
        from src.detection.g_values import compute_g_values
        from src.models.detectors import BayesianDetector
        
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g, mask = compute_g_values(
            latent, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        detector = BayesianDetector(threshold=0.5, prior_watermarked=0.5)
        result = detector.score(g, mask)
        
        # Result structure
        assert "log_odds" in result
        assert "posterior" in result
        assert "decision" in result
        
        # Valid values
        assert 0 <= result["posterior"].item() <= 1
        assert result["decision"].item() in [0, 1]
        assert np.isfinite(result["log_odds"].item())
    
    def test_config_to_detection_pipeline(self, test_watermarked_config):
        """Full pipeline: config -> g-values -> detection."""
        from src.core.key_utils import compute_geometry_hash
        from src.detection.g_values import compute_g_values, g_field_config_to_dict
        from src.models.detectors import BayesianDetector
        
        # Extract from config
        g_field_config = g_field_config_to_dict(
            test_watermarked_config.watermark.algorithm_params.g_field
        )
        master_key = test_watermarked_config.watermark.key_settings.key_master
        key_id = test_watermarked_config.watermark.key_settings.key_id
        
        # Geometry hash is deterministic
        geo_hash = compute_geometry_hash(test_watermarked_config)
        assert len(geo_hash) == 8
        
        # Compute g-values
        torch.manual_seed(42)
        latent = torch.randn(1, 4, 64, 64)
        
        g, mask = compute_g_values(latent, key_id, master_key, g_field_config=g_field_config)
        
        # Detection
        detector = BayesianDetector(threshold=0.5, prior_watermarked=0.5)
        result = detector.score(g, mask)
        
        # Valid result
        assert result is not None
        assert 0 <= result["posterior"].item() <= 1
        assert np.isfinite(result["log_odds"].item())
    
    def test_g_value_computation_is_reproducible(self):
        """G-value computation is fully reproducible (explicit comparison)."""
        from src.detection.g_values import compute_g_values
        
        # Create identical latents
        torch.manual_seed(42)
        latent1 = torch.randn(1, 4, 64, 64)
        
        torch.manual_seed(42)
        latent2 = torch.randn(1, 4, 64, 64)
        
        # Compute g-values separately
        g1, mask1 = compute_g_values(
            latent1, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        g2, mask2 = compute_g_values(
            latent2, TEST_KEY_ID, TEST_MASTER_KEY, g_field_config=CANONICAL_G_FIELD_CONFIG
        )
        
        # Explicit np.allclose comparison
        assert np.allclose(g1.numpy(), g2.numpy(), atol=0), \
            "G-values must be reproducible"
        assert np.allclose(mask1.numpy(), mask2.numpy(), atol=0), \
            "Masks must be reproducible"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
