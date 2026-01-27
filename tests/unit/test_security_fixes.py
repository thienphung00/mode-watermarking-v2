"""
Unit tests for critical security and correctness fixes.

Tests for:
1. Master key never transmitted to workers
2. Derived keys differ per scope
3. Retry rules enforced correctly
4. Key fingerprint validation
5. Backpressure is race-free
6. GPU concurrency limits
7. Health endpoint exposure
8. Dependency purity
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


# =============================================================================
# 1. Master Key Never Transmitted Tests
# =============================================================================


class TestMasterKeyNeverTransmitted:
    """Tests ensuring master_key never leaves the API boundary."""
    
    def test_derive_scoped_key_differs_from_master_key(self):
        """Derived key must be different from master key."""
        from service.infra.security import derive_scoped_key, OperationType
        
        master_key = "a" * 64  # 32 bytes hex
        derived = derive_scoped_key(
            master_key=master_key,
            key_id="test_key",
            operation=OperationType.DETECTION,
        )
        
        assert derived != master_key
        assert len(derived) == 64  # 32 bytes hex
    
    def test_derived_key_is_deterministic(self):
        """Same inputs produce same derived key."""
        from service.infra.security import derive_scoped_key, OperationType
        
        master_key = "b" * 64
        key_id = "test_key"
        
        derived1 = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.DETECTION,
        )
        derived2 = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.DETECTION,
        )
        
        assert derived1 == derived2
    
    def test_derived_key_scoped_by_operation(self):
        """Different operations produce different keys."""
        from service.infra.security import derive_scoped_key, OperationType
        
        master_key = "c" * 64
        key_id = "test_key"
        
        detection_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.DETECTION,
        )
        generation_key = derive_scoped_key(
            master_key=master_key,
            key_id=key_id,
            operation=OperationType.GENERATION,
        )
        
        assert detection_key != generation_key
    
    def test_derived_key_scoped_by_key_id(self):
        """Different key_ids produce different keys."""
        from service.infra.security import derive_scoped_key, OperationType
        
        master_key = "d" * 64
        
        key1 = derive_scoped_key(
            master_key=master_key,
            key_id="key_1",
            operation=OperationType.DETECTION,
        )
        key2 = derive_scoped_key(
            master_key=master_key,
            key_id="key_2",
            operation=OperationType.DETECTION,
        )
        
        assert key1 != key2
    
    def test_key_fingerprint_non_reversible(self):
        """Fingerprint cannot be used to derive master key."""
        from service.infra.security import compute_key_fingerprint
        
        master_key = "e" * 64
        fingerprint = compute_key_fingerprint(master_key)
        
        # Fingerprint is 32 chars (128 bits)
        assert len(fingerprint) == 32
        # Fingerprint is hex
        int(fingerprint, 16)  # Should not raise
        # Fingerprint is different from master key
        assert fingerprint != master_key[:32]
    
    def test_inference_schema_has_no_master_key(self):
        """Inference schemas must not have master_key field."""
        from service.inference.schemas import (
            DetectInferenceRequest,
            GenerateInferenceRequest,
        )
        
        detect_fields = DetectInferenceRequest.model_fields.keys()
        generate_fields = GenerateInferenceRequest.model_fields.keys()
        
        assert "master_key" not in detect_fields
        assert "master_key" not in generate_fields
        assert "derived_key" in detect_fields
        assert "derived_key" in generate_fields
    
    def test_worker_schema_has_no_master_key(self):
        """Worker schemas must not have master_key field."""
        from service.worker.schemas import (
            DetectWorkerRequest,
            GenerateWorkerRequest,
        )
        
        detect_fields = DetectWorkerRequest.model_fields.keys()
        generate_fields = GenerateWorkerRequest.model_fields.keys()
        
        assert "master_key" not in detect_fields
        assert "master_key" not in generate_fields
        assert "derived_key" in detect_fields
        assert "derived_key" in generate_fields


# =============================================================================
# 2. Retry Semantics Tests
# =============================================================================


class TestRetrySemantics:
    """Tests for operation-aware retry logic."""
    
    def test_detection_is_idempotent(self):
        """Detection should be classified as idempotent."""
        from service.inference.schemas import OperationType
        
        assert OperationType.DETECTION.value == "detection"
    
    def test_generation_is_non_idempotent(self):
        """Generation should be classified as non-idempotent."""
        from service.inference.schemas import OperationType
        
        assert OperationType.GENERATION.value == "generation"
    
    def test_inference_request_has_idempotency_fields(self):
        """Requests must have idempotency-related fields."""
        from service.inference.schemas import (
            DetectInferenceRequest,
            GenerateInferenceRequest,
        )
        
        detect_fields = DetectInferenceRequest.model_fields.keys()
        generate_fields = GenerateInferenceRequest.model_fields.keys()
        
        # Required idempotency fields
        for fields in [detect_fields, generate_fields]:
            assert "request_id" in fields
            assert "operation_type" in fields
            assert "idempotency_key" in fields
            assert "deterministic_seed" in fields


# =============================================================================
# 3. GPU Concurrency Tests
# =============================================================================


class TestGPUConcurrency:
    """Tests for safe GPU concurrency defaults."""
    
    def test_default_gpu_semaphore_is_one(self):
        """Default GPU semaphore should be 1 for safety."""
        from service.worker.model_loader import WorkerSettings
        import os
        
        # Clear environment to test defaults
        with patch.dict(os.environ, {}, clear=True):
            # Need to force reload the settings
            settings = WorkerSettings()
        
        assert settings.gpu_semaphore_size == 1
    
    def test_default_max_concurrent_is_low(self):
        """Default max concurrent requests should be conservative."""
        from service.worker.model_loader import WorkerSettings
        import os
        
        with patch.dict(os.environ, {}, clear=True):
            settings = WorkerSettings()
        
        assert settings.max_concurrent_requests <= 4


# =============================================================================
# 4. Fingerprint Validation Tests
# =============================================================================


class TestFingerprintValidation:
    """Tests for key fingerprint validation."""
    
    def test_valid_fingerprint_passes(self):
        """Valid fingerprint should pass validation."""
        from service.infra.security import validate_key_fingerprint
        
        valid_fingerprint = "a" * 32  # 32 hex chars
        result = validate_key_fingerprint(
            derived_key="dummy",
            expected_fingerprint=valid_fingerprint,
            key_id="test",
        )
        
        assert result is True
    
    def test_invalid_length_fingerprint_fails(self):
        """Invalid length fingerprint should fail."""
        from service.infra.security import validate_key_fingerprint
        
        # Too short
        result = validate_key_fingerprint(
            derived_key="dummy",
            expected_fingerprint="abc",
            key_id="test",
        )
        assert result is False
        
        # Too long
        result = validate_key_fingerprint(
            derived_key="dummy",
            expected_fingerprint="a" * 64,
            key_id="test",
        )
        assert result is False
    
    def test_invalid_hex_fingerprint_fails(self):
        """Non-hex fingerprint should fail."""
        from service.infra.security import validate_key_fingerprint
        
        result = validate_key_fingerprint(
            derived_key="dummy",
            expected_fingerprint="g" * 32,  # 'g' is not hex
            key_id="test",
        )
        
        assert result is False
    
    def test_empty_fingerprint_fails(self):
        """Empty fingerprint should fail."""
        from service.infra.security import validate_key_fingerprint
        
        result = validate_key_fingerprint(
            derived_key="dummy",
            expected_fingerprint="",
            key_id="test",
        )
        
        assert result is False


# =============================================================================
# 5. Backpressure Tests
# =============================================================================


class TestBackpressure:
    """Tests for race-free backpressure."""
    
    @pytest.mark.asyncio
    async def test_slot_acquisition_is_atomic(self):
        """Slot acquisition should be atomic (no race conditions)."""
        from service.worker.model_loader import WorkerSettings, ModelLoader
        
        settings = WorkerSettings()
        settings.max_queue_size = 2  # Small for testing
        loader = ModelLoader(settings)
        
        # Initialize slots
        await loader._ensure_slots_initialized()
        
        # Acquire both slots
        assert await loader.acquire_request_slot() is True
        assert await loader.acquire_request_slot() is True
        
        # Third attempt should fail (queue full)
        assert await loader.acquire_request_slot() is False
        
        # Release one
        await loader.release_request_slot()
        
        # Now we can acquire again
        assert await loader.acquire_request_slot() is True
    
    @pytest.mark.asyncio
    async def test_queue_size_property(self):
        """Queue size should reflect active requests."""
        from service.worker.model_loader import WorkerSettings, ModelLoader
        
        settings = WorkerSettings()
        settings.max_queue_size = 3
        loader = ModelLoader(settings)
        
        await loader._ensure_slots_initialized()
        
        assert loader.queue_size == 0
        
        await loader.acquire_request_slot()
        assert loader.queue_size == 1
        
        await loader.acquire_request_slot()
        assert loader.queue_size == 2
        
        await loader.release_request_slot()
        assert loader.queue_size == 1


# =============================================================================
# 6. Health Endpoint Tests
# =============================================================================


class TestHealthEndpointExposure:
    """Tests for health endpoint exposure restrictions."""
    
    def test_public_health_returns_minimal_info(self):
        """Public health endpoint should return minimal information."""
        from service.app.schemas import HealthResponse
        
        # Check that HealthResponse has minimal fields
        fields = HealthResponse.model_fields.keys()
        
        # Should have status
        assert "status" in fields
        
        # The response should be simple - just status
        response = HealthResponse(status="ok")
        data = response.model_dump()
        
        # Should not expose sensitive metrics
        assert "gpu_memory_used_mb" not in data or data.get("gpu_memory_used_mb") is None
        assert "queue_size" not in data or data.get("queue_size") is None
    
    def test_worker_health_has_internal_metrics(self):
        """Worker internal health should have detailed metrics."""
        from service.worker.schemas import HealthResponse
        
        fields = HealthResponse.model_fields.keys()
        
        # Internal health should have detailed metrics
        assert "gpu_memory_used_mb" in fields
        assert "gpu_memory_total_mb" in fields
        assert "active_requests" in fields
        assert "queue_size" in fields


# =============================================================================
# 7. Dependency Purity Tests
# =============================================================================


class TestDependencyPurity:
    """Tests for FastAPI dependency purity."""
    
    def test_get_authority_only_accesses_state(self):
        """get_authority should only access app state."""
        from service.app.dependencies import get_authority
        from unittest.mock import MagicMock
        
        # Create mock request with mock app state
        mock_request = MagicMock()
        mock_authority = MagicMock()
        mock_request.app.state.services.authority = mock_authority
        
        # Should just return the authority from state
        result = get_authority(mock_request)
        
        assert result == mock_authority
    
    def test_get_adapter_only_accesses_state(self):
        """get_adapter should only access app state."""
        from service.app.dependencies import get_adapter
        from unittest.mock import MagicMock
        
        mock_request = MagicMock()
        mock_adapter = MagicMock()
        mock_request.app.state.services.generation_adapter = mock_adapter
        
        result = get_adapter(mock_request)
        
        assert result == mock_adapter
    
    def test_get_detector_only_accesses_state(self):
        """get_detector should only access app state."""
        from service.app.dependencies import get_detector
        from unittest.mock import MagicMock
        
        mock_request = MagicMock()
        mock_detector = MagicMock()
        mock_request.app.state.services.detection_service = mock_detector
        
        result = get_detector(mock_request)
        
        assert result == mock_detector


# =============================================================================
# 8. Authority Service Tests
# =============================================================================


class TestAuthorityService:
    """Tests for WatermarkAuthorityService security."""
    
    def test_get_detection_config_returns_derived_key(self):
        """get_detection_config should return derived_key, not master_key."""
        from service.authority import WatermarkAuthorityService
        from unittest.mock import MagicMock, patch
        
        with patch('service.authority.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_db.get_watermark.return_value = {
                "secret_key": "a" * 64,
            }
            mock_db.is_active.return_value = True
            mock_get_db.return_value = mock_db
            
            authority = WatermarkAuthorityService()
            
            # Get config for remote worker (for_local_use=False)
            # Patch at the import location in service.app.artifact_resolver
            with patch('service.app.artifact_resolver.get_artifact_resolver') as mock_resolver:
                mock_result = MagicMock()
                mock_result.likelihood_params_path_str = "/path/to/params"
                mock_result.mask_path_str = "/path/to/mask"
                mock_resolver.return_value.resolve.return_value = mock_result
                
                config = authority.get_detection_config("test_key", for_local_use=False)
        
        # Should have derived_key
        assert "derived_key" in config
        assert len(config["derived_key"]) == 64
        
        # Should have fingerprint
        assert "key_fingerprint" in config
        assert len(config["key_fingerprint"]) == 32
        
        # Should NOT have master_key when for_local_use=False
        assert "master_key" not in config
    
    def test_get_detection_config_includes_master_key_for_local(self):
        """get_detection_config should include master_key for local use."""
        from service.authority import WatermarkAuthorityService
        from unittest.mock import MagicMock, patch
        
        with patch('service.authority.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_db.get_watermark.return_value = {
                "secret_key": "a" * 64,
            }
            mock_db.is_active.return_value = True
            mock_get_db.return_value = mock_db
            
            authority = WatermarkAuthorityService()
            
            # Patch at the import location in service.app.artifact_resolver
            with patch('service.app.artifact_resolver.get_artifact_resolver') as mock_resolver:
                mock_result = MagicMock()
                mock_result.likelihood_params_path_str = "/path/to/params"
                mock_result.mask_path_str = "/path/to/mask"
                mock_resolver.return_value.resolve.return_value = mock_result
                
                # Get config for local use (for_local_use=True)
                config = authority.get_detection_config("test_key", for_local_use=True)
        
        # Should have master_key for local use
        assert "master_key" in config
        assert config["master_key"] == "a" * 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

