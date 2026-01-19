"""
Secure metadata schema for watermark storage.

This module defines type-safe schemas for watermark metadata with explicit
security constraints. It ensures that only safe, minimal metadata is stored
while preventing accidental exposure of raw tensors or sensitive information.

Security Design:
    - zT_hash is stored (cryptographic hash, irreversible)
    - initial_latents are NEVER stored (raw tensor, privacy risk)
    - prompt text can be optionally hashed (privacy preservation)
    - All fields validated at serialization time

See: docs/SECURITY_ANALYSIS.md for full security analysis.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class MetadataVersion(str, Enum):
    """Metadata schema versions for backward compatibility."""
    V1_0 = "1.0"
    V1_1 = "1.1"


class MinimalWatermarkMetadata(BaseModel):
    """
    Minimal metadata schema for watermark detection.
    
    This schema contains the absolute minimum fields required for
    watermark detection. All other fields can be derived from these
    plus the secret key_master and algorithm configuration.
    
    Security Properties:
        - zT_hash: Cryptographic hash (SHA-256 truncated to 128 bits)
                   Cannot be inverted to recover original z_T tensor
        - sample_id: Arbitrary identifier, no privacy risk
        - strategy_version: Public algorithm version
    
    Size: ~100 bytes (compact JSON)
    """
    
    sample_id: str = Field(
        ...,
        description="Unique sample identifier for seed derivation",
        min_length=1,
        max_length=128,
    )
    
    zT_hash: str = Field(
        ...,
        description="SHA-256 hash of initial latent tensor (32 hex chars)",
        pattern=r"^[a-f0-9]{32}$",
    )
    
    strategy_version: MetadataVersion = Field(
        default=MetadataVersion.V1_0,
        description="Algorithm version for parameter lookup",
    )
    
    @field_validator("zT_hash")
    @classmethod
    def validate_zT_hash(cls, v: str) -> str:
        """Validate zT_hash is a valid 32-character hex string."""
        if len(v) != 32:
            raise ValueError(f"zT_hash must be 32 characters, got {len(v)}")
        try:
            int(v, 16)  # Validate hex
        except ValueError:
            raise ValueError("zT_hash must be a valid hexadecimal string")
        return v.lower()


class StandardWatermarkMetadata(MinimalWatermarkMetadata):
    """
    Standard metadata schema with recommended fields.
    
    Extends minimal schema with timestamp and model information
    for audit trails and compatibility verification.
    
    Security Properties:
        - All MinimalWatermarkMetadata properties
        - timestamp: Temporal info only, no privacy risk
        - model_id: Public model identifier, no privacy risk
    
    Size: ~200 bytes (compact JSON)
    """
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp (ISO8601 UTC)",
    )
    
    model_id: str = Field(
        default="stable-diffusion-v1-5",
        description="Diffusion model identifier",
        max_length=64,
    )
    
    def to_iso_timestamp(self) -> str:
        """Return timestamp as ISO8601 string."""
        return self.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


class ExtendedWatermarkMetadata(StandardWatermarkMetadata):
    """
    Extended metadata schema with full audit trail.
    
    Includes additional fields for comprehensive tracking and
    experiment management. All fields are security-safe.
    
    Security Properties:
        - All StandardWatermarkMetadata properties
        - key_id: Public key identifier (not the secret key)
        - experiment_id: Arbitrary identifier
        - prompt_hash: One-way hash of prompt text (privacy preserving)
        - generation_params: Public algorithm parameters
    
    Size: ~400 bytes (compact JSON)
    """
    
    key_id: Optional[str] = Field(
        default=None,
        description="Public key identifier (NOT the secret key_master)",
        max_length=64,
    )
    
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment identifier for batch tracking",
        max_length=64,
    )
    
    prompt_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of prompt text (16 hex chars)",
        pattern=r"^[a-f0-9]{16}$|^$",
    )
    
    generation_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Public generation parameters (guidance_scale, steps)",
    )
    
    @classmethod
    def hash_prompt(cls, prompt: str) -> str:
        """
        Hash prompt text for privacy-preserving storage.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            16-character hex hash (64 bits, sufficient for deduplication)
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Security Validation
# ============================================================================


class SecureMetadataValidator:
    """
    Validates metadata for security compliance.
    
    Ensures that stored metadata does not contain:
    - Raw tensor data (initial_latents, intermediate_latents, G_t)
    - Secret keys (key_master)
    - Privacy-sensitive content (raw prompts without hashing)
    """
    
    # Fields that MUST NOT be stored
    FORBIDDEN_FIELDS = {
        "initial_latents",
        "latents",
        "intermediate_latents",
        "g_schedule",
        "G_t",
        "noise_pred",
        "key_master",
        "key_secret",
        "alpha_schedule",  # Can be derived from config
        "g_field_tensors",
    }
    
    # Fields that should be hashed if stored
    HASH_RECOMMENDED_FIELDS = {
        "prompt",
        "negative_prompt",
    }
    
    @classmethod
    def validate(cls, metadata: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate metadata for security compliance.
        
        Args:
            metadata: Dictionary of metadata to validate
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check for forbidden fields
        for field in cls.FORBIDDEN_FIELDS:
            if field in metadata:
                violations.append(f"FORBIDDEN: '{field}' must not be stored")
        
        # Check for unhashed sensitive fields
        for field in cls.HASH_RECOMMENDED_FIELDS:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str) and len(value) > 32:
                    # Likely raw text, not a hash
                    violations.append(
                        f"PRIVACY_RISK: '{field}' appears to be raw text, "
                        f"consider storing '{field}_hash' instead"
                    )
        
        # Check for tensor-like data
        for key, value in metadata.items():
            if cls._looks_like_tensor(value):
                violations.append(
                    f"TENSOR_DATA: '{key}' appears to contain tensor data, "
                    f"which must not be stored"
                )
        
        return len(violations) == 0, violations
    
    @classmethod
    def _looks_like_tensor(cls, value: Any) -> bool:
        """Check if a value appears to be tensor data."""
        # Check for numpy arrays
        if hasattr(value, "tobytes") or hasattr(value, "numpy"):
            return True
        
        # Check for large nested lists (likely tensor data)
        if isinstance(value, list):
            if len(value) > 100:  # Large flat list
                return True
            if len(value) > 0 and isinstance(value[0], list):
                # Nested list, likely 2D+ tensor
                if len(value) * len(value[0]) > 1000:
                    return True
        
        return False
    
    @classmethod
    def sanitize(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove forbidden fields and return sanitized metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Sanitized metadata with forbidden fields removed
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # Skip forbidden fields
            if key in cls.FORBIDDEN_FIELDS:
                continue
            
            # Skip tensor-like data
            if cls._looks_like_tensor(value):
                continue
            
            # Hash sensitive text fields
            if key in cls.HASH_RECOMMENDED_FIELDS:
                if isinstance(value, str) and len(value) > 32:
                    sanitized[f"{key}_hash"] = hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest()[:16]
                    continue
            
            sanitized[key] = value
        
        return sanitized


# ============================================================================
# Factory Functions
# ============================================================================


def create_metadata_from_generation_result(
    result: Dict[str, Any],
    schema: str = "standard",
) -> MinimalWatermarkMetadata:
    """
    Create secure metadata from generation result.
    
    This function extracts only safe fields from the full generation
    result, ensuring no raw tensors or sensitive data are included.
    
    Args:
        result: Full result from generate_with_watermark()
        schema: "minimal", "standard", or "extended"
        
    Returns:
        Appropriate metadata schema instance
        
    Raises:
        ValueError: If required fields are missing
    """
    # Extract safe fields from result
    metadata = result.get("metadata", {})
    
    # Required fields
    sample_id = metadata.get("sample_id") or result.get("sample_id")
    zT_hash = result.get("zT_hash") or metadata.get("zT_hash")
    
    if not sample_id:
        raise ValueError("sample_id is required but not found in result")
    if not zT_hash:
        raise ValueError("zT_hash is required but not found in result")
    
    # Build base kwargs
    kwargs = {
        "sample_id": sample_id,
        "zT_hash": zT_hash,
    }
    
    if schema == "minimal":
        return MinimalWatermarkMetadata(**kwargs)
    
    # Standard schema adds timestamp and model_id
    kwargs["timestamp"] = datetime.now(timezone.utc)
    kwargs["model_id"] = metadata.get("model_id", "stable-diffusion-v1-5")
    
    if schema == "standard":
        return StandardWatermarkMetadata(**kwargs)
    
    # Extended schema adds more fields
    kwargs["key_id"] = metadata.get("key_id") or metadata.get("key_info", {}).get("key_id")
    kwargs["experiment_id"] = metadata.get("experiment_id") or metadata.get("key_info", {}).get("experiment_id")
    
    # Hash prompt if present
    prompt = metadata.get("prompt")
    if prompt:
        kwargs["prompt_hash"] = ExtendedWatermarkMetadata.hash_prompt(prompt)
    
    # Extract generation params
    gen_params = {}
    if "guidance_scale" in metadata:
        gen_params["guidance_scale"] = metadata["guidance_scale"]
    if "num_inference_steps" in metadata:
        gen_params["num_inference_steps"] = metadata["num_inference_steps"]
    if gen_params:
        kwargs["generation_params"] = gen_params
    
    return ExtendedWatermarkMetadata(**kwargs)


def serialize_metadata(
    metadata: MinimalWatermarkMetadata,
    format: str = "json",
) -> str | bytes:
    """
    Serialize metadata to storage format.
    
    Args:
        metadata: Metadata instance to serialize
        format: "json" (default), "json_compact", or "cbor"
        
    Returns:
        Serialized metadata string or bytes
    """
    # Convert to dict, handling datetime serialization
    data = metadata.model_dump()
    
    # Convert datetime to ISO string
    if "timestamp" in data and isinstance(data["timestamp"], datetime):
        data["timestamp"] = data["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Convert enum to value
    if "strategy_version" in data:
        data["strategy_version"] = str(data["strategy_version"].value)
    
    if format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    
    elif format == "json_compact":
        return json.dumps(data, separators=(",", ":"))
    
    elif format == "cbor":
        try:
            import cbor2
            return cbor2.dumps(data)
        except ImportError:
            raise ImportError("cbor2 package required for CBOR serialization")
    
    else:
        raise ValueError(f"Unknown format: {format}")


def deserialize_metadata(
    data: str | bytes,
    format: str = "json",
    schema: str = "minimal",
) -> MinimalWatermarkMetadata:
    """
    Deserialize metadata from storage format.
    
    Args:
        data: Serialized metadata
        format: "json" or "cbor"
        schema: "minimal", "standard", or "extended"
        
    Returns:
        Metadata instance
    """
    if format in ("json", "json_compact"):
        parsed = json.loads(data)
    elif format == "cbor":
        try:
            import cbor2
            parsed = cbor2.loads(data)
        except ImportError:
            raise ImportError("cbor2 package required for CBOR deserialization")
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Parse timestamp if present
    if "timestamp" in parsed and isinstance(parsed["timestamp"], str):
        parsed["timestamp"] = datetime.fromisoformat(
            parsed["timestamp"].replace("Z", "+00:00")
        )
    
    # Select schema class
    if schema == "minimal":
        return MinimalWatermarkMetadata(**parsed)
    elif schema == "standard":
        return StandardWatermarkMetadata(**parsed)
    elif schema == "extended":
        return ExtendedWatermarkMetadata(**parsed)
    else:
        raise ValueError(f"Unknown schema: {schema}")

