"""
Watermark Authority Service.

This service is the cryptographic and statistical authority of the system.
It owns:
- master_key (never exposed to clients)
- key_id → watermark policy mapping
- Embedding configuration
- Detection configuration (Bayesian parameters)
- Calibration version

All watermark decisions flow from this authority.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from service.infra.db import get_db

from service.infra.security import generate_watermark_id, generate_master_key

logger = logging.getLogger(__name__)


class WatermarkAuthorityService:
    """
    Watermark authority service.
    
    This service is the root of trust for watermarking:
    - Owns master_key (never exposed)
    - Resolves key_id → watermark policy
    - Decides embedding and detection configuration
    - Produces watermark payloads for generation
    - Produces detection configuration for detection
    """
    
    def __init__(self):
        """Initialize watermark authority service."""
        self.db = get_db()
    
    @staticmethod
    def _compute_policy_version(
        embedding_config: Dict[str, Any],
        g_field_config: Dict[str, Any],
        detection_config: Dict[str, Any],
    ) -> str:
        """
        Compute deterministic policy version from statistical parameters.
        
        policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        It is a deterministic hash of:
        - embedding_config (watermark embedding parameters)
        - g_field_config (G-field generation parameters)
        - detection_config (Bayesian detector parameters: threshold, prior, likelihood config)
        
        It does NOT depend on:
        - key_id (key-specific, not policy-specific)
        - user input (runtime values)
        - master_key (cryptographic, not statistical)
        
        Args:
            embedding_config: Embedding configuration dict
            g_field_config: G-field configuration dict
            detection_config: Detection configuration dict (Bayesian parameters)
        
        Returns:
            Deterministic policy version string (SHA-256 hash, truncated to 16 chars for readability)
        """
        # Create a stable, sorted representation of all statistical parameters
        # Sort keys to ensure deterministic ordering
        policy_dict = {
            "embedding_config": dict(sorted(embedding_config.items())),
            "g_field_config": dict(sorted(g_field_config.items())),
            # Only include statistical parameters from detection_config, not runtime values
            "detection_config": {
                "detector_type": detection_config.get("detector_type"),
                "threshold": detection_config.get("threshold"),
                "prior_watermarked": detection_config.get("prior_watermarked"),
                "likelihood_params_path": detection_config.get("likelihood_params_path"),
            },
        }
        
        # Serialize to JSON with sorted keys for deterministic hashing
        policy_json = json.dumps(policy_dict, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(policy_json.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Truncate to 16 characters for readability (still provides 64 bits of entropy)
        policy_version = hash_hex[:16]
        
        return policy_version
    
    def create_watermark_policy(
        self,
        key_id: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new watermark policy.
        
        This generates a new master_key and key_id, and stores the policy.
        
        Args:
            key_id: Optional key identifier (if None, generates new one)
            embedding_config: Optional embedding configuration override
            detection_config: Optional detection configuration override
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - watermark_version: Policy version
                - embedding_config: Embedding configuration
                - detection_config: Detection configuration
        """
        # Generate key_id and master_key
        if key_id is None:
            key_id = generate_watermark_id()
        master_key = generate_master_key()
        
        # Default embedding config (seed-bias for Phase-1)
        default_embedding_config = {
            "lambda_strength": 0.05,
            "domain": "frequency",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
        }
        embedding_config = embedding_config or default_embedding_config
        
        # Default detection config (Bayesian for Phase-1)
        default_detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        detection_config = detection_config or default_detection_config
        
        # G-field config must match generation config (for policy version computation)
        g_field_config = {
            "mapping_mode": "binary",
            "domain": embedding_config.get("domain", "frequency"),
            "frequency_mode": "bandpass",
            "low_freq_cutoff": embedding_config.get("low_freq_cutoff", 0.05),
            "high_freq_cutoff": embedding_config.get("high_freq_cutoff", 0.4),
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        watermark_version = self._compute_policy_version(
            embedding_config,
            g_field_config,
            detection_config,
        )
        
        # Store in database
        self.db.create_watermark(
            watermark_id=key_id,
            secret_key=master_key,
            model="stable-diffusion-v1-5",  # Phase-1: SD only
            strategy="seed_bias",
        )
        
        logger.info(f"Created watermark policy: key_id={key_id}")
        
        return {
            "key_id": key_id,
            "watermark_version": watermark_version,
            "embedding_config": embedding_config,
            "detection_config": detection_config,
        }
    
    def get_watermark_payload(
        self,
        key_id: str,
    ) -> Dict[str, Any]:
        """
        Get watermark payload for generation.
        
        This returns the configuration needed by GenerationAdapter to embed
        the watermark. The master_key is included (but never exposed to clients).
        
        Args:
            key_id: Public key identifier
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - master_key: Master key (for adapter use only)
                - embedding_config: Embedding configuration
                - watermark_version: Policy version
        
        Raises:
            ValueError: If key_id not found or revoked
        """
        # Load watermark record
        record = self.db.get_watermark(key_id)
        if record is None:
            raise ValueError(f"Watermark {key_id} not found")
        
        if not self.db.is_active(key_id):
            raise ValueError(f"Watermark {key_id} is revoked")
        
        master_key = record["secret_key"]
        
        # Get policy (for now, use defaults; in future, store in DB)
        # TODO: Store policy in database for versioning
        embedding_config = {
            "lambda_strength": 0.05,
            "domain": "frequency",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
        }
        
        # G-field config must match generation config (for policy version computation)
        g_field_config = {
            "mapping_mode": "binary",
            "domain": embedding_config["domain"],
            "frequency_mode": "bandpass",
            "low_freq_cutoff": embedding_config["low_freq_cutoff"],
            "high_freq_cutoff": embedding_config["high_freq_cutoff"],
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Default detection config (for policy version computation)
        detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        watermark_version = self._compute_policy_version(
            embedding_config,
            g_field_config,
            detection_config,
        )
        
        return {
            "key_id": key_id,
            "master_key": master_key,  # Internal use only
            "embedding_config": embedding_config,
            "watermark_version": watermark_version,
        }
    
    def get_detection_config(
        self,
        key_id: str,
    ) -> Dict[str, Any]:
        """
        Get detection configuration for detection.
        
        This returns the configuration needed by DetectionService to detect
        the watermark. The master_key is included (but never exposed to clients).
        
        Args:
            key_id: Public key identifier
        
        Returns:
            Dictionary with:
                - key_id: Public key identifier
                - master_key: Master key (for detector use only)
                - detection_config: Detection configuration (Bayesian parameters)
                - watermark_version: Policy version
                - g_field_config: G-field configuration (must match generation)
        
        Raises:
            ValueError: If key_id not found or revoked
        """
        # Load watermark record
        record = self.db.get_watermark(key_id)
        if record is None:
            raise ValueError(f"Watermark {key_id} not found")
        
        if not self.db.is_active(key_id):
            raise ValueError(f"Watermark {key_id} is revoked")
        
        master_key = record["secret_key"]
        
        # Get policy (for now, use defaults; in future, store in DB)
        # TODO: Store policy in database for versioning
        detection_config = {
            "detector_type": "bayesian",
            "likelihood_params_path": None,  # TODO: Load from calibration
            "threshold": 0.5,
            "prior_watermarked": 0.5,
        }
        
        # G-field config must match generation config
        g_field_config = {
            "mapping_mode": "binary",
            "domain": "frequency",
            "frequency_mode": "bandpass",
            "low_freq_cutoff": 0.05,
            "high_freq_cutoff": 0.4,
            "normalize_zero_mean": True,
            "normalize_unit_variance": True,
        }
        
        # Inversion parameters are part of watermark policy, not user preference.
        # These must match research calibration assumptions to ensure accurate detection.
        # DetectionService will enforce these parameters and reject mismatched user input.
        # 
        # CRITICAL: These parameters must remain aligned with research calibration assumptions in src/.
        # Research layer requires guidance_scale=1.0 for DDIM inversion (see src/detection/inversion.py).
        # This ensures detection reliability and Bayesian calibration correctness.
        inversion_config = {
            "num_inference_steps": 50,
            "guidance_scale": 1.0,  # Must be 1.0 for DDIM inversion correctness (research requirement)
            "prompt_required": False,  # Prompt is optional but recommended for accuracy
        }
        
        # Calibration is scoped per (model_family, embedding_schema_version).
        # For Phase-1: stable-diffusion-v1-5 with seed_bias_v1
        # This will later be resolved dynamically.
        model_family = "stable-diffusion-v1-5"
        embedding_schema_version = "seed_bias_v1"
        
        # Get resolved paths from startup validation cache
        # Use centralized artifact resolver (validated at startup)
        # This provides a single source of truth for artifact paths
        # NO path guessing or fallback searching - paths come from env vars only
        from service.app.artifact_resolver import get_artifact_resolver
        
        resolver = get_artifact_resolver()
        artifact_result = resolver.resolve()
        
        # Get resolved paths from centralized resolver
        calibration_path = artifact_result.likelihood_params_path_str
        mask_path = artifact_result.mask_path_str
        
        # Update detection_config with resolved paths
        detection_config["likelihood_params_path"] = calibration_path
        detection_config["mask_path"] = mask_path
        
        # Log resolved paths for debugging
        logger.debug(
            f"Detection config paths resolved: "
            f"likelihood_params_path={calibration_path}, "
            f"mask_path={mask_path}"
        )
        
        # Compute deterministic policy version from statistical parameters
        # policy_version uniquely identifies the statistical assumptions used for detection and calibration.
        # This ensures results are reproducible, auditable, and scientifically defensible.
        watermark_version = self._compute_policy_version(
            embedding_config={
                "lambda_strength": 0.05,  # Match generation defaults
                "domain": "frequency",
                "low_freq_cutoff": 0.05,
                "high_freq_cutoff": 0.4,
            },
            g_field_config=g_field_config,
            detection_config=detection_config,
        )
        
        return {
            "key_id": key_id,
            "master_key": master_key,  # Internal use only
            "detection_config": detection_config,
            "g_field_config": g_field_config,
            "inversion": inversion_config,
            "watermark_version": watermark_version,
        }
    
    def revoke_watermark(self, key_id: str) -> bool:
        """
        Revoke a watermark (mark as inactive).
        
        Args:
            key_id: Public key identifier
        
        Returns:
            True if revoked, False if not found
        """
        return self.db.revoke_watermark(key_id)

