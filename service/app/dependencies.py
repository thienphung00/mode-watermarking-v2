"""
Shared dependencies for FastAPI routes.

Provides dependency injection for:
- WatermarkAuthorityService
- GenerationAdapter (Phase-1: Stable Diffusion only)
- DetectionService (Bayesian-only)

No research-layer instantiation here.
No SD-specific logic here.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from service.authority import WatermarkAuthorityService
from service.generation import GenerationAdapter, StableDiffusionSeedBiasAdapter
from service.detection import DetectionService
from service.app.artifact_resolver import get_artifact_resolver

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """
    Detect available device with priority: mps > cuda > cpu.
    
    Returns:
        Device string ("mps", "cuda", or "cpu")
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# Global service caches
_authority_service: Optional[WatermarkAuthorityService] = None
_generation_adapter: Optional[GenerationAdapter] = None
_pipeline_cache: Optional[StableDiffusionPipeline] = None
_pipeline_initialized: bool = False  # Track if pipeline was preloaded at startup
_detection_service: Optional[DetectionService] = None


def get_watermark_authority() -> WatermarkAuthorityService:
    """
    Get WatermarkAuthorityService singleton.
    
    Returns:
        WatermarkAuthorityService instance
    """
    global _authority_service
    
    if _authority_service is None:
        _authority_service = WatermarkAuthorityService()
        logger.info("WatermarkAuthorityService initialized")
    
    return _authority_service


def preload_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None) -> None:
    """
    Preload Stable Diffusion pipeline at startup.
    
    This function should be called during FastAPI startup to ensure the pipeline
    is loaded once and frozen. This eliminates first-request latency spikes and
    prevents accidental re-initialization.
    
    Args:
        model_id: Hugging Face model identifier
        device: Device to load on (default: auto-detect with priority: mps > cuda > cpu)
    
    Raises:
        RuntimeError: If pipeline is already initialized
    """
    global _pipeline_cache, _pipeline_initialized
    
    if _pipeline_initialized:
        if _pipeline_cache is None:
            raise RuntimeError("Pipeline initialization state is inconsistent")
        logger.warning("Pipeline already preloaded, skipping")
        return
    
    if device is None:
        device = detect_device()
    
    logger.info(f"Preloading Stable Diffusion pipeline: {model_id} on {device}")
    
    # Load pipeline without dtype (dtype will be set via .to(device))
    # This avoids passing dtype to constructor which causes warnings
    _pipeline_cache = StableDiffusionPipeline.from_pretrained(model_id)
    
    # Move to device and set appropriate dtype
    # MPS and CPU use float32, CUDA can use float16 for performance
    if device == "cuda":
        _pipeline_cache = _pipeline_cache.to(device, dtype=torch.float16)
    else:
        _pipeline_cache = _pipeline_cache.to(device, dtype=torch.float32)
    
    _pipeline_initialized = True
    logger.info("Pipeline preloaded successfully at startup")


def get_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None) -> StableDiffusionPipeline:
    """
    Get preloaded Stable Diffusion pipeline (singleton).
    
    This function returns the pipeline that was preloaded at startup.
    It never initializes the pipeline itself to prevent hidden reloads.
    
    This is used internally by GenerationAdapter and DetectionService.
    FastAPI routes should not call this directly.
    
    Args:
        model_id: Hugging Face model identifier (must match preloaded model)
        device: Device string (must match preloaded device, ignored if already set)
    
    Returns:
        StableDiffusionPipeline instance
    
    Raises:
        RuntimeError: If pipeline was not preloaded at startup
    """
    global _pipeline_cache, _pipeline_initialized
    
    if not _pipeline_initialized or _pipeline_cache is None:
        raise RuntimeError(
            "Pipeline was not preloaded at startup. "
            "Ensure preload_pipeline() is called during FastAPI startup."
        )
    
    # Validate model_id matches (optional check, but helpful for debugging)
    if model_id != "runwayml/stable-diffusion-v1-5":
        logger.warning(
            f"Requested model_id={model_id} but preloaded model is "
            f"runwayml/stable-diffusion-v1-5. Returning preloaded pipeline."
        )
    
    return _pipeline_cache


def get_generation_adapter() -> GenerationAdapter:
    """
    Get GenerationAdapter singleton (Phase-1: Stable Diffusion only).
    
    TODO (Phase-2): This adapter can be removed when moving to client-side generation.
    The API will only issue watermark credentials, not perform generation.
    
    Returns:
        GenerationAdapter instance (currently StableDiffusionSeedBiasAdapter)
    """
    global _generation_adapter
    
    if _generation_adapter is None:
        # Phase-1: Stable Diffusion only
        _generation_adapter = StableDiffusionSeedBiasAdapter(
            model_id="runwayml/stable-diffusion-v1-5",
            device=None,  # Auto-detect
            use_fp16=True,
        )
        logger.info("GenerationAdapter initialized (Stable Diffusion)")
    
    return _generation_adapter


def get_detection_service() -> DetectionService:
    """
    Get DetectionService singleton (Bayesian-only).
    
    ⚠️ DETECTION LAYER ⚠️
    Detection uses detector_type="bayesian" only.
    Legacy detection modes (fast_only/hybrid/full_inversion) are deprecated.
    
    Returns:
        DetectionService instance
    
    Raises:
        RuntimeError: If detection artifacts are not available
    """
    global _detection_service
    
    if _detection_service is None:
        # Check artifact availability before creating service
        resolver = get_artifact_resolver()
        artifact_result = resolver.resolve()
        
        if not artifact_result.is_available:
            raise RuntimeError(
                f"Detection artifacts not available: {artifact_result.error_message}\n"
                f"Detection service cannot be initialized without valid artifacts."
            )
        
        authority = get_watermark_authority()
        pipeline = get_pipeline()
        _detection_service = DetectionService(
            authority=authority,
            pipeline=pipeline,
        )
        # Logging happens in DetectionService.__init__
    
    return _detection_service


def is_detection_available() -> bool:
    """
    Check if detection artifacts are available.
    
    Returns:
        True if detection artifacts are available, False otherwise
    """
    resolver = get_artifact_resolver()
    artifact_result = resolver.resolve()
    return artifact_result.is_available


def get_detection_availability_status() -> dict:
    """
    Get detailed detection artifact availability status.
    
    Returns:
        Dictionary with availability status and paths
    """
    resolver = get_artifact_resolver()
    return resolver.get_availability_status()

