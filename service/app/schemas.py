"""
Pydantic schemas for API requests and responses.

Product-level schemas only. No research internals exposed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Generate Schemas
# ============================================================================


class GenerateRequest(BaseModel):
    """Request schema for POST /generate endpoint."""
    
    prompt: str = Field(..., description="Text prompt for image generation")
    key_id: Optional[str] = Field(None, description="Optional key identifier (if None, creates new)")
    num_inference_steps: Optional[int] = Field(50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(1.0, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed (optional)")
    height: Optional[int] = Field(None, description="Image height (optional, must be > 0 if provided)")
    width: Optional[int] = Field(None, description="Image width (optional, must be > 0 if provided)")
    
    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: Optional[int]) -> Optional[int]:
        """Validate that height and width are > 0 if provided."""
        if v is not None and v <= 0:
            raise ValueError("Height and width must be greater than 0 if provided")
        return v


class GenerationMetadata(BaseModel):
    """Generation metadata (product-level, no research internals)."""
    
    seed: Optional[int] = Field(None, description="Random seed used")
    num_inference_steps: int = Field(..., description="Number of inference steps")
    guidance_scale: float = Field(..., description="Guidance scale")
    model_version: str = Field(..., description="Model version identifier")
    height: Optional[int] = Field(None, description="Image height")
    width: Optional[int] = Field(None, description="Image width")


class GenerateResponse(BaseModel):
    """Response schema for POST /generate endpoint."""
    
    image_url: Optional[str] = Field(None, description="URL to generated image (if stored)")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (if returned directly)")
    key_id: str = Field(..., description="Watermark key identifier")
    generation_metadata: GenerationMetadata = Field(..., description="Generation metadata")
    watermark_version: str = Field(..., description="Watermark policy version")


# ============================================================================
# Detect Schemas
# ============================================================================


class DetectRequest(BaseModel):
    """Request schema for POST /detect endpoint."""
    
    image_base64: str = Field(..., description="Base64-encoded image")
    key_id: str = Field(..., description="Watermark key identifier")
    
    # NOTE: Detection is prompt-agnostic. The API does not require or trust prompts.
    # Detection uses unconditional DDIM inversion (prompt="", guidance_scale=1.0).


class DetectResponse(BaseModel):
    """Response schema for POST /detect endpoint."""
    
    detected: bool = Field(..., description="Whether watermark was detected")
    score: float = Field(..., description="Detection score (log-odds, higher = more confident)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (posterior probability, 0-1)")
    policy_version: str = Field(..., description="Watermark policy version used")
    posterior: float = Field(..., ge=0.0, le=1.0, description="Posterior probability P(watermarked | g)")
    log_odds: float = Field(..., description="Log-odds ratio")
    is_watermarked: bool = Field(..., description="Detection decision (same as detected)")
    watermark_version: str = Field(..., description="Watermark policy version (same as policy_version)")
    g_field_config_hash: Optional[str] = Field(None, description="G-field config hash for validation")


# ============================================================================
# Evaluation Schemas
# ============================================================================


class ImperceptibilityEvalRequest(BaseModel):
    """Request schema for POST /evaluate/imperceptibility endpoint."""
    
    prompt: str = Field(..., description="Text prompt for image generation")
    seed: int = Field(..., description="Random seed (required for determinism)")
    num_inference_steps: Optional[int] = Field(default=50, ge=1, le=100, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(default=1.0, ge=1.0, le=20.0, description="Guidance scale")
    height: Optional[int] = Field(default=512, ge=1, description="Image height")
    width: Optional[int] = Field(default=512, ge=1, description="Image width")
    
    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: Optional[int]) -> Optional[int]:
        """Validate that height and width are > 0 if provided."""
        if v is not None and v <= 0:
            raise ValueError("Height and width must be greater than 0 if provided")
        return v


class DifferenceMetrics(BaseModel):
    """Difference metrics between baseline and watermarked images."""
    
    l2: float = Field(..., description="Normalized L2 distance (pixel space)")
    psnr: float = Field(..., description="Peak Signal-to-Noise Ratio (dB)")
    ssim: float = Field(..., ge=0.0, le=1.0, description="Structural Similarity Index (0-1)")


class ModelInfo(BaseModel):
    """Model information for evaluation response."""
    
    model_family: str = Field(..., description="Model family identifier")
    num_inference_steps: int = Field(..., description="Number of inference steps used")
    guidance_scale: float = Field(..., description="Guidance scale used")


class ImperceptibilityEvalResponse(BaseModel):
    """
    Response schema for POST /evaluate/imperceptibility endpoint.
    
    ⚠️ EVALUATION-ONLY ⚠️
    This endpoint is not part of the production watermarking system.
    Outputs must not be used for detection benchmarking.
    """
    
    baseline_image_base64: str = Field(..., description="Base64-encoded baseline (unwatermarked) image")
    watermarked_image_base64: str = Field(..., description="Base64-encoded watermarked image (evaluation-only)")
    difference_metrics: DifferenceMetrics = Field(..., description="Difference metrics between images")
    model_info: ModelInfo = Field(..., description="Model and generation parameters")
    disclaimer: str = Field(
        default="Evaluation-only output. Not valid for production use or detection.",
        description="Disclaimer about evaluation-only nature of output"
    )


# ============================================================================
# Health Check Schema
# ============================================================================


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    
    status: str = Field("ok", description="Service status")

