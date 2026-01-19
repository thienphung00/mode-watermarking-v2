"""
Route for generating watermarked images.

POST /generate

Phase-1: Hosted generation using Stable Diffusion
Phase-2: Client-side generation (this endpoint will be deprecated)
"""
from __future__ import annotations

import base64
import io
import logging

from fastapi import APIRouter, HTTPException

from service.app.schemas import GenerateRequest, GenerateResponse
from service.app.dependencies import (
    get_watermark_authority,
    get_generation_adapter,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a watermarked image.
    
    Phase-1: Hosted generation using Stable Diffusion with seed-bias watermarking.
    
    TODO (Phase-2): This endpoint will be deprecated when moving to client-side generation.
    The API will only issue watermark credentials, not perform generation.
    
    This endpoint:
    1. Creates or retrieves watermark policy from WatermarkAuthorityService
    2. Uses GenerationAdapter to generate watermarked image
    3. Returns image and metadata
    
    Args:
        request: GenerateRequest with prompt, optional key_id, and generation parameters
    
    Returns:
        GenerateResponse with image, key_id, generation metadata, and watermark version
    """
    try:
        # Step 1: Get services
        authority = get_watermark_authority()
        adapter = get_generation_adapter()
        
        # Step 2: Get or create watermark policy
        # Phase-1 only: Auto-create watermark policy when key_id is missing
        # In Phase-2, key_id will be required and issued via separate credential endpoint
        if request.key_id:
            # Use existing watermark
            try:
                watermark_payload = authority.get_watermark_payload(request.key_id)
                key_id = request.key_id
            except ValueError as e:
                # Improve error messaging for unknown key_id
                raise HTTPException(
                    status_code=404,
                    detail="Unknown watermark key_id. Create a watermark policy first or omit key_id to auto-generate one."
                )
        else:
            # Phase-1 only: Auto-create watermark policy when key_id is missing
            # This automatically generates a secure key_id and secret key internally,
            # persists the policy in the database, and uses it for generation.
            # The newly created key_id is returned in the response.
            policy = authority.create_watermark_policy()
            key_id = policy["key_id"]
            watermark_payload = authority.get_watermark_payload(key_id)
        
        logger.info(f"Generating image with key_id={key_id}")
        logger.info(f"API received prompt: '{request.prompt}'")
        
        # VERIFICATION: Seed randomization ensures different images per generation
        # Same key_id + different seeds = different images, but same g-values (deterministic watermark)
        # The watermark (G-field) is based ONLY on key_id, not seed
        
        # Step 3: Validate parameters against research assumptions
        # This check prevents silent distribution shift between research calibration and production usage.
        # Research layer canonical assumptions (must match src/core/config.py and detection requirements):
        RESEARCH_ASSUMPTIONS = {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "scheduler": "DDIM",
            "prediction_type": "epsilon",
            "trained_timesteps": 1000,
            # Note: inference_timesteps and guidance_scale may vary for generation quality,
            # but detection inversion requires guidance_scale=1.0 (checked in detection service)
        }
        
        # Get adapter model info to verify model_id
        adapter_model_info = adapter.get_model_info()
        requested_model_id = adapter_model_info.get("model_id", "unknown")
        
        # Critical check: model_id must match research assumption
        if requested_model_id != RESEARCH_ASSUMPTIONS["model_id"]:
            error_msg = (
                f"Model mismatch: API uses model_id='{requested_model_id}' but research calibration "
                f"assumes '{RESEARCH_ASSUMPTIONS['model_id']}'. Detection reliability cannot be guaranteed."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Get generation parameters (use defaults if not provided)
        final_guidance_scale = request.guidance_scale or 1.0
        final_num_steps = request.num_inference_steps or 50
        
        # CRITICAL: Randomize seed if not provided to ensure different images per generation
        # Same key_id should yield different images but same g-values (deterministic watermark)
        import secrets
        final_seed = request.seed
        if final_seed is None:
            final_seed = secrets.randbelow(2**31)  # Random seed in [0, 2^31)
            logger.info(f"Seed not provided, randomized to: {final_seed}")
        else:
            logger.info(f"Using provided seed: {final_seed}")
        
        # Soft check: Log warning if guidance_scale differs from detection requirement
        # Detection inversion requires guidance_scale=1.0
        if final_guidance_scale != 1.0:
            logger.warning(
                f"Generation using guidance_scale={final_guidance_scale} but detection inversion requires 1.0. "
                f"API value={final_guidance_scale}, research detection requirement=1.0. "
                f"Detection will use guidance_scale=1.0 for inversion (required for mathematical correctness)."
            )
        
        # Soft check: Log warning if num_inference_steps differs significantly from common research values
        if final_num_steps not in [25, 50]:
            logger.warning(
                f"Generation using num_inference_steps={final_num_steps} (API value={final_num_steps}, "
                f"common research values=25 or 50). Detection must use matching num_inference_steps for accurate inversion."
            )
        
        # Step 4: Generate image using adapter
        logger.info(f"Adapter being called with prompt: '{request.prompt}'")
        result = adapter.generate(
            prompt=request.prompt,
            watermark_payload=watermark_payload,
            num_inference_steps=final_num_steps,
            guidance_scale=final_guidance_scale,
            seed=final_seed,
            height=request.height,
            width=request.width,
        )
        logger.info(f"Adapter completed generation for prompt: '{request.prompt}'")
        
        # Step 4: Encode image to base64
        image = result["image"]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Step 5: Build response
        generation_metadata = result["generation_metadata"]
        
        from service.app.schemas import GenerationMetadata
        
        metadata = GenerationMetadata(
            seed=generation_metadata.get("seed"),
            num_inference_steps=generation_metadata["num_inference_steps"],
            guidance_scale=generation_metadata["guidance_scale"],
            model_version=generation_metadata["model_version"],
            height=generation_metadata.get("height"),
            width=generation_metadata.get("width"),
        )
        
        logger.info(f"Generated image successfully: key_id={key_id}")
        
        return GenerateResponse(
            image_base64=image_base64,
            key_id=key_id,
            generation_metadata=metadata,
            watermark_version=result["watermark_version"],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

