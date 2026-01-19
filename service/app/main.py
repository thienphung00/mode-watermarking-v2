"""
FastAPI application entrypoint.

Run with:
    uvicorn service.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

from .routes import generate, detect, health, evaluate, demo
from .middleware import RateLimitMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.
    
    Loads models and configs at startup.
    Validates artifact paths and consistency (config hash, mask shape).
    Preloads Stable Diffusion pipeline to eliminate first-request latency.
    """
    # Startup
    logger.info("Starting watermarking service...")
    
    # Preload Stable Diffusion pipeline at startup
    # This eliminates first-request latency spikes and prevents accidental re-initialization
    from .dependencies import preload_pipeline
    try:
        preload_pipeline()
        logger.info("✓ Pipeline preloaded at startup")
    except Exception as e:
        logger.error(f"Failed to preload pipeline at startup: {e}", exc_info=True)
        raise RuntimeError(f"Pipeline preload failed: {e}") from e
    
    # Enable micro-batching for detection if artifacts are available
    # This improves GPU utilization by batching detection requests
    from .dependencies import get_detection_service, is_detection_available
    if is_detection_available():
        try:
            detection_service = get_detection_service()
            # Start worker asynchronously (lifespan is async context)
            await detection_service.enable_micro_batching(
                batch_window_ms=20.0,  # 20ms collection window
                max_batch_size=8,      # Max 8 requests per batch
            )
            logger.info("✓ Detection micro-batching enabled")
        except Exception as e:
            logger.warning(f"Failed to enable detection micro-batching: {e}")
            # Don't fail startup if micro-batching fails - detection will still work
    
    # Register test keys at startup if enabled
    # This makes key registration deterministic, version-controlled, and environment-aware
    # Controlled by REGISTER_TEST_KEYS environment variable (explicit opt-in)
    import os
    if os.getenv("REGISTER_TEST_KEYS") == "true":
        logger.info("REGISTER_TEST_KEYS enabled - registering test keys at startup")
        try:
            from .dependencies import get_watermark_authority
            authority = get_watermark_authority()
            key_id = "test_batch_001"
            
            # Check if key already exists (idempotent)
            # Try to get the key, but handle decryption errors gracefully
            # (keys encrypted with different encryption keys will fail to decrypt)
            try:
                existing_record = authority.db.get_watermark(key_id)
                if existing_record is not None:
                    logger.info(f"Key {key_id} already registered in authority registry. Skipping registration.")
                    # Key exists and can be decrypted, no action needed
                else:
                    # Key doesn't exist, register it
                    logger.info(f"Registering {key_id} with Authority Service...")
                    policy = authority.create_watermark_policy(key_id=key_id)
                    logger.info(
                        f"Successfully registered {key_id}: "
                        f"watermark_version={policy['watermark_version']}"
                    )
            except Exception as e:
                # If decryption fails (e.g., encryption key mismatch), check if key exists in DB
                # If it exists but can't be decrypted, we'll overwrite it with a new registration
                if key_id in authority.db._db:
                    logger.warning(
                        f"Key {key_id} exists in database but cannot be decrypted "
                        f"(encryption key mismatch). Will register new key."
                    )
                # Register key (will overwrite if exists, create if doesn't)
                logger.info(f"Registering {key_id} with Authority Service...")
                policy = authority.create_watermark_policy(key_id=key_id)
                logger.info(
                    f"Successfully registered {key_id}: "
                    f"watermark_version={policy['watermark_version']}"
                )
        except Exception as e:
            # Don't fail startup if registration fails - log and continue
            logger.warning(
                f"Failed to register test keys at startup: {e}. "
                f"Service will continue, but test keys may not be available."
            )
    
    # Resolve and validate artifact paths at startup using centralized resolver
    from .artifact_resolver import get_artifact_resolver
    
    # Log environment variable status at startup
    env_likelihood_path = os.getenv("LIKELIHOOD_PARAMS_PATH")
    logger.info(f"LIKELIHOOD_PARAMS_PATH environment variable: {env_likelihood_path if env_likelihood_path else 'NOT SET'}")
    
    # Use centralized artifact resolver (validates and caches at startup)
    resolver = get_artifact_resolver()
    artifact_result = resolver.resolve()
    
    if artifact_result.is_available:
        # Validate artifacts can be loaded (fail fast if invalid)
        logger.info("Validating detector artifacts at startup...")
        try:
            from .dependencies import get_watermark_authority
            from service.detector_artifacts import DetectorArtifacts
            
            authority = get_watermark_authority()
            
            # Create a test policy to get g_field_config
            test_policy = authority.create_watermark_policy()
            test_key_id = test_policy["key_id"]
            config = authority.get_detection_config(test_key_id)
            g_field_config = config["g_field_config"]
            
            # Try to load artifacts (this will validate consistency)
            # This will raise if artifacts are invalid
            artifacts = DetectorArtifacts(
                likelihood_params_path=artifact_result.likelihood_params_path_str,
                mask_path=artifact_result.mask_path_str,
                g_field_config=g_field_config,
            )
            
            logger.info(
                f"✓ Artifact validation passed: "
                f"num_positions={artifacts.num_positions}, "
                f"mask_shape={list(artifacts.mask.shape) if artifacts.mask is not None else None}"
            )
            
        except Exception as e:
            # Fail fast if artifacts are invalid
            logger.error(
                f"Artifact validation failed at startup: {e}\n"
                f"  This indicates the artifacts are misconfigured or invalid.\n"
                f"  Detection will not work correctly.\n"
                f"  Please check:\n"
                f"    - LIKELIHOOD_PARAMS_PATH points to a valid likelihood_params.json\n"
                f"    - MASK_PATH (if set) points to a valid mask tensor\n"
                f"    - Artifacts match the g-field configuration"
            )
            raise RuntimeError(
                f"Failed to validate detector artifacts at startup: {e}"
            ) from e
    else:
        # Artifacts not available - log warning but don't fail startup (demo mode may work)
        logger.warning(
            f"Detection artifacts not available at startup: {artifact_result.error_message}\n"
            f"  Detection endpoints will be unavailable until artifacts are configured.\n"
            f"  Demo endpoints will return clear error messages."
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down watermarking service...")
    
    # Disable micro-batching if it was enabled
    if is_detection_available():
        try:
            detection_service = get_detection_service()
            if detection_service._detection_worker is not None:
                await detection_service.disable_micro_batching()
                logger.info("✓ Detection micro-batching disabled")
        except Exception as e:
            logger.warning(f"Error disabling detection micro-batching during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Watermarking Service",
    description="Custodial API for watermark generation and detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(generate.router, prefix="/api/v1", tags=["generation"])
app.include_router(detect.router, prefix="/api/v1", tags=["detection"])
app.include_router(evaluate.router, prefix="/api/v1", tags=["evaluation"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(demo.router, prefix="/api/v1", tags=["demo"])

# Path to demo.html
DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """
    Serve the Phase-1 demo page at /demo (user-friendly URL).
    
    This is a presentation-only demo for non-technical users (e.g., recruiters).
    It wraps the existing API endpoints without modifying any watermark logic.
    """
    try:
        with open(DEMO_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Demo not found</h1><p>demo.html is missing from service/app/static/</p>",
            status_code=404
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Watermarking Service",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "detect": "/api/v1/detect",
            "evaluate": "/api/v1/evaluate/imperceptibility",
            "health": "/api/v1/health",
            "demo": "/demo",
            "demo_verify": "/api/v1/demo/verify",
        }
    }

