# Service Architecture and File Structure

This document describes the architecture and file structure of the `service/` directory and how it interfaces with the implementation in `src/`.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Service Components](#core-service-components)
4. [API Layer](#api-layer)
5. [Integration with `src/`](#integration-with-src)
6. [Request Flow](#request-flow)
7. [Shared Models and Configuration](#shared-models-and-configuration)
8. [Architectural Assumptions and Couplings](#architectural-assumptions-and-couplings)
9. [Extension Points](#extension-points)

---

## Overview

The `service/` directory implements a FastAPI-based REST API for watermark generation and detection. It acts as a **product layer** that wraps the research implementation in `src/` with:

- **API endpoints** for generation, detection, and evaluation
- **Service abstractions** that hide research internals from clients
- **Authority management** for cryptographic key management and policy enforcement
- **Artifact management** for detection model artifacts (likelihood parameters, masks)
- **Infrastructure** for database, security, and middleware

The service layer is designed to be **research-agnostic** at the API boundary while leveraging `src/` for core watermarking algorithms.

---

## Directory Structure

```
service/
├── app/                          # FastAPI application layer
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── dependencies.py           # Dependency injection (singletons)
│   ├── schemas.py                 # Pydantic request/response schemas
│   ├── middleware.py              # Rate limiting middleware
│   ├── artifact_resolver.py       # Centralized artifact path resolution
│   ├── artifact_paths.py          # Legacy artifact path utilities (deprecated)
│   ├── routes/                    # API route handlers
│   │   ├── __init__.py
│   │   ├── generate.py           # POST /api/v1/generate
│   │   ├── detect.py              # POST /api/v1/detect
│   │   ├── evaluate.py            # POST /api/v1/evaluate/imperceptibility
│   │   ├── health.py              # GET /api/v1/health
│   │   └── demo.py                # Demo endpoints
│   └── static/
│       └── demo.html              # Demo UI
├── authority.py                   # WatermarkAuthorityService (key management)
├── detection.py                   # DetectionService (Bayesian detection)
├── detector_artifacts.py          # DetectorArtifacts (artifact loader)
├── evaluation.py                  # ImperceptibilityEvaluationService
├── evaluation_metrics.py           # L2, PSNR, SSIM metrics
├── generation/                    # Generation adapters
│   ├── __init__.py
│   ├── base.py                    # GenerationAdapter abstract interface
│   └── stable_diffusion.py        # StableDiffusionSeedBiasAdapter
├── infra/                         # Infrastructure components
│   ├── __init__.py
│   ├── db.py                      # WatermarkKeyDB (in-memory key storage)
│   └── security.py                # Key generation and encryption
├── configs/
│   └── runtime.yaml               # Runtime configuration
├── Dockerfile                     # Containerization
├── README.md                      # Service documentation
├── ARCHITECTURE.md                # Existing architecture notes
└── STARTUP_OPTIMIZATION.md        # Startup optimization notes
```

---

## Core Service Components

### `service/app/main.py`

**Purpose**: FastAPI application entry point and lifecycle management.

**Responsibilities**:
- Creates FastAPI app instance with CORS and rate limiting middleware
- Registers routers for all API endpoints
- Manages application lifespan (startup/shutdown)
- Validates detection artifacts at startup (fail-fast if invalid)
- Serves demo HTML page at `/demo`

**Key Features**:
- **Lazy pipeline loading**: Stable Diffusion pipeline (~4GB) is loaded on first request, not at startup
- **Startup validation**: Artifacts are validated at startup using `ArtifactResolver` and `DetectorArtifacts`
- **Graceful degradation**: If artifacts are unavailable, detection endpoints return clear errors but service still starts

**Entry Points**:
- Root endpoint: `GET /` - Returns service info and available endpoints
- Demo endpoint: `GET /demo` - Serves demo HTML page

**Dependencies**:
- `service.app.routes.*` - All route modules
- `service.app.middleware.RateLimitMiddleware` - Rate limiting
- `service.app.artifact_resolver.get_artifact_resolver()` - Artifact resolution

---

### `service/authority.py`

**Purpose**: Cryptographic and statistical authority for watermarking system.

**Responsibilities**:
- **Master key management**: Generates and stores master keys (never exposed to clients)
- **Policy management**: Creates and retrieves watermark policies (key_id → policy mapping)
- **Policy versioning**: Computes deterministic `policy_version` hash from statistical parameters
- **Configuration resolution**: Provides embedding and detection configurations for generation and detection

**Key Classes**:
- `WatermarkAuthorityService`: Main authority service

**Key Methods**:
- `create_watermark_policy()`: Creates new watermark policy with generated key_id and master_key
- `get_watermark_payload()`: Returns watermark configuration for generation (includes master_key internally)
- `get_detection_config()`: Returns detection configuration (includes master_key, g_field_config, inversion params)
- `revoke_watermark()`: Revokes a watermark (marks as inactive)

**Policy Version Computation**:
- `policy_version` is a deterministic SHA-256 hash of:
  - `embedding_config` (lambda_strength, domain, frequency cutoffs)
  - `g_field_config` (mapping_mode, domain, frequency_mode, normalization)
  - `detection_config` (detector_type, threshold, prior, likelihood_params_path)
- Ensures reproducibility and auditability of watermark policies

**Dependencies**:
- `service.infra.db.get_db()` - Key storage
- `service.infra.security.generate_watermark_id()`, `generate_master_key()` - Key generation
- `service.app.artifact_resolver.get_artifact_resolver()` - Artifact path resolution

**Storage**:
- Uses `WatermarkKeyDB` (in-memory, JSON-backed) for key storage
- Keys are encrypted at rest using `KeyEncryption` (Fernet symmetric encryption)

---

### `service/detection.py`

**Purpose**: Detection service using Bayesian detector only.

**Responsibilities**:
- **Image → z_T inversion**: Encodes image to z_0, performs DDIM inversion to z_T
- **G-value computation**: Computes g-values from z_T using canonical `compute_g_values()` from `src/`
- **Bayesian detection**: Instantiates `BayesianDetector` from `src/models/detectors.py` and runs detection
- **Result formatting**: Returns product-level results (no research internals exposed)

**Key Classes**:
- `DetectionService`: Main detection service
- `StableDiffusionDetectionBackend`: SD-specific operations (VAE encoding, DDIM inversion)

**Key Methods**:
- `detect()`: Full detection pipeline (image → z_T → g-values → Bayesian detection)
- `detect_from_g_values()`: Detection from precomputed g-values (for testing/equivalence validation)

**Detection Flow**:
1. Fetch detection config from `WatermarkAuthorityService`
2. Load detector artifacts (`DetectorArtifacts`)
3. Encode image to z_0 (VAE encoding, SD-specific)
4. Perform prompt-free DDIM inversion to z_T (unconditional: `prompt=""`, `guidance_scale=1.0`)
5. Compute g-values from z_T using `src.detection.g_values.compute_g_values()`
6. Apply mask + binarization (exact same logic as research scripts)
7. Instantiate `BayesianDetector` and run detection
8. Return product-level results

**Critical Assumptions**:
- **Stable Diffusion only**: Detection assumes SD VAE encoding and DDIM inversion
- **Unconditional inversion**: Uses `prompt=""`, `guidance_scale=1.0` (required for DDIM correctness)
- **Prompt-agnostic**: Never requires or trusts prompts from clients

**Dependencies**:
- `src.detection.g_values.compute_g_values()` - Canonical g-value computation
- `src.detection.inversion.DDIMInverter` - DDIM inversion
- `src.models.detectors.BayesianDetector` - Bayesian detector
- `service.authority.WatermarkAuthorityService` - Detection configuration
- `service.detector_artifacts.DetectorArtifacts` - Artifact loading

---

### `service/detector_artifacts.py`

**Purpose**: Loader and validator for detection artifacts.

**Responsibilities**:
- **Artifact loading**: Loads `likelihood_params.json` and optional `mask.pt`
- **Consistency validation**: Validates config hash and mask shape match training
- **Caching**: Caches loaded artifacts in memory

**Key Classes**:
- `DetectorArtifacts`: Artifact loader and validator

**Validation Checks**:
1. **Config hash**: Computed g-field config hash must match hash in `likelihood_params.json` metadata
2. **Mask shape**: If mask is provided, `mask.sum()` must equal `num_positions` from likelihood params

**Artifacts**:
- `likelihood_params.json`: Trained likelihood parameters (probs_watermarked, probs_unwatermarked, num_positions)
- `mask.pt`: Optional structural mask tensor (binary mask for g-value positions)

**Dependencies**:
- None (standalone artifact loader)

---

### `service/generation/stable_diffusion.py`

**Purpose**: Stable Diffusion adapter for Phase-1 hosted generation.

**Responsibilities**:
- **Pipeline management**: Lazy-loads Stable Diffusion pipeline
- **Watermark embedding**: Applies seed-bias watermarking at z_T using `src/engine` components
- **G-value validation**: Computes g-values from generated z_T to validate watermark statistic

**Key Classes**:
- `StableDiffusionSeedBiasAdapter`: Implements `GenerationAdapter` interface

**Key Methods**:
- `generate()`: Generates watermarked image using `src.engine.pipeline.generate_with_watermark()`
- `get_model_info()`: Returns model metadata

**Generation Flow**:
1. Extract watermark payload (key_id, master_key, embedding_config)
2. Create `SeedBiasConfig` from embedding_config
3. Create `SeedBiasStrategy` from `src.engine.strategies.seed_bias`
4. Prepare strategy for sample (key_id determines G-field, seed determines image variation)
5. Call `src.engine.pipeline.generate_with_watermark()` to generate image
6. Compute g-values from generated z_T for validation (optional, debug only)

**Dependencies**:
- `src.core.config.*` - Configuration models (SeedBiasConfig, DiffusionConfig, etc.)
- `src.detection.g_values.compute_g_values()` - G-value computation (for validation)
- `src.engine.pipeline.create_pipeline()`, `generate_with_watermark()` - Pipeline creation and generation
- `src.engine.strategies.seed_bias.SeedBiasStrategy` - Seed-bias watermarking strategy

---

### `service/evaluation.py`

**Purpose**: Evaluation service for imperceptibility comparison (evaluation-only, not production).

**Responsibilities**:
- **Paired generation**: Generates baseline and watermarked images with identical parameters
- **Optimized diffusion**: Uses shared diffusion context to reduce compute by ~40-50%
- **Difference metrics**: Computes L2, PSNR, SSIM between images

**Key Classes**:
- `ImperceptibilityEvaluationService`: Evaluation service

**Key Methods**:
- `evaluate_imperceptibility()`: Generates paired images and computes metrics
- `prepare_shared_diffusion_context()`: Prepares shared context (text embeddings, scheduler, base z_T)
- `_run_dual_diffusion_loop()`: Runs single diffusion loop for both latents

**Security Notes**:
- Uses fixed, hardcoded evaluation master key (not stored, not issued)
- Does NOT use `WatermarkAuthorityService`
- Evaluation images are NOT valid for production detection

**Dependencies**:
- `src.core.config.SeedBiasConfig` - Seed-bias configuration
- `src.engine.pipeline.*` - Pipeline utilities
- `src.engine.strategies.seed_bias.SeedBiasStrategy` - Seed-bias strategy
- `service.evaluation_metrics.compute_all_metrics()` - Difference metrics

---

### `service/infra/db.py`

**Purpose**: In-memory database for watermark key storage.

**Responsibilities**:
- **Key storage**: Stores watermark_id → secret_key mapping
- **Metadata storage**: Stores model, strategy, status, created_at
- **Persistence**: Optional JSON-backed persistence to disk

**Key Classes**:
- `WatermarkKeyDB`: In-memory key database

**Storage Format**:
- JSON file: `service_data/watermark_keys.json`
- Keys are encrypted using `KeyEncryption` (Fernet)

**Dependencies**:
- `service.infra.security.KeyEncryption` - Key encryption

**Note**: In production, this should be replaced with a proper database (PostgreSQL, etc.)

---

### `service/infra/security.py`

**Purpose**: Security utilities for key generation and encryption.

**Responsibilities**:
- **Key generation**: Generates secure watermark IDs and master keys
- **Key encryption**: Encrypts/decrypts secret keys for storage

**Key Functions**:
- `generate_watermark_id()`: Generates unique watermark ID (format: `wm_xxxxx`)
- `generate_master_key()`: Generates 32-byte cryptographically secure master key
- `KeyEncryption`: Fernet-based symmetric encryption for key storage

**Dependencies**:
- `cryptography.fernet.Fernet` - Symmetric encryption

---

## API Layer

### `service/app/dependencies.py`

**Purpose**: Dependency injection for FastAPI routes.

**Responsibilities**:
- **Singleton management**: Provides singleton instances of services
- **Lazy initialization**: Services are created on first access
- **Device detection**: Auto-detects device (mps > cuda > cpu)

**Key Functions**:
- `get_watermark_authority()`: Returns `WatermarkAuthorityService` singleton
- `get_pipeline()`: Returns `StableDiffusionPipeline` singleton (lazy-loaded)
- `get_generation_adapter()`: Returns `GenerationAdapter` singleton (currently `StableDiffusionSeedBiasAdapter`)
- `get_detection_service()`: Returns `DetectionService` singleton (validates artifacts available)
- `is_detection_available()`: Checks if detection artifacts are available
- `get_detection_availability_status()`: Returns detailed artifact availability status

**Global Caches**:
- `_authority_service`: `WatermarkAuthorityService` instance
- `_generation_adapter`: `GenerationAdapter` instance
- `_pipeline_cache`: `StableDiffusionPipeline` instance
- `_detection_service`: `DetectionService` instance

---

### `service/app/routes/`

**Purpose**: FastAPI route handlers for API endpoints.

#### `service/app/routes/generate.py`

**Endpoint**: `POST /api/v1/generate`

**Flow**:
1. Get `WatermarkAuthorityService` and `GenerationAdapter` from dependencies
2. Get or create watermark policy (auto-creates if `key_id` not provided)
3. Validate model_id matches research assumptions
4. Generate image using adapter
5. Encode image to base64
6. Return response with image, key_id, metadata, watermark_version

**Request Schema**: `GenerateRequest` (prompt, optional key_id, generation params)
**Response Schema**: `GenerateResponse` (image_base64, key_id, generation_metadata, watermark_version)

**Dependencies**:
- `service.app.dependencies.get_watermark_authority()`
- `service.app.dependencies.get_generation_adapter()`

---

#### `service/app/routes/detect.py`

**Endpoint**: `POST /api/v1/detect`

**Flow**:
1. Check if detection artifacts are available
2. Get `DetectionService` from dependencies
3. Decode base64 image
4. Run detection (prompt-agnostic)
5. Return detection results

**Request Schema**: `DetectRequest` (image_base64, key_id)
**Response Schema**: `DetectResponse` (detected, score, confidence, policy_version, etc.)

**Dependencies**:
- `service.app.dependencies.get_detection_service()`
- `service.app.dependencies.is_detection_available()`

**Error Handling**:
- 503 if artifacts unavailable
- 404 if watermark not found or revoked
- 400 if validation fails (config mismatch, shape mismatch)

---

#### `service/app/routes/evaluate.py`

**Endpoint**: `POST /api/v1/evaluate/imperceptibility`

**Flow**:
1. Get `ImperceptibilityEvaluationService` from dependencies
2. Generate baseline and watermarked images with shared context
3. Compute difference metrics (L2, PSNR, SSIM)
4. Encode images to base64
5. Return response with both images and metrics

**Request Schema**: `ImperceptibilityEvalRequest` (prompt, seed, generation params)
**Response Schema**: `ImperceptibilityEvalResponse` (baseline_image_base64, watermarked_image_base64, difference_metrics, model_info)

**Dependencies**:
- `service.evaluation.ImperceptibilityEvaluationService`

**Note**: Evaluation-only endpoint, not part of production watermarking system.

---

#### `service/app/routes/health.py`

**Endpoint**: `GET /api/v1/health`

**Returns**: `HealthResponse` with status="ok"

---

#### `service/app/routes/demo.py`

**Endpoints**: Demo endpoints for presentation (non-technical users)

---

### `service/app/schemas.py`

**Purpose**: Pydantic schemas for API requests and responses.

**Key Schemas**:
- `GenerateRequest`, `GenerateResponse`, `GenerationMetadata`
- `DetectRequest`, `DetectResponse`
- `ImperceptibilityEvalRequest`, `ImperceptibilityEvalResponse`, `DifferenceMetrics`, `ModelInfo`
- `HealthResponse`

**Design Principle**: Product-level schemas only, no research internals exposed.

---

### `service/app/middleware.py`

**Purpose**: Rate limiting middleware.

**Key Classes**:
- `RateLimitMiddleware`: FastAPI middleware for rate limiting
- `RateLimiter`: Simple in-memory rate limiter

**Rate Limits**:
- `/api/v1/detect`: 50 requests per 60 seconds per IP
- `/api/v1/evaluate`: 20 requests per 60 seconds per IP

**Note**: In production, use Redis or similar for distributed rate limiting.

---

### `service/app/artifact_resolver.py`

**Purpose**: Centralized artifact resolution and availability tracking.

**Responsibilities**:
- **Environment variable resolution**: Reads `LIKELIHOOD_PARAMS_PATH` and `MASK_PATH`
- **Path validation**: Validates paths exist and are readable
- **Startup caching**: Caches resolved paths at startup
- **Availability tracking**: Tracks artifact availability state

**Key Classes**:
- `ArtifactResolver`: Centralized artifact resolver
- `ArtifactResolutionResult`: Result of artifact resolution

**Key Methods**:
- `resolve()`: Resolves artifact paths from environment variables
- `get_availability_status()`: Returns availability status for diagnostics

**Design Principle**: Single source of truth for artifact paths. No path guessing or fallback searching.

**Dependencies**:
- None (standalone resolver)

---

## Integration with `src/`

The `service/` layer integrates with `src/` through **explicit imports** of research components. The integration is **one-way**: `service/` depends on `src/`, but `src/` has no knowledge of `service/`.

### Key Integration Points

#### 1. G-Value Computation

**Service Usage**: `service/detection.py`, `service/generation/stable_diffusion.py`

**Source Implementation**: `src/detection/g_values.py`

```python
from src.detection.g_values import compute_g_values
```

**Purpose**: Canonical g-value computation for watermark detection. Both generation and detection use this function to ensure consistency.

**Usage**:
- **Generation**: Validates watermark statistic after generation (optional, debug only)
- **Detection**: Computes g-values from inverted z_T for detection

---

#### 2. DDIM Inversion

**Service Usage**: `service/detection.py`

**Source Implementation**: `src/detection/inversion.py`

```python
from src.detection.inversion import DDIMInverter
```

**Purpose**: Inverts image to initial latent z_T using DDIM. Used in detection to recover the latent where watermark was embedded.

**Usage**:
- **Detection**: `StableDiffusionDetectionBackend.invert_to_zT()` uses `DDIMInverter.perform_full_inversion()`

---

#### 3. Bayesian Detector

**Service Usage**: `service/detection.py`

**Source Implementation**: `src/models/detectors.py`

```python
from src.models.detectors import BayesianDetector
```

**Purpose**: Bayesian detector for watermark detection. Uses trained likelihood parameters from artifacts.

**Usage**:
- **Detection**: `DetectionService.detect()` instantiates `BayesianDetector` and calls `detector.score()`

---

#### 4. Generation Pipeline

**Service Usage**: `service/generation/stable_diffusion.py`, `service/evaluation.py`

**Source Implementation**: `src/engine/pipeline.py`

```python
from src.engine.pipeline import create_pipeline, generate_with_watermark
```

**Purpose**: Creates Stable Diffusion pipeline and generates watermarked images.

**Usage**:
- **Generation**: `StableDiffusionSeedBiasAdapter.generate()` uses `generate_with_watermark()`
- **Evaluation**: `ImperceptibilityEvaluationService` uses pipeline utilities

---

#### 5. Seed-Bias Strategy

**Service Usage**: `service/generation/stable_diffusion.py`, `service/evaluation.py`

**Source Implementation**: `src/engine/strategies/seed_bias.py`

```python
from src.engine.strategies.seed_bias import SeedBiasStrategy
```

**Purpose**: Seed-bias watermarking strategy. Applies watermark at z_T using spherical mixing.

**Usage**:
- **Generation**: `StableDiffusionSeedBiasAdapter` creates `SeedBiasStrategy` and passes to `generate_with_watermark()`
- **Evaluation**: `ImperceptibilityEvaluationService` uses `SeedBiasStrategy` for evaluation watermarking

---

#### 6. Configuration Models

**Service Usage**: `service/generation/stable_diffusion.py`, `service/evaluation.py`

**Source Implementation**: `src/core/config.py`

```python
from src.core.config import (
    DiffusionConfig,
    SeedBiasConfig,
    GFieldConfig,
    PRFConfig,
    # ... other config models
)
```

**Purpose**: Type-safe configuration models using Pydantic. Ensures configuration consistency.

**Usage**:
- **Generation**: `StableDiffusionSeedBiasAdapter` creates `SeedBiasConfig` and `DiffusionConfig`
- **Evaluation**: `ImperceptibilityEvaluationService` uses `SeedBiasConfig`

---

### Integration Patterns

1. **Explicit Imports**: All `src/` imports are explicit and documented
2. **No Circular Dependencies**: `src/` has no knowledge of `service/`
3. **Abstraction Layers**: Service layer provides abstractions (e.g., `GenerationAdapter`, `DetectionService`) that hide `src/` internals
4. **Configuration Passing**: Service layer constructs `src/` config objects from authority policies
5. **Canonical Functions**: Service layer uses canonical functions from `src/` (e.g., `compute_g_values()`) to ensure consistency

---

## Request Flow

### Generation Request Flow

```
Client Request (POST /api/v1/generate)
    ↓
service/app/routes/generate.py::generate()
    ↓
service/app/dependencies.py::get_watermark_authority()
    ↓
service/authority.py::WatermarkAuthorityService
    ├── create_watermark_policy() [if key_id not provided]
    └── get_watermark_payload()
    ↓
service/app/dependencies.py::get_generation_adapter()
    ↓
service/generation/stable_diffusion.py::StableDiffusionSeedBiasAdapter.generate()
    ├── Creates SeedBiasConfig from embedding_config
    ├── Creates SeedBiasStrategy (src/engine/strategies/seed_bias.py)
    └── Calls src/engine/pipeline.py::generate_with_watermark()
        ├── Uses src/engine/strategies/seed_bias.py::SeedBiasStrategy
        └── Uses src/algorithms/g_field.py::GFieldGenerator
    ↓
Returns image + metadata
    ↓
service/app/routes/generate.py encodes image to base64
    ↓
Client Response (GenerateResponse)
```

---

### Detection Request Flow

```
Client Request (POST /api/v1/detect)
    ↓
service/app/routes/detect.py::detect()
    ↓
service/app/dependencies.py::get_detection_service()
    ↓
service/detection.py::DetectionService.detect()
    ├── service/authority.py::WatermarkAuthorityService.get_detection_config()
    │   └── Returns: master_key, detection_config, g_field_config, inversion_config
    ├── service/detector_artifacts.py::DetectorArtifacts (loads artifacts)
    ├── service/detection.py::StableDiffusionDetectionBackend.encode_image()
    │   └── Uses src/detection/inversion.py::DDIMInverter.encode_image()
    ├── service/detection.py::StableDiffusionDetectionBackend.invert_to_zT()
    │   └── Uses src/detection/inversion.py::DDIMInverter.perform_full_inversion()
    ├── src/detection/g_values.py::compute_g_values()
    │   └── Uses src/algorithms/g_field.py::GFieldGenerator
    │   └── Uses src/detection/prf.py::PRFKeyDerivation
    ├── Applies mask + binarization (service/detection.py logic)
    └── src/models/detectors.py::BayesianDetector.score()
    ↓
Returns detection results
    ↓
service/app/routes/detect.py formats response
    ↓
Client Response (DetectResponse)
```

---

### Evaluation Request Flow

```
Client Request (POST /api/v1/evaluate/imperceptibility)
    ↓
service/app/routes/evaluate.py::evaluate_imperceptibility()
    ↓
service/evaluation.py::ImperceptibilityEvaluationService.evaluate_imperceptibility()
    ├── prepare_shared_diffusion_context()
    │   └── Uses src/engine/pipeline.py::prepare_initial_latents()
    │   └── Uses src/engine/sampling_utils.py::get_text_embeddings()
    ├── Creates SeedBiasStrategy (src/engine/strategies/seed_bias.py)
    ├── _apply_watermark_bias_to_latent() (applies watermark to z_T)
    ├── _run_dual_diffusion_loop() (processes both latents)
    └── service/evaluation_metrics.py::compute_all_metrics()
    ↓
Returns baseline + watermarked images + metrics
    ↓
service/app/routes/evaluate.py encodes images to base64
    ↓
Client Response (ImperceptibilityEvalResponse)
```

---

## Shared Models and Configuration

### Configuration Models (`src/core/config.py`)

**Shared Between Service and Research**:
- `DiffusionConfig`: Diffusion model configuration
- `SeedBiasConfig`: Seed-bias watermarking configuration
- `GFieldConfig`: G-field generation configuration
- `PRFConfig`: PRF-based key derivation configuration
- `KeySettings`: Key derivation settings

**Usage in Service**:
- `service/generation/stable_diffusion.py`: Creates `SeedBiasConfig` and `DiffusionConfig`
- `service/evaluation.py`: Creates `SeedBiasConfig`

**Note**: Service layer constructs these configs from authority policies, ensuring consistency.

---

### G-Field Configuration

**Shared Structure**:
```python
g_field_config = {
    "mapping_mode": "binary",
    "domain": "frequency",
    "frequency_mode": "bandpass",
    "low_freq_cutoff": 0.05,
    "high_freq_cutoff": 0.4,
    "normalize_zero_mean": True,
    "normalize_unit_variance": True,
}
```

**Usage**:
- **Generation**: `service/generation/stable_diffusion.py` constructs from `embedding_config`
- **Detection**: `service/authority.py` provides in `get_detection_config()`
- **G-Value Computation**: `src/detection/g_values.py::compute_g_values()` uses for G-field generation

**Consistency**: Config hash is validated in `DetectorArtifacts` to ensure generation and detection use same config.

---

### Watermark Policy Structure

**Authority Policy**:
```python
{
    "key_id": "wm_xxxxx",
    "watermark_version": "abc123...",  # Policy version hash
    "embedding_config": {
        "lambda_strength": 0.05,
        "domain": "frequency",
        "low_freq_cutoff": 0.05,
        "high_freq_cutoff": 0.4,
    },
    "detection_config": {
        "detector_type": "bayesian",
        "likelihood_params_path": "/path/to/likelihood_params.json",
        "threshold": 0.5,
        "prior_watermarked": 0.5,
    },
}
```

**Watermark Payload (Generation)**:
```python
{
    "key_id": "wm_xxxxx",
    "master_key": "...",  # Internal use only
    "embedding_config": {...},
    "watermark_version": "abc123...",
}
```

**Detection Config**:
```python
{
    "key_id": "wm_xxxxx",
    "master_key": "...",  # Internal use only
    "detection_config": {...},
    "g_field_config": {...},
    "inversion": {
        "num_inference_steps": 50,
        "guidance_scale": 1.0,  # Must be 1.0 for DDIM correctness
        "prompt_required": False,
    },
    "watermark_version": "abc123...",
}
```

---

### Artifact Structure

**Likelihood Parameters** (`likelihood_params.json`):
```json
{
    "num_positions": 16384,
    "watermarked": {
        "probs": [0.1, 0.2, ...]  # Per-position probabilities
    },
    "unwatermarked": {
        "probs": [0.5, 0.5, ...]
    },
    "g_field_config_hash": "abc123..."  # Config hash for validation
}
```

**Mask** (`mask.pt`):
- Binary tensor `[N]` or `[C, H, W]` flattened to `[N]`
- `mask.sum()` must equal `num_positions` from likelihood params

---

## Architectural Assumptions and Couplings

### Stable Diffusion Assumptions

**Tight Coupling**:
- **Generation**: `StableDiffusionSeedBiasAdapter` assumes Stable Diffusion pipeline
- **Detection**: `DetectionService` assumes SD VAE encoding and DDIM inversion
- **Evaluation**: `ImperceptibilityEvaluationService` assumes SD pipeline

**Impact**:
- Adding support for other models (e.g., SDXL, DALL-E) requires new adapters/backends
- Detection backend abstraction (`StableDiffusionDetectionBackend`) is designed for extension but currently only supports SD

**Mitigation**:
- `GenerationAdapter` interface allows multiple implementations
- `DetectionService` delegates SD-specific operations to `StableDiffusionDetectionBackend` (extensible)

---

### Inversion Assumptions

**Critical Assumption**:
- Detection requires **unconditional DDIM inversion** (`prompt=""`, `guidance_scale=1.0`)
- This is enforced in `DetectionService.detect()` and `WatermarkAuthorityService.get_detection_config()`

**Rationale**:
- Research layer requires `guidance_scale=1.0` for DDIM inversion correctness (see `src/detection/inversion.py`)
- Detection is prompt-agnostic (never requires or trusts prompts)

**Impact**:
- Generation can use any `guidance_scale`, but detection will always use `1.0` for inversion
- This is acceptable because watermark is embedded at z_T, not affected by guidance during generation

---

### Artifact Dependencies

**Tight Coupling**:
- Detection requires precomputed artifacts (`likelihood_params.json`, optional `mask.pt`)
- Artifacts must match g-field config used during generation
- Artifacts are validated at startup (fail-fast if invalid)

**Impact**:
- Service cannot start detection without valid artifacts
- Artifact paths must be set via environment variables (`LIKELIHOOD_PARAMS_PATH`, `MASK_PATH`)

**Mitigation**:
- `ArtifactResolver` provides clear error messages for missing artifacts
- Service starts even if artifacts unavailable (detection endpoints return 503)

---

### Policy Versioning

**Assumption**:
- `policy_version` is computed deterministically from statistical parameters
- Same parameters → same `policy_version` (reproducible)

**Impact**:
- Policy changes (e.g., threshold, prior) result in new `policy_version`
- Clients can track policy versions for auditability

**Mitigation**:
- Policy version computation is explicit and documented in `WatermarkAuthorityService._compute_policy_version()`

---

### Database Assumptions

**Tight Coupling**:
- `WatermarkKeyDB` is in-memory, JSON-backed (not production-ready)
- Keys are encrypted using Fernet (symmetric encryption)

**Impact**:
- Not suitable for distributed deployments
- Encryption key management is basic (should use environment variables in production)

**Mitigation**:
- Database interface is abstracted (`get_db()`), can be replaced with proper database
- Encryption can be enhanced with proper key management

---

### Device Assumptions

**Assumption**:
- Service auto-detects device (mps > cuda > cpu)
- FP16 is only used for CUDA (MPS and CPU use FP32)

**Impact**:
- Performance varies by device
- Model loading behavior differs by device

**Mitigation**:
- Device detection is explicit and logged
- FP16 usage is conditional on device type

---

## Extension Points

### Adding New Generation Models

**Extension Point**: `GenerationAdapter` interface

**Steps**:
1. Create new adapter class implementing `GenerationAdapter` (e.g., `SDXLAdapter`)
2. Implement `generate()` and `get_model_info()` methods
3. Update `service/app/dependencies.py::get_generation_adapter()` to return new adapter
4. Ensure adapter uses `src/` components for watermarking (e.g., `SeedBiasStrategy`)

**Example**:
```python
class SDXLAdapter(GenerationAdapter):
    def generate(self, prompt, watermark_payload, ...):
        # Create SDXL pipeline
        # Apply watermark using src/engine components
        # Return image + metadata
```

---

### Adding New Detection Backends

**Extension Point**: `StableDiffusionDetectionBackend` pattern

**Steps**:
1. Create new backend class (e.g., `SDXLDetectionBackend`)
2. Implement `encode_image()` and `invert_to_zT()` methods (or equivalent for non-inversion models)
3. Update `DetectionService` to support multiple backends (e.g., based on model_family)
4. Ensure backend uses `src/` components for g-value computation

**Example**:
```python
class SDXLDetectionBackend:
    def encode_image(self, image):
        # SDXL-specific encoding
    def invert_to_zT(self, z_0, ...):
        # SDXL-specific inversion (or skip if not needed)
```

---

### Adding New Detection Algorithms

**Extension Point**: `src/models/detectors.py`

**Steps**:
1. Implement new detector class in `src/models/detectors.py` (e.g., `HybridDetector`)
2. Update `DetectionService` to support new detector type
3. Update `WatermarkAuthorityService` to provide detector configuration
4. Ensure detector uses same g-value computation and artifacts

**Note**: Currently, only `BayesianDetector` is supported. Legacy detectors (fast_only, hybrid, full_inversion) are deprecated.

---

### Adding New Evaluation Metrics

**Extension Point**: `service/evaluation_metrics.py`

**Steps**:
1. Add new metric function (e.g., `compute_fid()`)
2. Update `compute_all_metrics()` to include new metric
3. Update `DifferenceMetrics` schema in `service/app/schemas.py`
4. Update `ImperceptibilityEvalResponse` schema

---

### Replacing Database

**Extension Point**: `service/infra/db.py`

**Steps**:
1. Implement new database class with same interface as `WatermarkKeyDB`
2. Update `get_db()` to return new database instance
3. Ensure encryption/decryption is handled (or delegate to new database)

**Example**:
```python
class PostgreSQLWatermarkDB:
    def create_watermark(self, watermark_id, secret_key, ...):
        # Store in PostgreSQL
    def get_watermark(self, watermark_id):
        # Retrieve from PostgreSQL
    # ... other methods
```

---

### Adding New Artifact Types

**Extension Point**: `service/detector_artifacts.py`

**Steps**:
1. Add new artifact loading logic to `DetectorArtifacts`
2. Update `ArtifactResolver` to resolve new artifact paths
3. Update validation logic if needed

**Example**:
```python
class DetectorArtifacts:
    def __init__(self, ..., new_artifact_path=None):
        # Load new artifact
        self.new_artifact = self._load_new_artifact(new_artifact_path)
```

---

## Summary

The `service/` directory provides a **product layer** that:

1. **Wraps research implementation**: Uses `src/` components for core watermarking algorithms
2. **Provides API abstraction**: Hides research internals from clients
3. **Manages authority**: Handles cryptographic keys and policy management
4. **Handles artifacts**: Loads and validates detection model artifacts
5. **Supports evaluation**: Provides evaluation endpoints (isolated from production)

**Key Design Principles**:
- **Separation of concerns**: Service layer handles API, authority, artifacts; `src/` handles algorithms
- **Explicit dependencies**: All `src/` imports are explicit and documented
- **Abstraction layers**: Service provides abstractions (adapters, services) that hide internals
- **Fail-fast validation**: Artifacts and configurations are validated at startup
- **Graceful degradation**: Service starts even if artifacts unavailable (detection returns errors)

**Future Extensions**:
- Phase-2: Client-side generation (API only issues credentials)
- Multi-model support: SDXL, DALL-E, etc.
- Production database: Replace in-memory DB with PostgreSQL
- Distributed rate limiting: Replace in-memory limiter with Redis

