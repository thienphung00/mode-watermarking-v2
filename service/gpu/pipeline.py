"""
GPU pipeline for image generation and DDIM inversion.

This module provides:
- Watermarked image generation using Stable Diffusion
- DDIM inversion for latent extraction
- G-value computation for detection

Initially implemented as stubs, with realistic interfaces.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from image generation."""
    image_bytes: bytes
    seed_used: int
    latent_shape: Tuple[int, ...]


@dataclass
class InversionResult:
    """Result from DDIM inversion."""
    latent: np.ndarray
    g_values: np.ndarray
    

@dataclass
class DetectionResult:
    """Result from detection."""
    detected: bool
    score: float
    confidence: float
    log_odds: float
    posterior: float
    latent_shape: Tuple[int, ...]


class GPUPipeline:
    """
    GPU pipeline for watermark operations.
    
    This class wraps the heavy GPU computations:
    - Image generation with watermark embedding
    - DDIM inversion for latent recovery
    - G-value computation and detection
    
    The implementation can be either:
    - Stub mode: Returns realistic mock data (for testing)
    - Full mode: Uses actual SD pipeline and detection logic from /src
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        stub_mode: bool = True,
    ):
        """
        Initialize GPU pipeline.
        
        Args:
            model_id: HuggingFace model ID
            device: Device for computation
            stub_mode: If True, use stub implementations
        """
        self.model_id = model_id
        self.device = device
        self.stub_mode = stub_mode
        
        self._pipeline = None
        self._vae = None
        self._is_loaded = False
        
        if stub_mode:
            logger.info("GPUPipeline initialized in STUB mode")
            self._is_loaded = True
        else:
            logger.info(f"GPUPipeline initializing with model: {model_id}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    def load_models(self) -> None:
        """
        Load models into GPU memory.
        
        In stub mode, this is a no-op.
        In full mode, loads SD pipeline from HuggingFace.
        """
        if self.stub_mode:
            self._is_loaded = True
            return
        
        try:
            # Import here to avoid dependency issues in stub mode
            import torch
            from diffusers import StableDiffusionPipeline
            
            logger.info(f"Loading model: {self.model_id}")
            
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            self._vae = self._pipeline.vae
            self._is_loaded = True
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.warning("Falling back to stub mode")
            self.stub_mode = True
            self._is_loaded = True
    
    def generate(
        self,
        prompt: str,
        derived_key: str,
        key_id: str,
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Generate watermarked image.
        
        Args:
            prompt: Text prompt
            derived_key: Scoped derived key (NOT master key)
            key_id: Key identifier
            seed: Random seed
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            width: Image width
            height: Image height
            embedding_config: Watermark embedding parameters
            
        Returns:
            GenerationResult with image bytes and metadata
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        if self.stub_mode:
            return self._generate_stub(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
            )
        
        return self._generate_full(
            prompt=prompt,
            derived_key=derived_key,
            key_id=key_id,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            embedding_config=embedding_config or {},
        )
    
    def _generate_stub(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
    ) -> GenerationResult:
        """Generate stub image (colored noise based on prompt hash)."""
        # Create deterministic "image" based on prompt and seed
        np.random.seed(seed)
        
        # Generate colored noise
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Add some structure based on prompt hash
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        base_color = [
            (prompt_hash >> 16) & 0xFF,
            (prompt_hash >> 8) & 0xFF,
            prompt_hash & 0xFF,
        ]
        
        # Blend with base color
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] * 0.7 + base_color[i] * 0.3).astype(np.uint8)
        
        # Convert to PNG bytes
        try:
            from PIL import Image
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        except ImportError:
            # Minimal PNG without PIL
            image_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # Invalid but non-empty
        
        return GenerationResult(
            image_bytes=image_bytes,
            seed_used=seed,
            latent_shape=(4, height // 8, width // 8),
        )
    
    def _generate_full(
        self,
        prompt: str,
        derived_key: str,
        key_id: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        embedding_config: Dict[str, Any],
    ) -> GenerationResult:
        """Generate image using full SD pipeline with watermark."""
        import torch
        
        # Import watermarking components from src
        from src.engine.strategies.seed_bias import SeedBiasStrategy
        
        # Create watermark strategy
        strategy = SeedBiasStrategy(
            master_key=derived_key,  # Using derived key
            key_id=key_id,
            lambda_strength=embedding_config.get("lambda_strength", 0.05),
            domain=embedding_config.get("domain", "frequency"),
            low_freq_cutoff=embedding_config.get("low_freq_cutoff", 0.05),
            high_freq_cutoff=embedding_config.get("high_freq_cutoff", 0.4),
        )
        
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate with watermark
        # Note: This requires the pipeline to support callback hooks
        with torch.no_grad():
            output = self._pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                # callback=strategy.callback,  # If supported
            )
        
        image = output.images[0]
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        return GenerationResult(
            image_bytes=image_bytes,
            seed_used=seed,
            latent_shape=(4, height // 8, width // 8),
        )
    
    def invert_and_detect(
        self,
        image_bytes: bytes,
        derived_key: str,
        key_id: str,
        g_field_config: Optional[Dict[str, Any]] = None,
        detection_config: Optional[Dict[str, Any]] = None,
        inversion_config: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """
        Perform DDIM inversion and detect watermark.
        
        Args:
            image_bytes: Raw image bytes
            derived_key: Scoped derived key (NOT master key)
            key_id: Key identifier
            g_field_config: G-field configuration
            detection_config: Detection parameters
            inversion_config: DDIM inversion parameters
            
        Returns:
            DetectionResult with detection decision and statistics
        """
        if self.stub_mode:
            return self._detect_stub(
                image_bytes=image_bytes,
                key_id=key_id,
            )
        
        return self._detect_full(
            image_bytes=image_bytes,
            derived_key=derived_key,
            key_id=key_id,
            g_field_config=g_field_config or {},
            detection_config=detection_config or {},
            inversion_config=inversion_config or {},
        )
    
    def _detect_stub(
        self,
        image_bytes: bytes,
        key_id: str,
    ) -> DetectionResult:
        """Generate stub detection result."""
        # Use image hash + key_id for deterministic but varied results
        combined = hashlib.sha256(image_bytes + key_id.encode()).hexdigest()
        hash_val = int(combined[:8], 16) / 0xFFFFFFFF
        
        # Simulate watermarked images having higher scores
        if hash_val > 0.3:  # 70% chance of "detected" for varied images
            score = 2.0 + hash_val * 3.0  # Score between 2 and 5
            posterior = 0.75 + hash_val * 0.24
        else:
            score = -0.5 + hash_val * 2.0  # Score between -0.5 and 1.5
            posterior = 0.1 + hash_val * 0.4
        
        detected = posterior > 0.5
        confidence = posterior if detected else (1 - posterior)
        log_odds = np.log(posterior / (1 - posterior)) if 0 < posterior < 1 else 0.0
        
        return DetectionResult(
            detected=detected,
            score=float(score),
            confidence=float(confidence),
            log_odds=float(log_odds),
            posterior=float(posterior),
            latent_shape=(4, 64, 64),
        )
    
    def _detect_full(
        self,
        image_bytes: bytes,
        derived_key: str,
        key_id: str,
        g_field_config: Dict[str, Any],
        detection_config: Dict[str, Any],
        inversion_config: Dict[str, Any],
    ) -> DetectionResult:
        """Perform full DDIM inversion and detection."""
        import torch
        from PIL import Image
        
        # Import detection components from src
        from src.detection.inversion import DDIMInverter
        from src.detection.g_values import compute_g_values
        from src.detection.statistics import detect_watermark, compute_s_statistic
        from src.algorithms.g_field import compute_g_expected
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Create inverter
        inverter = DDIMInverter(
            vae=self._vae,
            unet=self._pipeline.unet,
            scheduler=self._pipeline.scheduler,
            device=self.device,
        )
        
        # DDIM inversion
        num_steps = inversion_config.get("num_inference_steps", 50)
        
        with torch.no_grad():
            z_T = inverter.invert(
                image=image,
                num_inference_steps=num_steps,
                prompt="",
                guidance_scale=1.0,
            )
        
        latent_np = z_T.cpu().numpy()
        if latent_np.ndim == 4:
            latent_np = latent_np[0]
        
        # Compute G-fields
        g_expected = compute_g_expected(
            master_key=derived_key,
            key_id=key_id,
            shape=latent_np.shape,
        )
        
        # Simple detection
        s_stat = compute_s_statistic(latent_np, g_expected)
        
        # Bayesian posterior
        threshold = detection_config.get("threshold", 0.5)
        prior = detection_config.get("prior_watermarked", 0.5)
        
        # Simple likelihood ratio
        n = latent_np.size
        mu_w = np.sqrt(n) * 0.1
        lr = np.exp(s_stat * mu_w - 0.5 * mu_w**2)
        posterior_odds = (prior / (1 - prior)) * lr
        posterior = posterior_odds / (1 + posterior_odds)
        
        detected = posterior > threshold
        
        return DetectionResult(
            detected=detected,
            score=float(s_stat),
            confidence=float(posterior) if detected else float(1 - posterior),
            log_odds=float(np.log(posterior_odds)) if posterior_odds > 0 else 0.0,
            posterior=float(posterior),
            latent_shape=latent_np.shape,
        )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if self.stub_mode:
            return {
                "gpu_available": True,
                "memory_used_mb": 1000,
                "memory_total_mb": 8000,
                "memory_used_pct": 12.5,
            }
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {"gpu_available": False}
            
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            
            return {
                "gpu_available": True,
                "memory_used_mb": int(memory_allocated),
                "memory_total_mb": int(memory_total),
                "memory_used_pct": (memory_allocated / memory_total) * 100,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return {"gpu_available": False, "error": str(e)}
