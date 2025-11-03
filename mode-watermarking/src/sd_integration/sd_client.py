"""
Stable Diffusion client wrapper with denoising-time watermark bias hooks.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from ..config.config_loader import ConfigLoader
from .timestep_mapper import TimestepMapper
from ..utils.io import ManifestIO, ensure_dir


@dataclass
class SDClient:
    """Stable Diffusion client with watermark embedding support."""
    
    config_paths: Dict[str, str]  # Paths to config YAMLs
    device: str = "cuda"
    
    # Internal state
    _pipeline: Optional[Any] = field(default=None, init=False, repr=False)
    _denoiser_hook: Optional[Callable] = field(default=None, init=False, repr=False)
    _timestep_mapper: Optional[TimestepMapper] = field(default=None, init=False, repr=False)
    _original_unet_forward: Optional[Callable] = field(default=None, init=False, repr=False)
    _config_loader: Optional[ConfigLoader] = field(default=None, init=False, repr=False)
    _diffusion_cfg: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _model_cfg: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _watermark_cfg: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize config loader and load configurations."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library not available. Install with: pip install diffusers"
            )
        
        self._config_loader = ConfigLoader()
        self._diffusion_cfg = self._config_loader.load_diffusion_config(
            self.config_paths["diffusion"]
        )
        self._model_cfg = self._config_loader.load_model_architecture_config(
            self.config_paths["model"]
        )
        self._watermark_cfg = self._config_loader.load_watermark_config(
            self.config_paths["watermark"]
        )
    
    def initialize_pipeline(self) -> None:
        """Load pretrained SD model with DDIM scheduler."""
        # Get model ID from config or default
        model_id = self._model_cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
        
        # Initialize DDIM scheduler with config parameters
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        
        # Configure scheduler parameters from config
        scheduler.config.num_train_timesteps = self._diffusion_cfg["trained_timesteps"]
        scheduler.config.beta_start = self._diffusion_cfg["beta_start"]
        scheduler.config.beta_end = self._diffusion_cfg["beta_end"]
        scheduler.config.beta_schedule = self._diffusion_cfg["beta_schedule"]
        scheduler.config.prediction_type = self._diffusion_cfg["prediction_type"]
        
        # Load pipeline
        dtype = torch.float16 if self._model_cfg.get("use_fp16", True) else torch.float32
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=dtype,
            safety_checker=None,  # Disable for research
            requires_safety_checker=False
        ).to(self.device)
        
        # Enable gradient checkpointing if configured
        if self._model_cfg.get("gradient_checkpointing", False):
            self._pipeline.unet.enable_gradient_checkpointing()
        
        # Set DDIM timesteps for inference (eta=0.0 for deterministic)
        num_inference_steps = self._diffusion_cfg["inference_timesteps"]
        self._pipeline.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps
        )
        
        # Initialize timestep mapper
        self._timestep_mapper = TimestepMapper(
            trained_timesteps=self._diffusion_cfg["trained_timesteps"],
            inference_timesteps=num_inference_steps
        )
        
        # Store original UNet forward for hooking
        self._original_unet_forward = self._pipeline.unet.forward
    
    def register_denoiser_hook(self, hook_fn: Callable) -> None:
        """
        Register hook to modify noise predictions during UNet forward.
        
        Args:
            hook_fn: Function that takes (eps_t, timestep, timestep_mapper, ...) and returns modified eps_t
        """
        self._denoiser_hook = hook_fn
        
        # Wrap UNet forward method
        def hooked_forward(
            sample: torch.Tensor,
            timestep,
            encoder_hidden_states: torch.Tensor,
            **kwargs
        ) -> torch.Tensor:
            # Call original forward to get predicted noise
            noise_pred = self._original_unet_forward(
                sample,
                timestep,
                encoder_hidden_states,
                **kwargs
            )
            
            # Apply watermark bias via hook if registered
            if self._denoiser_hook is not None:
                noise_pred = self._denoiser_hook(
                    noise_pred,
                    timestep,
                    self._timestep_mapper
                )
            
            return noise_pred
        
        # Replace UNet forward method
        self._pipeline.unet.forward = hooked_forward
    
    def set_attention_processor(self, processor_type: str = "default") -> None:
        """
        Set attention processor for memory optimization.
        
        Args:
            processor_type: "default", "xformers", "sliced"
        """
        if processor_type == "xformers":
            try:
                from diffusers.models.attention_processor import XFormersAttnProcessor
                self._pipeline.unet.set_attn_processor(XFormersAttnProcessor())
            except ImportError:
                print("Warning: xformers not available, using default attention")
        elif processor_type == "sliced":
            # Enable attention slicing
            self._pipeline.enable_attention_slicing()
        # Default: use standard AttnProcessor2_0
    
    def update_scheduler_config(
        self,
        beta_start: Optional[float] = None,
        beta_end: Optional[float] = None,
        beta_schedule: Optional[str] = None,
        num_train_timesteps: Optional[int] = None
    ) -> None:
        """Update scheduler configuration dynamically."""
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        scheduler = self._pipeline.scheduler
        
        if beta_start is not None:
            scheduler.config.beta_start = beta_start
        if beta_end is not None:
            scheduler.config.beta_end = beta_end
        if beta_schedule is not None:
            scheduler.config.beta_schedule = beta_schedule
        if num_train_timesteps is not None:
            scheduler.config.num_train_timesteps = num_train_timesteps
        
        # Reinitialize scheduler
        scheduler = DDIMScheduler.from_config(scheduler.config)
        self._pipeline.scheduler = scheduler
    
    def _hash_latent(self, z: np.ndarray) -> str:
        """Hash latent tensor for manifest."""
        m = hashlib.sha256()
        m.update(z.tobytes())
        return m.hexdigest()[:16]
    
    def _create_manifest(
        self,
        prompt: str,
        key_config: Dict[str, Any],
        seed: Optional[int],
        num_steps: int,
        cfg_scale: float
    ) -> Dict[str, Any]:
        """Create manifest metadata for generated image."""
        timestamp = int(time.time())
        sample_id = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]
        
        watermark_cfg = self._watermark_cfg.get("watermark", {})
        
        manifest: Dict[str, Any] = {
            "image_path": "",  # To be populated by caller when saving
            "sample_id": sample_id,
            "sample_seed": seed if seed is not None else int(watermark_cfg.get("base_seed", 0)),
            "zT_hash": "",  # Will be populated if latent available
            "key_id": watermark_cfg.get("key_id", "unknown"),
            "key_info": {
                "key_scheme": watermark_cfg.get("key_scheme", "LCG-v1"),
                "a": watermark_cfg.get("lcg", {}).get("a"),
                "c": watermark_cfg.get("lcg", {}).get("c"),
                "m": watermark_cfg.get("lcg", {}).get("m"),
                "bit_pos": watermark_cfg.get("lcg", {}).get("bit_pos"),
                "mapping_mode": watermark_cfg.get("g_field", {}).get("mapping_mode", "binary"),
            },
            "alpha_schedule": [],  # To be populated from bias config
            "mask_id": self._watermark_cfg.get("mask", {}).get("mask_id", "none"),
            "mode": self._watermark_cfg.get("bias", {}).get("mode", "non-distortionary"),
            "experiment_id": watermark_cfg.get("experiment_id", "exp"),
            "timestamp": timestamp,
            "num_inference_steps": num_steps,
            "guidance_scale": cfg_scale,
        }
        
        return manifest
    
    def generate(
        self,
        prompt: str,
        key_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate watermarked image with manifest metadata.
        
        Args:
            prompt: Text prompt for image generation
            key_config: Optional key configuration (overrides config)
            seed: Random seed for reproducibility
            num_inference_steps: Number of DDIM steps (overrides config)
            guidance_scale: CFG guidance scale (overrides config)
            negative_prompt: Optional negative prompt
        
        Returns:
            Tuple of (generated_image, manifest_dict)
        """
        if self._pipeline is None:
            raise ValueError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Override config if provided
        num_steps = num_inference_steps or self._diffusion_cfg["inference_timesteps"]
        cfg_scale = guidance_scale or self._diffusion_cfg["guidance_scale"]
        
        # Update scheduler timesteps if num_steps changed
        if num_steps != self._diffusion_cfg["inference_timesteps"]:
            self._pipeline.scheduler.set_timesteps(num_inference_steps=num_steps)
            # Recreate timestep mapper if steps changed
            self._timestep_mapper = TimestepMapper(
                trained_timesteps=self._diffusion_cfg["trained_timesteps"],
                inference_timesteps=num_steps
            )
        
        # Use key_config if provided, otherwise use config
        final_key_config = key_config if key_config is not None else self._watermark_cfg.get("watermark", {})
        
        # Generate image using pipeline
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=cfg_scale,
            generator=generator,
            eta=0.0,  # Deterministic DDIM
            output_type="pil"
        )
        
        image = output.images[0]
        
        # Create manifest
        manifest = self._create_manifest(prompt, final_key_config, seed, num_steps, cfg_scale)
        
        return image, manifest
    
    def save_manifest(self, sample_metadata: Dict[str, Any], path: str) -> None:
        """Persist manifest as JSON."""
        ensure_dir(path)
        ManifestIO.write_json(path, sample_metadata)
