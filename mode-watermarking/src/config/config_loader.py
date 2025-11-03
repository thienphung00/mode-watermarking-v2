"""
Configuration loader for parsing YAML configs and providing typed access.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and parse YAML configuration files."""

    @staticmethod
    def load_yaml(path: str) -> dict:
        """
        Load YAML file and return as dictionary.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Dictionary containing config values
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config if config is not None else {}

    @staticmethod
    def load_diffusion_config(path: str) -> dict:
        """
        Parse diffusion_config.yaml and return relevant parameters.
        
        Args:
            path: Path to diffusion_config.yaml
            
        Returns:
            Dictionary with diffusion parameters:
            - trained_timesteps, inference_timesteps
            - beta_start, beta_end, beta_schedule
            - prediction_type, scheduler
            - guidance_scale
        """
        config = ConfigLoader.load_yaml(path)
        diffusion = config.get("diffusion", {})
        inference = config.get("inference", {})
        
        # Validate required fields
        required = ["trained_timesteps", "inference_timesteps", "beta_start", "beta_end"]
        for field in required:
            if field not in diffusion:
                raise ValueError(f"Missing required field in diffusion config: {field}")
        
        # Validate beta range
        if diffusion["beta_start"] >= diffusion["beta_end"]:
            raise ValueError(
                f"beta_start ({diffusion['beta_start']}) must be < beta_end ({diffusion['beta_end']})"
            )
        
        return {
            "trained_timesteps": diffusion["trained_timesteps"],
            "inference_timesteps": diffusion.get("inference_timesteps", inference.get("num_inference_steps", 50)),
            "beta_start": diffusion["beta_start"],
            "beta_end": diffusion["beta_end"],
            "beta_schedule": diffusion.get("beta_schedule", "linear"),
            "prediction_type": diffusion.get("prediction_type", "epsilon"),
            "scheduler": diffusion.get("scheduler", "DDIM"),
            "scheduler_kwargs": diffusion.get("scheduler_kwargs", {}),
            "guidance_scale": diffusion.get("guidance_scale", inference.get("guidance_scale", 7.5)),
            "guidance_scale_range": diffusion.get("guidance_scale_range", [3.0, 15.0]),
        }

    @staticmethod
    def load_watermark_config(path: str) -> dict:
        """
        Parse watermark_config.yaml and return relevant parameters.
        
        Args:
            path: Path to watermark_config.yaml
            
        Returns:
            Dictionary with watermark parameters:
            - lcg params (a, c, m, bit_pos)
            - g_field config
            - bias config (alpha_schedule, mode)
            - mask config
        """
        config = ConfigLoader.load_yaml(path)
        watermark = config.get("watermark", {})
        bias = config.get("bias", {})
        mask = config.get("mask", {})
        injection = config.get("injection", {})
        
        lcg = watermark.get("lcg", {})
        
        return {
            "watermark": {
                "key_scheme": watermark.get("key_scheme", "LCG-v1"),
                "key_master": watermark.get("key_master", "<private>"),
                "lcg": {
                    "a": lcg.get("a", 6364136223846793005),
                    "c": lcg.get("c", 1442695040888963407),
                    "m": lcg.get("m", 18446744073709551616),
                    "bit_pos": lcg.get("bit_pos", 30),
                },
                "key_id": watermark.get("key_id", "default_key_001"),
                "base_seed": watermark.get("base_seed", 12345),
                "experiment_id": watermark.get("experiment_id", "exp_001"),
                "g_field": watermark.get("g_field", {}),
                "content_hash_strength": watermark.get("content_hash_strength", 0.3),
                "time_dependent_offset": watermark.get("time_dependent_offset", True),
            },
            "bias": {
                "mode": bias.get("mode", "non_distortionary"),
                "alpha_schedule": bias.get("alpha_schedule", {}),
                "adaptive_scaling": bias.get("adaptive_scaling", True),
                "min_strength": bias.get("min_strength", 0.005),
                "max_strength": bias.get("max_strength", 0.025),
            },
            "mask": {
                "enabled": mask.get("enabled", True),
                "mask_id": mask.get("mask_id", "hfreq_1"),
                "mask_type": mask.get("mask_type", "frequency"),
                "frequency_mask": mask.get("frequency_mask", {}),
                "spatial_mask": mask.get("spatial_mask", {}),
            },
            "injection": injection,
        }

    @staticmethod
    def load_model_architecture_config(path: str) -> dict:
        """
        Parse model_architecture.yaml and return relevant parameters.
        
        Args:
            path: Path to model_architecture.yaml
            
        Returns:
            Dictionary with model architecture parameters:
            - unet config
            - vae config
            - text_encoder config
            - attention config
            - memory_optimization config
        """
        config = ConfigLoader.load_yaml(path)
        
        return {
            "model_id": config.get("model_id", "runwayml/stable-diffusion-v1-5"),
            "unet": config.get("unet", {}),
            "vae": config.get("vae", {}),
            "text_encoder": config.get("text_encoder", {}),
            "attention": config.get("attention", {}),
            "memory_optimization": config.get("memory_optimization", {}),
            "use_fp16": config.get("memory_optimization", {}).get("use_fp16", True),
            "gradient_checkpointing": config.get("memory_optimization", {}).get("gradient_checkpointing", False),
        }

    @staticmethod
    def merge_configs(*configs: dict) -> dict:
        """
        Merge multiple config dictionaries.
        
        Args:
            *configs: Variable number of config dictionaries
            
        Returns:
            Merged dictionary (later configs override earlier ones)
        """
        merged = {}
        for config in configs:
            merged.update(config)
        return merged

