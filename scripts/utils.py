"""
Shared utility functions for scripts_v2 scripts.

Provides common helpers for logging, file I/O, and configuration management.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )
    
    logger = logging.getLogger(__name__)
    return logger


def save_metadata(
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save metadata dictionary to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Path to JSON file
        
    Returns:
        Metadata dictionary
    """
    with open(metadata_path, "r") as f:
        return json.load(f)


def save_latent_tensor(
    latent: torch.Tensor,
    output_path: Path,
) -> None:
    """
    Save latent tensor to file.
    
    Args:
        latent: Latent tensor to save
        output_path: Path to save tensor (.pt file)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent.cpu(), output_path)


def load_latent_tensor(latent_path: Path) -> torch.Tensor:
    """
    Load latent tensor from file.
    
    Args:
        latent_path: Path to .pt file
        
    Returns:
        Loaded latent tensor
    """
    return torch.load(latent_path, map_location="cpu")


def load_prompt_list(prompt_file: Path) -> List[str]:
    """
    Load prompts from text file (one per line).
    
    Args:
        prompt_file: Path to text file
        
    Returns:
        List of prompt strings
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_prompt_list(prompts: List[str], output_path: Path) -> None:
    """
    Save prompts to text file (one per line).
    
    Args:
        prompts: List of prompt strings
        output_path: Path to save text file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")


def get_device(device: Optional[str] = None, use_fp16: bool = True) -> str:
    """
    Get appropriate device string.
    
    Args:
        device: Explicit device string (cuda/cpu/mps)
        use_fp16: Whether to use FP16 (requires CUDA)
        
    Returns:
        Device string
    """
    if device:
        return device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def enable_torch_compile(model: torch.nn.Module) -> torch.nn.Module:
    """
    Enable torch.compile for model (if available).
    
    Args:
        model: PyTorch model
        
    Returns:
        Compiled model (or original if compile not available)
    """
    try:
        if hasattr(torch, "compile"):
            return torch.compile(model)
    except Exception:
        pass
    return model


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

