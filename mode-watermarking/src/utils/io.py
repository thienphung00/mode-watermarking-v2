"""
Image, array, and manifest I/O utilities for reproducible data operations.

Provides unified interfaces for reading/writing images, arrays, and metadata
across the watermarking pipeline.
"""

import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


class AssetStore:
    """Manages asset paths relative to a root directory."""

    def __init__(self, root_dir: str):
        """
        Initialize asset store.

        Args:
            root_dir: Base directory for all assets
        """
        self.root_dir = Path(root_dir).resolve()

    def resolve(self, *parts: str) -> str:
        """
        Resolve a path relative to root directory.

        Args:
            *parts: Path components

        Returns:
            Absolute path string
        """
        return str(self.root_dir / Path(*parts))


class ImageIO:
    """Handles image reading and writing operations."""

    @staticmethod
    def read_image(path: str, mode: str = "RGB") -> np.ndarray:
        """
        Read an image from disk.

        Args:
            path: Path to image file
            mode: PIL mode (RGB, RGBA, L, etc.)

        Returns:
            Image as numpy array in [H, W, C] format, uint8
        """
        img = Image.open(path).convert(mode)
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def write_image(
        path: str, image: np.ndarray, quality: int = 95
    ) -> None:
        """
        Write an image to disk.

        Args:
            path: Output path
            image: Image array in [H, W, C] format, uint8
            quality: JPEG quality (1-100)
        """
        ensure_dir(path)
        img = Image.fromarray(image.astype(np.uint8))
        img.save(path, quality=quality)


class ArrayIO:
    """Handles numpy array serialization."""

    @staticmethod
    def read_npz(path: str) -> Dict[str, np.ndarray]:
        """
        Read numpy arrays from .npz file.

        Args:
            path: Path to .npz file

        Returns:
            Dictionary of array names to arrays
        """
        return dict(np.load(path))

    @staticmethod
    def write_npz(
        path: str, arrays: Dict[str, np.ndarray]
    ) -> None:
        """
        Write numpy arrays to .npz file.

        Args:
            path: Output path
            arrays: Dictionary of array names to arrays
        """
        ensure_dir(path)
        np.savez(path, **arrays)


class ManifestIO:
    """Handles JSON and line-based manifest file operations."""

    @staticmethod
    def read_json(path: str) -> Union[Dict, List]:
        """
        Read JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON data (dict or list)
        """
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def write_json(
        path: str, data: Union[Dict, List], indent: int = 2
    ) -> None:
        """
        Write JSON file.

        Args:
            path: Output path
            data: Data to serialize (dict or list)
            indent: JSON indentation
        """
        ensure_dir(path)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def read_lines(path: str) -> List[str]:
        """
        Read text file line by line.

        Args:
            path: Path to text file

        Returns:
            List of lines (without trailing newlines)
        """
        with open(path, "r") as f:
            return [line.rstrip("\n") for line in f]

    @staticmethod
    def write_lines(path: str, lines: List[str]) -> None:
        """
        Write lines to text file.

        Args:
            path: Output path
            lines: List of lines to write
        """
        ensure_dir(path)
        with open(path, "w") as f:
            f.write("\n".join(lines))
            if lines:  # Add trailing newline if non-empty
                f.write("\n")


class ReproIO:
    """Utilities for reproducible operations (seeding, temp dirs)."""

    @staticmethod
    def seed_all(seed: int) -> None:
        """
        Set seeds for all random number generators.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # PyTorch not available

    @staticmethod
    @contextmanager
    def with_tempdir(prefix: str):
        """
        Create temporary directory context manager.

        Args:
            prefix: Prefix for temp directory name

        Yields:
            Path string to temporary directory
        """
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        try:
            yield tmpdir
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


def ensure_dir(path: str) -> None:
    """
    Ensure parent directory exists for given path.

    Args:
        path: File or directory path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_images(
    path: str, patterns: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> List[str]:
    """
    List all image files in directory.

    Args:
        path: Directory to search
        patterns: File extension patterns (case-insensitive)

    Returns:
        List of absolute paths to image files
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    image_files = []
    for pattern in patterns:
        image_files.extend(path_obj.glob(f"*{pattern}"))
        image_files.extend(path_obj.glob(f"*{pattern.upper()}"))

    return sorted([str(f.resolve()) for f in image_files])
