"""
Global experiment context for managing state across components.

Provides a centralized place to store experiment configuration, logging,
and shared resources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config import AppConfig


@dataclass
class ExperimentContext:
    """
    Global experiment context.

    Holds configuration, paths, and shared resources for an experiment run.
    """

    config: AppConfig
    experiment_id: str
    output_dir: Path
    device: str = "cuda"

    # Shared resources (lazy-loaded)
    _pipeline: Optional[Any] = field(default=None, repr=False)
    _vae: Optional[Any] = field(default=None, repr=False)
    _logger: Optional[Any] = field(default=None, repr=False)

    # Runtime state
    _metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls, config: AppConfig, experiment_id: str, output_dir: str | Path, device: str = "cuda"
    ) -> "ExperimentContext":
        """
        Create context from configuration.

        Args:
            config: Application configuration
            experiment_id: Unique experiment identifier
            output_dir: Output directory path
            device: Device to use (cuda/cpu)

        Returns:
            ExperimentContext instance
        """
        return cls(
            config=config,
            experiment_id=experiment_id,
            output_dir=Path(output_dir),
            device=device,
        )

    @property
    def pipeline(self) -> Any:
        """Get or create the diffusion pipeline (lazy-loaded)."""
        if self._pipeline is None:
            from ..engine.pipeline import create_pipeline

            self._pipeline = create_pipeline(self.config.diffusion, device=self.device)
        return self._pipeline

    @property
    def vae(self) -> Any:
        """Get VAE from pipeline (lazy-loaded)."""
        if self._vae is None:
            self._vae = self.pipeline.vae
        return self._vae

    @property
    def logger(self) -> Any:
        """Get or create experiment logger (lazy-loaded)."""
        if self._logger is None:
            import logging
            
            # Create simple logger
            self._logger = logging.getLogger(self.experiment_id)
            self._logger.setLevel(logging.INFO)
            
            # Add file handler
            log_dir = self.output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_dir / f"{self.experiment_id}.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self._logger.addHandler(handler)
        return self._logger

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata key-value pair."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self._metadata.get(key, default)

    def get_all_metadata(self) -> Dict[str, Any]:
        """Get all metadata as dictionary."""
        return self._metadata.copy()

    def cleanup(self) -> None:
        """Cleanup resources."""
        # Clean up pipeline
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        # Clean up VAE
        if self._vae is not None:
            del self._vae
            self._vae = None

        # Clean up logger
        if self._logger is not None:
            if hasattr(self._logger, "close"):
                self._logger.close()
            self._logger = None

