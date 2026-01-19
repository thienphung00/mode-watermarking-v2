"""
Abstract base classes and interfaces for the watermarking system.

Implements the Strategy pattern to decouple watermark logic from the diffusion pipeline.

Strategy 1 (Pre-Scheduler Injection):
    noise_pred = unet(latents, t, encoder_hidden_states=...)
    noise_pred = noise_pred + alpha_t * G_t     # Inject HERE (before scheduler)
    latents = scheduler.step(noise_pred, t, latents).prev_sample

The watermark bias is injected into the predicted noise BEFORE the scheduler step,
NOT after. This ensures the scheduler operates on the watermarked noise prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from ..engine.hooks import TimestepMapper, WatermarkHook


# ============================================================================
# Watermark Strategy Pattern
# ============================================================================


class WatermarkStrategy(ABC):
    """
    Abstract strategy for watermark embedding.

    This interface decouples the "how" of watermarking from the pipeline execution.
    Instead of `if mode == 'watermarked'`, we use polymorphism.
    """

    @abstractmethod
    def get_hook(self) -> Optional[Callable]:
        """
        Returns the hook function to be registered in the denoiser.

        Returns:
            Callable that modifies noise predictions, or None for no-op.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata for the manifest (keys, seeds, modes, etc.).

        Returns:
            Dictionary containing watermark metadata to be stored.
        """
        pass

    @abstractmethod
    def prepare_for_sample(self, sample_id: str, prompt: str, **kwargs) -> None:
        """
        Prepare strategy for a new sample generation.

        Args:
            sample_id: Unique sample identifier
            prompt: Text prompt for generation
            **kwargs: Additional sample-specific parameters
        """
        pass


class NullStrategy(WatermarkStrategy):
    """Strategy for unwatermarked mode (no-op)."""

    def get_hook(self) -> Optional[Callable]:
        """Returns None (no hook needed for unwatermarked)."""
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Returns minimal metadata."""
        return {"mode": "unwatermarked"}

    def prepare_for_sample(self, sample_id: str, prompt: str, **kwargs) -> None:
        """No preparation needed for unwatermarked mode."""
        pass


class LatentInjectionStrategy(WatermarkStrategy):
    """
    Strategy for latent-space watermark injection using Strategy 1.

    Strategy 1 (Pre-Scheduler Injection):
        1. UNet predicts noise: noise_pred = unet(latents, t, ...)
        2. Hook modifies noise IN-PLACE: noise_pred.add_(alpha_t * G_t)
        3. Scheduler receives modified prediction: scheduler.step(noise_pred, t, latents)

    This ensures watermark bias is injected BEFORE the scheduler step.
    The scheduler operates on the watermarked noise prediction, NOT on clean noise.

    Supports both distortionary and non-distortionary modes through
    configuration of alpha schedules and masks.
    """

    def __init__(
        self,
        config: Any,  # WatermarkedConfig
        g_schedule: Dict[int, np.ndarray],
        alpha_schedule: Dict[int, float],
        mask: Optional[np.ndarray],
        key_provider: Any,  # PRFKeyDerivation from detection/prf.py
        timestep_mapper: Optional["TimestepMapper"] = None,
        device: str = "cuda",
    ):
        """
        Initialize latent injection strategy.

        Args:
            config: Watermarked configuration
            g_schedule: Pre-computed G-field schedule {trained_timestep: G_t}
                        Keys are TRAINED timesteps (0, 20, 40, ..., 980), NOT scheduler timesteps
            alpha_schedule: Pre-computed alpha schedule {trained_timestep: alpha_t}
                           Keys are TRAINED timesteps, matching g_schedule
            mask: Spatial mask [C, H, W] (optional, can be None)
            key_provider: PRF key derivation provider (from detection/prf.py)
            timestep_mapper: Maps DDIM step_index → trained_timestep for correct G_t lookup
            device: Device to preload tensors on
        """
        self.config = config
        self.g_schedule = g_schedule
        self.alpha_schedule = alpha_schedule
        self.mask = mask
        self.key_provider = key_provider
        self.timestep_mapper = timestep_mapper
        self.device = device

        # State for current sample
        self._current_sample_id: Optional[str] = None
        self._current_prompt: Optional[str] = None
        self._current_seed: Optional[int] = None
        self._current_key_id: Optional[str] = None
        self._current_zT_hash: Optional[str] = None  # Hash of initial latent for detection

        # Cached WatermarkHook instance (created lazily in get_hook)
        self._hook: Optional["WatermarkHook"] = None

    def get_hook(
        self,
        store_intermediates: bool = False,
        timesteps_to_store: Optional[list[int]] = None,
    ) -> Callable:
        """
        Returns the WatermarkHook for pre-scheduler bias injection.

        The hook implements Strategy 1:
            UNet → WatermarkHook (modify noise_pred IN-PLACE) → Scheduler → Next z_t

        The hook:
            - Receives: (step_index, timestep, latents, noise_pred=...)
            - Maps step_index → trained_t via timestep_mapper
            - Looks up G_t = g_schedule[trained_t] and alpha_t = alpha_schedule[trained_t]
            - Applies IN-PLACE: noise_pred.add_(alpha_t * G_t)
            - Returns {"noise_pred": noise_pred} (same tensor, modified)

        Args:
            store_intermediates: If True, store intermediate latents and deltas
            timesteps_to_store: List of step indices to store (None = store all)

        Returns:
            WatermarkHook instance that modifies noise_pred in-place before scheduler step
        """
        # If hook exists but parameters changed, recreate it
        if self._hook is not None:
            if hasattr(self._hook, "store_intermediates"):
                if (
                    self._hook.store_intermediates != store_intermediates
                    or self._hook.timesteps_to_store != timesteps_to_store
                ):
                    # Parameters changed, recreate hook
                    self._hook = None
            else:
                # Old hook without store_intermediates, recreate
                self._hook = None

        if self._hook is not None:
            return self._hook

        # Import here to avoid circular imports
        from ..engine.hooks import WatermarkHook

        if self.timestep_mapper is None:
            raise ValueError(
                "timestep_mapper is required for Strategy 1 injection. "
                "Ensure it is passed during strategy initialization."
            )

        # Create WatermarkHook with pre-converted tensors for efficiency
        self._hook = WatermarkHook(
            g_schedule=self.g_schedule,
            alpha_schedule=self.alpha_schedule,
            mask=self.mask,
            timestep_mapper=self.timestep_mapper,
            device=self.device,
            store_intermediates=store_intermediates,
            timesteps_to_store=timesteps_to_store,
        )

        return self._hook

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns watermark metadata for manifest.
        
        The metadata includes zT_hash for logging purposes only. Detection uses
        PRF-based key derivation (key_id) and does not require zT_hash.
        """
        return {
            "mode": "watermarked",
            "sample_id": self._current_sample_id,
            "sample_seed": self._current_seed,
            "key_id": self._current_key_id,
            "zT_hash": self._current_zT_hash,  # For metadata logging only
            "key_info": {
                "key_id": self.config.key_settings.key_id,
                "experiment_id": self.config.key_settings.experiment_id,
            },
            "algorithm_params": {
                "target_snr": self.config.algorithm_params.bias.target_snr,  # Use actual value from bias config
                "bias_mode": self.config.algorithm_params.bias.mode,
                "alpha_bounds": self.config.algorithm_params.bias.alpha_bounds,
            },
        }

    def prepare_for_sample(
        self, sample_id: str, prompt: str, seed: Optional[int] = None, **kwargs
    ) -> None:
        """
        Prepare strategy for a new sample (interface method).

        Note: For LatentInjectionStrategy, the actual G-field regeneration
        is done by create_per_sample_strategy() in strategy_factory.py.
        This method is kept for interface compatibility but does minimal work.

        Args:
            sample_id: Unique sample identifier
            prompt: Text prompt
            seed: Optional seed override
            **kwargs: Additional parameters
        """
        # Metadata is set by create_per_sample_strategy()
        # This is a no-op to satisfy the interface
        pass


# ============================================================================
# Dataset Interface (for future use)
# ============================================================================


class WatermarkDatasetInterface(ABC):
    """Abstract interface for watermark datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Returns:
            Dictionary with keys: 'image', 'label', 'metadata'
        """
        pass


# ============================================================================
# Detector Interface (for future use)
# ============================================================================


class DetectorInterface(ABC):
    """Abstract interface for watermark detectors."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output logits or predictions
        """
        pass

    @abstractmethod
    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss' and optional metrics
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss' and metrics
        """
        pass

