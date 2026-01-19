"""
Alpha scheduling for watermark strength based on target SNR.

Contains:
- AlphaScheduler: Computes alpha schedules for watermark injection
- SchedulingFactory: Creates scheduler instances from configuration (scheduler-agnostic)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


class AlphaScheduler:
    """
    Computes alpha schedules for watermark injection based on target SNR.

    Alpha controls watermark strength: eps' = eps + alpha_t * (M ⊙ G_t)
    """

    # Mode profiles: (target_snr, alpha_bounds)
    MODE_PROFILES = {
        "non_distortionary": {
            "target_snr": 0.05,  # 5%
            "alpha_bounds": (0.0, 0.08),
        },
        "distortionary": {
            "target_snr": 0.15,  # 15%
            "alpha_bounds": (0.05, 0.25),
        },
    }

    def __init__(
        self,
        mode: str = "non_distortionary",
        target_snr: Optional[float] = None,
        alpha_bounds: Optional[Tuple[float, float]] = None,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        num_diffusion_steps: int = 1000,
    ):
        """
        Initialize alpha scheduler.

        Args:
            mode: "non_distortionary" or "distortionary"
            target_snr: Target SNR (overrides mode default if provided)
            alpha_bounds: (min, max) bounds for alpha (overrides mode default)
            beta_start: Beta schedule start
            beta_end: Beta schedule end
            num_diffusion_steps: Number of diffusion timesteps
        """
        self.mode = mode.lower()
        if self.mode not in self.MODE_PROFILES:
            raise ValueError(
                f"mode must be one of {list(self.MODE_PROFILES.keys())}, got '{mode}'"
            )

        profile = self.MODE_PROFILES[self.mode]

        # Use provided values or defaults from profile
        self.target_snr = target_snr if target_snr is not None else profile["target_snr"]
        self.alpha_bounds = (
            alpha_bounds if alpha_bounds is not None else profile["alpha_bounds"]
        )

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_diffusion_steps = num_diffusion_steps

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.target_snr <= 0:
            raise ValueError(f"target_snr must be positive, got {self.target_snr}")

        alpha_min, alpha_max = self.alpha_bounds
        if alpha_min < 0 or alpha_max < alpha_min:
            raise ValueError(
                f"alpha_bounds must satisfy 0 <= min <= max, got {self.alpha_bounds}"
            )

        if self.beta_start >= self.beta_end:
            raise ValueError(
                f"beta_start must be < beta_end, got {self.beta_start}, {self.beta_end}"
            )

        if self.num_diffusion_steps <= 0:
            raise ValueError(
                f"num_diffusion_steps must be positive, got {self.num_diffusion_steps}"
            )

    def generate_schedule(
        self,
        timesteps: list[int],
        g_field_energies: Dict[int, float],
        latent_shape: Tuple[int, int, int] = (4, 64, 64),
        envelope_config: Optional[Dict] = None,
    ) -> Dict[int, float]:
        """
        Generate alpha schedule for given timesteps.

        Args:
            timesteps: List of trained timesteps (e.g., [999, 979, 959, ...])
            g_field_energies: Dictionary mapping timestep -> ||G_t||^2
            latent_shape: Latent tensor shape [C, H, W]
            envelope_config: Optional envelope shaping config

        Returns:
            Dictionary mapping timestep -> alpha_t
        """
        num_steps = len(timesteps)
        alpha_schedule = {}

        for idx, t in enumerate(timesteps):
            # Get G-field energy for this timestep
            g_energy = g_field_energies.get(t)
            if g_energy is None:
                raise ValueError(f"G-field energy not provided for timestep {t}")

            # Compute alpha_bar_t (cumulative product of alphas from diffusion)
            alpha_bar = self.compute_alpha_bar_t(t)

            # Compute expected latent noise energy
            latent_noise_energy = self.compute_latent_noise_energy(alpha_bar, latent_shape)

            # Compute alpha from SNR formula: SNR = (alpha^2 * ||G||^2) / E[||noise||^2]
            # => alpha = sqrt(SNR * E[||noise||^2] / ||G||^2)
            alpha = self._compute_alpha_from_snr(
                self.target_snr, latent_noise_energy, g_energy
            )

            # Clamp to bounds
            alpha = np.clip(alpha, self.alpha_bounds[0], self.alpha_bounds[1])

            alpha_schedule[t] = float(alpha)

        # Apply envelope shaping if requested
        if envelope_config is not None:
            alpha_schedule = self._apply_envelope(alpha_schedule, timesteps, envelope_config)

        return alpha_schedule

    def compute_alpha_bar_t(self, timestep: int) -> float:
        """
        Compute cumulative alpha_bar_t from beta schedule.

        alpha_bar_t = ∏(1 - beta_s) for s=1 to t

        Args:
            timestep: Timestep index (0 to num_diffusion_steps-1)

        Returns:
            Cumulative product alpha_bar_t
        """
        if not 0 <= timestep < self.num_diffusion_steps:
            raise ValueError(
                f"timestep must be in [0, {self.num_diffusion_steps-1}], got {timestep}"
            )

        # Linear beta schedule
        betas = np.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps)
        alphas = 1.0 - betas

        # Cumulative product up to timestep
        alpha_bar_t = np.prod(alphas[: timestep + 1])
        return float(alpha_bar_t)

    @staticmethod
    def compute_latent_noise_energy(
        alpha_bar_t: float, latent_shape: Tuple[int, int, int]
    ) -> float:
        """
        Compute expected latent noise energy.

        E[||latent_noise||^2] = (1 - alpha_bar_t) * N
        where N = C * H * W

        Args:
            alpha_bar_t: Cumulative alpha_bar_t
            latent_shape: Latent shape [C, H, W]

        Returns:
            Expected noise energy
        """
        if not 0.0 <= alpha_bar_t <= 1.0:
            raise ValueError(f"alpha_bar_t must be in [0, 1], got {alpha_bar_t}")

        C, H, W = latent_shape
        N = C * H * W
        noise_energy = (1.0 - alpha_bar_t) * N
        return float(noise_energy)

    def _compute_alpha_from_snr(
        self, target_snr: float, latent_noise_energy: float, g_energy: float
    ) -> float:
        """
        Compute alpha from SNR formula.

        SNR = (alpha^2 * ||G||^2) / E[||noise||^2]
        => alpha = sqrt(SNR * E[||noise||^2] / ||G||^2)

        Args:
            target_snr: Target SNR ratio
            latent_noise_energy: Expected latent noise energy
            g_energy: G-field energy ||G||^2

        Returns:
            Computed alpha value
        """
        eps = 1e-12
        g_energy_safe = max(g_energy, eps)

        if latent_noise_energy <= eps or target_snr <= 0.0:
            return 0.0

        alpha = math.sqrt(target_snr * latent_noise_energy / g_energy_safe)
        return float(alpha)

    def _apply_envelope(
        self,
        alpha_schedule: Dict[int, float],
        timesteps: list[int],
        envelope_config: Dict,
    ) -> Dict[int, float]:
        """
        Apply envelope shaping to alpha schedule.

        Supports triangular envelope for concentrated late injection.

        Args:
            alpha_schedule: Original alpha schedule
            timesteps: List of timesteps
            envelope_config: Envelope configuration

        Returns:
            Modified alpha schedule with envelope applied
        """
        shape = envelope_config.get("shape", "triangular")
        if shape != "triangular":
            raise ValueError(f"Only triangular envelope supported, got '{shape}'")

        start_fraction = envelope_config.get("start_fraction", 0.70)
        peak_fraction = envelope_config.get("peak_fraction", 0.90)
        end_fraction = envelope_config.get("end_fraction", 1.00)

        num_steps = len(timesteps)
        enveloped = {}

        for idx, t in enumerate(timesteps):
            # Compute forward fraction (0 = first step, 1 = last step)
            # timesteps are in reverse order (999, 979, ..., 0)
            # so forward_idx goes from high to low
            forward_idx = num_steps - 1 - idx
            forward_fraction = forward_idx / max(num_steps - 1, 1)

            # Compute envelope weight
            if forward_fraction < start_fraction or forward_fraction > end_fraction:
                weight = 0.0
            elif forward_fraction <= peak_fraction:
                # Ramp up
                weight = (forward_fraction - start_fraction) / max(
                    peak_fraction - start_fraction, 1e-9
                )
            else:
                # Ramp down
                weight = 1.0 - (forward_fraction - peak_fraction) / max(
                    end_fraction - peak_fraction, 1e-9
                )

            weight = np.clip(weight, 0.0, 1.0)
            enveloped[t] = alpha_schedule[t] * weight

        return enveloped

    def verify_schedule(
        self,
        alpha_schedule: Dict[int, float],
        tolerance: float = 1e-6,
    ) -> Dict[str, bool]:
        """
        Verify that alpha schedule satisfies constraints.

        Args:
            alpha_schedule: Schedule to verify
            tolerance: Numerical tolerance

        Returns:
            Dictionary of verification results
        """
        results = {}

        # Check all values are within bounds
        alpha_min, alpha_max = self.alpha_bounds
        values = list(alpha_schedule.values())

        results["within_bounds"] = all(
            alpha_min - tolerance <= v <= alpha_max + tolerance for v in values
        )

        # Check all values are non-negative
        results["non_negative"] = all(v >= -tolerance for v in values)

        # Check all values are finite
        results["all_finite"] = all(math.isfinite(v) for v in values)

        return results


# ============================================================================
# Standalone Utility Functions
# ============================================================================


def compute_g_field_energy(
    g_field: Optional[np.ndarray] = None,
    g_field_shape: Optional[Tuple[int, int, int]] = None,
) -> float:
    """
    Compute G-field energy ||G||^2.

    Args:
        g_field: Actual G tensor [C, H, W] (if available)
        g_field_shape: Shape [C, H, W] (if g_field not provided)

    Returns:
        G-field energy
    """
    if g_field is not None:
        # Compute from actual tensor
        return float(np.sum(g_field**2))

    elif g_field_shape is not None:
        # Analytical computation for binary ±1
        # ||G||^2 = K where K = number of elements (each contributes 1^2 = 1)
        C, H, W = g_field_shape
        K = C * H * W
        return float(K)

    else:
        raise ValueError("Either g_field or g_field_shape must be provided")


# ============================================================================
# Scheduler Factory (Scheduler-Agnostic)
# ============================================================================


class SchedulingFactory:
    """
    Factory for creating diffusers schedulers from configuration.
    
    Supports multiple scheduler types:
    - DDIM (DDIMScheduler)
    - UniPC (UniPCMultistepScheduler)
    - DPMSolver (DPMSolverMultistepScheduler)
    - PNDM (PNDMScheduler)
    - Euler (EulerDiscreteScheduler)
    - EulerAncestral (EulerAncestralDiscreteScheduler)
    
    Usage:
        >>> scheduler = SchedulingFactory.create(diffusion_config)
        >>> pipeline.scheduler = scheduler
    """
    
    # Mapping from config names to diffusers scheduler classes
    SCHEDULER_MAP = {
        "ddim": "DDIMScheduler",
        "unipc": "UniPCMultistepScheduler",
        "dpmsolver": "DPMSolverMultistepScheduler",
        "dpmsolver++": "DPMSolverMultistepScheduler",
        "pndm": "PNDMScheduler",
        "euler": "EulerDiscreteScheduler",
        "euler_ancestral": "EulerAncestralDiscreteScheduler",
        "lms": "LMSDiscreteScheduler",
        "heun": "HeunDiscreteScheduler",
    }
    
    @classmethod
    def create(cls, diffusion_config: Any) -> Any:
        """
        Create a scheduler instance from diffusion configuration.
        
        Args:
            diffusion_config: DiffusionConfig Pydantic model with:
                - scheduler: str (e.g., "DDIM", "UniPC", "DPMSolver")
                - trained_timesteps: int
                - beta_start: float
                - beta_end: float
                - beta_schedule: str
                - prediction_type: str
                - scheduler_kwargs: dict (optional)
        
        Returns:
            Configured scheduler instance
        
        Raises:
            ValueError: If scheduler type is not supported
        """
        scheduler_name = diffusion_config.scheduler.lower().replace("-", "").replace("_", "")
        
        # Normalize scheduler name
        if scheduler_name in ("ddim",):
            scheduler_name = "ddim"
        elif scheduler_name in ("unipc", "unipcmultistep"):
            scheduler_name = "unipc"
        elif scheduler_name in ("dpmsolver", "dpmsolvermultistep", "dpmsolver++", "dpm++"):
            scheduler_name = "dpmsolver"
        elif scheduler_name in ("pndm",):
            scheduler_name = "pndm"
        elif scheduler_name in ("euler", "eulerdiscrete"):
            scheduler_name = "euler"
        elif scheduler_name in ("eulerancestral", "eulerancestraldiscrete"):
            scheduler_name = "euler_ancestral"
        elif scheduler_name in ("lms", "lmsdiscrete"):
            scheduler_name = "lms"
        elif scheduler_name in ("heun", "heundiscrete"):
            scheduler_name = "heun"
        
        if scheduler_name not in cls.SCHEDULER_MAP:
            raise ValueError(
                f"Unknown scheduler: '{diffusion_config.scheduler}'. "
                f"Supported: {list(cls.SCHEDULER_MAP.keys())}"
            )
        
        # Import scheduler class from diffusers
        scheduler_class_name = cls.SCHEDULER_MAP[scheduler_name]
        scheduler_class = cls._import_scheduler_class(scheduler_class_name)
        
        # Build scheduler kwargs
        scheduler_kwargs = cls._build_scheduler_kwargs(
            scheduler_name, scheduler_class, diffusion_config
        )
        
        return scheduler_class(**scheduler_kwargs)
    
    @classmethod
    def _import_scheduler_class(cls, class_name: str) -> type:
        """
        Import scheduler class from diffusers.
        
        Args:
            class_name: Name of the scheduler class
        
        Returns:
            Scheduler class
        """
        import diffusers
        
        if not hasattr(diffusers, class_name):
            raise ImportError(
                f"Scheduler class '{class_name}' not found in diffusers. "
                f"Ensure you have a compatible diffusers version installed."
            )
        
        return getattr(diffusers, class_name)
    
    @classmethod
    def _build_scheduler_kwargs(
        cls, 
        scheduler_name: str, 
        scheduler_class: type,
        diffusion_config: Any,
    ) -> Dict[str, Any]:
        """
        Build kwargs for scheduler initialization.
        
        Args:
            scheduler_name: Normalized scheduler name
            scheduler_class: Scheduler class
            diffusion_config: Diffusion configuration
        
        Returns:
            Dictionary of scheduler kwargs
        """
        # Base kwargs common to most schedulers
        base_kwargs = {
            "num_train_timesteps": diffusion_config.trained_timesteps,
            "beta_start": diffusion_config.beta_start,
            "beta_end": diffusion_config.beta_end,
            "beta_schedule": diffusion_config.beta_schedule,
            "prediction_type": diffusion_config.prediction_type,
        }
        
        # Additional kwargs from config
        extra_kwargs = dict(diffusion_config.scheduler_kwargs) if diffusion_config.scheduler_kwargs else {}
        
        # Scheduler-specific defaults
        if scheduler_name == "ddim":
            # DDIM-specific: EXPLICIT configuration for mathematical correctness
            # CRITICAL: Use direct assignment (NOT setdefault) to enforce identical config
            # This prevents silent divergence between generation and inversion
            extra_kwargs["clip_sample"] = False
            extra_kwargs["set_alpha_to_one"] = False
            extra_kwargs["timestep_spacing"] = "leading"
        
        elif scheduler_name == "dpmsolver":
            # DPMSolver-specific: algorithm_type, solver_order
            extra_kwargs.setdefault("algorithm_type", "dpmsolver++")
            extra_kwargs.setdefault("solver_order", 2)
        
        elif scheduler_name == "unipc":
            # UniPC-specific
            extra_kwargs.setdefault("solver_order", 2)
        
        # Merge base and extra kwargs
        kwargs = {**base_kwargs, **extra_kwargs}
        
        # Filter kwargs to only include those accepted by the scheduler
        # This prevents errors from passing unsupported kwargs
        import inspect
        sig = inspect.signature(scheduler_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return filtered_kwargs
    
    @classmethod
    def get_supported_schedulers(cls) -> list[str]:
        """
        Get list of supported scheduler names.
        
        Returns:
            List of supported scheduler names
        """
        return list(cls.SCHEDULER_MAP.keys())

