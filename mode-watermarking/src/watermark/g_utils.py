"""
Pseudo-random utilities for LCG-based g-field generation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import math

# Optional torch import - only needed if torch tensors are passed
try:
	import torch
	_TORCH_AVAILABLE = True
except ImportError:
	_TORCH_AVAILABLE = False
	torch = None  # type: ignore

def lcg_accumulate(
	seed0: int,
	num_steps: int,
	a: int,
	c: int,
	m: int,
	mix_words: Optional[List[int]] = None,
) -> List[int]:
	"""
	Accumulate LCG states with optional additive mixing per step.
	
	Formula: h_{i+1} = (a*h_i + c + mix_i) mod m
	"""
	h = seed0 % m
	stream: List[int] = []
	for i in range(num_steps):
		mix = (mix_words[i] if mix_words and i < len(mix_words) else 0) % m
		h = (a * h + c + mix) % m
		stream.append(h)
	return stream


def extract_bits_to_gvalues(
	h_stream: List[int],
	mapping: str = "binary",
	bit_pos: int = 30,
	normalize: bool = True,
) -> np.ndarray:
	"""
	Map LCG states to g-values (binary ±1 or continuous in [-1,1]).
	"""
	m = float(2 ** 64)
	arr = np.array(h_stream, dtype=np.uint64)
	if mapping == "binary":
		bits = ((arr >> np.uint64(bit_pos)) & np.uint64(1)).astype(np.int32)
		g = (2 * bits - 1).astype(np.float32)
		return g
	elif mapping == "continuous":
		u = arr.astype(np.float64) / m
		g = 2.0 * u - 1.0
		return g.astype(np.float32)
	else:
		raise ValueError(f"Unknown mapping: {mapping}")


def mix_content_with_latent(h_val: int, z_patch: np.ndarray, kappa: float) -> int:
	"""
	Optionally mix content hash with LCG value to content-condition g-fields.
	Returns a 64-bit integer.
	"""
	z_sum = float(np.sum(z_patch))
	delta = int((abs(z_sum) * kappa)) & 0xFFFFFFFFFFFFFFFF
	return (h_val ^ delta) & 0xFFFFFFFFFFFFFFFF


def time_dependent_offset(t: int, L: int) -> int:
	"""
	Return index offset for timestep t with flat length L.
	"""
	return max(0, t) * max(1, L)


def generate_adaptive_schedule(
	num_steps: int,
	peak_timestep: float,
	injection_start: float,
	injection_end: float,
	strength_range: Tuple[float, float],
	schedule_type: str = "linear"
) -> List[float]:
	"""
	Generate alpha schedule with shape:
	- 0.0 for t > injection_start (high noise, early steps)
	- Ramp up from injection_start to peak_timestep
	- Peak at peak_timestep
	- Ramp down from peak_timestep to injection_end
	- 0.0 for t < injection_end (low noise, late steps)
	
	Args:
		num_steps: Number of inference steps (e.g., 50 for DDIM)
		peak_timestep: Peak position (0.0-1.0, e.g., 0.4 = 40% of schedule)
		injection_start: Start injection position (0.0-1.0, e.g., 0.8 = 80%)
		injection_end: End injection position (0.0-1.0, e.g., 0.2 = 20%)
		strength_range: (min_strength, max_strength) tuple, e.g., (0.005, 0.025)
		schedule_type: "linear", "cosine", or "exponential"
	
	Returns:
		List of length num_steps
		Note: DDIM counts down from num_steps-1 to 0, so schedule[0] corresponds
		to the last (low noise) step and schedule[num_steps-1] to the first (high noise) step.
	"""
	schedule = [0.0] * num_steps
	min_strength, max_strength = strength_range
	
	# Convert normalized positions to step indices
	# DDIM steps are in reverse (999 -> 0 for trained, num_steps-1 -> 0 for inference)
	# injection_start (0.8) means high noise (early in diffusion, later in array)
	# injection_end (0.2) means low noise (late in diffusion, early in array)
	step_start = int((1.0 - injection_start) * num_steps)
	step_peak = int((1.0 - peak_timestep) * num_steps)
	step_end = int((1.0 - injection_end) * num_steps)
	
	# Ensure valid indices
	step_start = max(0, min(step_start, num_steps - 1))
	step_peak = max(0, min(step_peak, num_steps - 1))
	step_end = max(0, min(step_end, num_steps - 1))
	
	if schedule_type == "linear":
		# Ramp up phase (from injection_start to peak)
		if step_start > step_peak and step_start > 0:
			for i in range(step_peak, step_start + 1):
				if step_start != step_peak:
					t = (i - step_peak) / (step_start - step_peak)
				else:
					t = 1.0
				schedule[num_steps - 1 - i] = min_strength + t * (max_strength - min_strength)
		
		# Peak phase
		if step_peak >= 0:
			schedule[num_steps - 1 - step_peak] = max_strength
		
		# Ramp down phase (from peak to injection_end)
		if step_end < step_peak and step_peak > step_end:
			for i in range(step_end, step_peak):
				if step_peak != step_end:
					t = (step_peak - i) / (step_peak - step_end)
				else:
					t = 1.0
				schedule[num_steps - 1 - i] = min_strength + t * (max_strength - min_strength)
	
	elif schedule_type == "cosine":
		# Cosine interpolation for smoother transitions
		
		# Ramp up phase
		if step_start > step_peak and step_start > 0:
			for i in range(step_peak, step_start + 1):
				if step_start != step_peak:
					t = (i - step_peak) / (step_start - step_peak)
				else:
					t = 1.0
				t_cos = 0.5 * (1 - math.cos(math.pi * t))
				schedule[num_steps - 1 - i] = min_strength + t_cos * (max_strength - min_strength)
		
		# Peak
		if step_peak >= 0:
			schedule[num_steps - 1 - step_peak] = max_strength
		
		# Ramp down phase
		if step_end < step_peak and step_peak > step_end:
			for i in range(step_end, step_peak):
				if step_peak != step_end:
					t = (step_peak - i) / (step_peak - step_end)
				else:
					t = 1.0
				t_cos = 0.5 * (1 - math.cos(math.pi * t))
				schedule[num_steps - 1 - i] = min_strength + t_cos * (max_strength - min_strength)
	
	else:
		raise ValueError(f"Unknown schedule_type: {schedule_type}")
	
	return schedule


def generate_fixed_schedule(
	num_steps: int,
	fixed_values: List[float]
) -> List[float]:
	"""
	Interpolate fixed schedule to num_steps using linear interpolation.
	
	Args:
		num_steps: Target number of steps
		fixed_values: List of fixed alpha values
	
	Returns:
		List of length num_steps with interpolated values
	"""
	if len(fixed_values) == num_steps:
		return fixed_values
	
	if len(fixed_values) == 0:
		return [0.0] * num_steps
	
	# Interpolate
	indices = np.linspace(0, len(fixed_values) - 1, num_steps)
	schedule = np.interp(indices, range(len(fixed_values)), fixed_values)
	return schedule.tolist()


def compute_alpha_bar_t(
	timestep: int,
	beta_start: float,
	beta_end: float,
	num_timesteps: int = 1000,
	schedule_type: str = "linear"
) -> float:
	"""
	Compute α̅ₜ = ∏(1-β_s) from beta schedule.
	
	For linear schedule: β_t = β_start + (t/T) * (β_end - β_start)
	Then α̅ₜ = ∏_{s=1}^{t} (1 - β_s)
	
	Args:
		timestep: Timestep index (0 to num_timesteps-1)
		beta_start: Starting beta value
		beta_end: Ending beta value
		num_timesteps: Total number of timesteps
		schedule_type: "linear" or other (currently only linear supported)
	
	Returns:
		Cumulative product α̅ₜ
	"""
	if schedule_type == "linear":
		# Linear interpolation of betas
		betas = np.linspace(beta_start, beta_end, num_timesteps)
		alphas = 1.0 - betas
		# Cumulative product up to timestep
		alpha_bar_t = np.prod(alphas[:timestep + 1])
		return float(alpha_bar_t)
	else:
		raise ValueError(f"Unsupported schedule_type: {schedule_type}")


def compute_latent_noise_energy_from_schedule(
	alpha_bar_t: float,
	latent_shape: Tuple[int, int, int]
) -> float:
	"""
	Compute expected latent noise energy from diffusion schedule.
	
	E[||latent_noise||²] = (1 - α̅ₜ) * N
	where N = C * H * W (number of elements in latent)
	
	Args:
		alpha_bar_t: Cumulative product α̅ₜ at timestep t
		latent_shape: Latent shape [C, H, W]
	
	Returns:
		Expected noise energy
	"""
	C, H, W = latent_shape
	N = C * H * W
	noise_energy = (1.0 - alpha_bar_t) * N
	return float(noise_energy)


def compute_latent_noise_energy_from_sample(
	sampled_noise: Union[np.ndarray, torch.Tensor],
	alpha_bar_t: float
) -> float:
	"""
	Compute latent noise energy from actual sampled noise.
	
	||latent_noise||² = ||sqrt(1 - α̅ₜ) * ε||² = (1 - α̅ₜ) * ||ε||²
	
	Args:
		sampled_noise: Sampled noise tensor ε [B, C, H, W] or [C, H, W]
		alpha_bar_t: Cumulative product α̅ₜ at timestep t
	
	Returns:
		Measured noise energy
	"""
	# Convert to numpy if torch tensor
	if _TORCH_AVAILABLE and isinstance(sampled_noise, torch.Tensor):
		noise_np = sampled_noise.detach().cpu().numpy()
	else:
		noise_np = np.asarray(sampled_noise)
	
	# Compute ||ε||²
	epsilon_squared = np.sum(noise_np ** 2)
	
	# Scale by (1 - α̅ₜ)
	noise_energy = (1.0 - alpha_bar_t) * epsilon_squared
	return float(noise_energy)


def compute_g_field_energy(
	g_field: Optional[Union[np.ndarray, torch.Tensor]] = None,
	g_field_shape: Optional[Tuple[int, int, int]] = None,
	g_field_density: float = 1.0
) -> float:
	"""
	Compute g-field energy ||G||².
	
	If g_field tensor provided: compute ||G||² = sum(G²)
	Else if g_field_shape provided: use analytical ||G||² = K * density
	where K = C * H * W for binary ±1 values
	
	Args:
		g_field: Actual G tensor [C, H, W] or [B, C, H, W]
		g_field_shape: Shape [C, H, W] if g_field not provided
		g_field_density: Fraction of nonzero entries (default: 1.0 = full density)
	
	Returns:
		G-field energy ||G||²
	"""
	if g_field is not None:
		# Compute from actual tensor
		if _TORCH_AVAILABLE and isinstance(g_field, torch.Tensor):
			g_np = g_field.detach().cpu().numpy()
		else:
			g_np = np.asarray(g_field)
		
		# Sum of squares: ||G||² = sum(G²)
		g_energy = float(np.sum(g_np ** 2))
		return g_energy
	
	elif g_field_shape is not None:
		# Analytical computation for binary ±1
		# ||G||² = K where K = number of nonzero entries
		C, H, W = g_field_shape
		K = int(C * H * W * g_field_density)
		g_energy = float(K)  # For binary ±1, each entry contributes 1² = 1
		return g_energy
	
	else:
		raise ValueError("Either g_field or g_field_shape must be provided")


def verify_alpha_max(
	mode: str,
	target_snr: float = 0.005,
	g_field: Optional[Union[np.ndarray, torch.Tensor]] = None,
	g_field_shape: Optional[Tuple[int, int, int]] = None,
	g_field_density: float = 1.0,
	timestep: Optional[int] = None,
	alpha_bar_t: Optional[float] = None,
	latent_shape: Tuple[int, int, int] = (4, 64, 64),
	noise_estimate_method: str = "schedule",
	sampled_noise: Optional[Union[np.ndarray, torch.Tensor]] = None,
	beta_start: float = 0.00090,
	beta_end: float = 0.0100,
	num_timesteps: int = 1000,
	config_strength_range: Optional[Tuple[float, float]] = None,
	verbose: bool = True
) -> Dict[str, Any]:
	"""
	Verify maximum alpha (α_max) for watermark embedding based on target SNR.
	
	Computes α_max such that: ||α * G||² ≈ γ * ||latent_noise||²
	where γ = target_snr (default 0.5% = 0.005)
	
	Formula: α_max = sqrt(γ * ||latent_noise||² / ||G||²)
	
	Args:
		mode: "non_distortionary" or "distortionary"
		target_snr: Target signal-to-noise ratio (default: 0.005 = 0.5%)
		g_field: Actual G tensor [C, H, W] (optional)
		g_field_shape: Shape [C, H, W] if g_field not provided
		g_field_density: Fraction of nonzero entries (default: 1.0)
		timestep: Specific timestep for per-timestep calculation (optional)
		alpha_bar_t: Pre-computed α̅ₜ value (optional, computed if not provided)
		latent_shape: Latent shape [C, H, W] (default: [4, 64, 64])
		noise_estimate_method: "schedule", "sample", "empirical", or "all"
		sampled_noise: Sampled noise tensor for validation (optional)
		beta_start: Starting beta for schedule (default: 0.00085)
		beta_end: Ending beta for schedule (default: 0.0120)
		num_timesteps: Total timesteps (default: 1000)
		config_strength_range: Current config [min, max] for validation (optional)
		verbose: Print warnings if True (default: True)
	
	Returns:
		Dictionary with:
		- alpha_max: Computed maximum alpha
		- g_field_energy: ||G||²
		- latent_noise_energy: ||latent_noise||²
		- target_snr: γ
		- mode: Input mode
		- is_valid: True if alpha_max in acceptable range
		- acceptable_range: (min, max) for this mode
		- warning: Warning message if invalid
		- computation_method: Method used ("schedule", "sample", "empirical", "hybrid")
	"""
	# Validate mode
	if mode not in ["non_distortionary", "distortionary"]:
		raise ValueError(f"Invalid mode: {mode}. Must be 'non_distortionary' or 'distortionary'")
	
	# Set acceptable ranges for each mode
	if mode == "non_distortionary":
		acceptable_range = (0.01, 0.03)
	else:  # distortionary
		acceptable_range = (0.05, 0.15)
	
	# Determine timestep for computation
	if timestep is None:
		# Use mid-point timestep (α̅ₜ ≈ 0.5) as default
		timestep = num_timesteps // 2
	
	# Compute α̅ₜ if not provided
	if alpha_bar_t is None:
		alpha_bar_t = compute_alpha_bar_t(timestep, beta_start, beta_end, num_timesteps)
	
	# Compute ||G||²
	g_energy = compute_g_field_energy(g_field, g_field_shape, g_field_density)
	
	# Compute ||latent_noise||²
	latent_noise_energy = None
	computation_method = "schedule"
	validation_samples = None
	
	if noise_estimate_method in ["schedule", "all"]:
		# Analytical estimate from schedule (fast)
		latent_noise_energy = compute_latent_noise_energy_from_schedule(alpha_bar_t, latent_shape)
		computation_method = "schedule"
	
	if noise_estimate_method in ["sample", "all"] and sampled_noise is not None:
		# Validation with actual sample
		sampled_energy = compute_latent_noise_energy_from_sample(sampled_noise, alpha_bar_t)
		if noise_estimate_method == "all":
			# Compare schedule vs sample
			validation_samples = {
				"schedule_energy": latent_noise_energy,
				"sampled_energy": sampled_energy,
				"relative_diff": abs(sampled_energy - latent_noise_energy) / max(latent_noise_energy, 1e-10)
			}
			# Use sampled if available, else fall back to schedule
			latent_noise_energy = sampled_energy
			computation_method = "hybrid"
		else:
			latent_noise_energy = sampled_energy
			computation_method = "sample"
	
	if latent_noise_energy is None:
		# Fallback to empirical estimate (assume α̅ₜ ≈ 0.5)
		C, H, W = latent_shape
		N = C * H * W
		latent_noise_energy = 0.5 * N  # Conservative estimate
		computation_method = "empirical"
	
	# Compute α_max using formula: α_max = sqrt(γ * ||latent_noise||² / ||G||²)
	if g_energy < 1e-10:
		raise ValueError(f"G-field energy too small: {g_energy}. Check g_field or g_field_shape.")
	
	alpha_max = math.sqrt(target_snr * latent_noise_energy / g_energy)
	
	# Validate against acceptable range
	is_valid = acceptable_range[0] <= alpha_max <= acceptable_range[1]
	
	warning = None
	if not is_valid:
		if alpha_max < acceptable_range[0]:
			warning = (
				f"α_max ({alpha_max:.6f}) below acceptable range [{acceptable_range[0]}, {acceptable_range[1]}] "
				f"for {mode} mode. Watermark may be too weak for reliable detection."
			)
		else:
			warning = (
				f"α_max ({alpha_max:.6f}) above acceptable range [{acceptable_range[0]}, {acceptable_range[1]}] "
				f"for {mode} mode. Watermark may cause visible distortion."
			)
		if verbose:
			print(f"Warning: {warning}")
	
	# Validate against config if provided
	config_warning = None
	if config_strength_range is not None:
		config_min, config_max = config_strength_range
		if config_max > alpha_max * 1.1:  # 10% tolerance
			config_warning = (
				f"Config max_strength ({config_max}) exceeds computed α_max ({alpha_max:.6f}). "
				f"Consider reducing to {alpha_max:.6f} or less."
			)
			if verbose:
				print(f"Config Warning: {config_warning}")
	
	result = {
		"alpha_max": float(alpha_max),
		"g_field_energy": float(g_energy),
		"latent_noise_energy": float(latent_noise_energy),
		"target_snr": float(target_snr),
		"mode": mode,
		"is_valid": is_valid,
		"acceptable_range": acceptable_range,
		"warning": warning,
		"config_warning": config_warning,
		"computation_method": computation_method,
		"timestep": timestep,
		"alpha_bar_t": float(alpha_bar_t),
	}
	
	if validation_samples is not None:
		result["validation_samples"] = validation_samples
	
	return result

