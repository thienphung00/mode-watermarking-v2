"""
Noise prediction biasing for watermark embedding during denoising.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .timestep_mapper import TimestepMapper


def apply_bias(
	pred_noise: torch.Tensor,
	G_t: torch.Tensor,
	M: torch.Tensor,
	b_t: float,
	mode: str = "non_distortionary"
) -> torch.Tensor:
	"""
	Apply additive bias to predicted noise: eps' = eps + b_t * (M ⊙ G_t)
	
	Args:
		pred_noise: Predicted noise [B, C, H, W]
		G_t: Watermark tensor [B, C, H, W]
		M: Mask tensor [B, C, H, W]
		b_t: Bias strength scalar
		mode: Embedding mode ("non_distortionary" or "distortionary")
	
	Returns:
		Biased noise prediction
	"""
	# Compute bias
	bias = b_t * (M * G_t)
	
	# Mode-specific adjustments
	if mode == "non_distortionary":
		# Verify zero-mean property (debug mode)
		bias_mean = torch.mean(bias).item()
		if abs(bias_mean) > 0.01:
			print(f"Warning: Non-zero bias mean: {bias_mean:.6f} (expected ~0 for non-distortionary)")
	
	# Apply bias
	return pred_noise + bias


def biased_denoiser_hook(
	eps_t: torch.Tensor,
	timestep: Union[torch.Tensor, int],
	timestep_mapper: TimestepMapper,
	gfield_cache: Dict[int, Union[torch.Tensor, np.ndarray]],
	mask: Union[torch.Tensor, np.ndarray],
	alpha_schedule: List[float],
	mode: str = "non_distortionary"
) -> torch.Tensor:
	"""
	Hook called each DDIM step to apply watermark bias.
	
	Args:
		eps_t: Predicted noise [B, C, H, W]
		timestep: DDIM timestep (tensor or int from scheduler)
		timestep_mapper: Maps DDIM step to trained timestep
		gfield_cache: Dict[trained_timestep -> G_t]
		mask: Spatial mask [C, H, W] or [B, C, H, W]
		alpha_schedule: Bias strength per inference step
		mode: "non_distortionary" or "distortionary"
	
	Returns:
		Modified noise prediction
	"""
	# Convert timestep to int (handle both tensor and int)
	if isinstance(timestep, torch.Tensor):
		# Diffusers scheduler provides timestep as tensor with single value
		if timestep.numel() == 1:
			t_inference = int(timestep.item())
		else:
			# Batch of timesteps - use first one (they should be the same)
			t_inference = int(timestep.flatten()[0].item())
	else:
		t_inference = int(timestep)
	
	# DDIM scheduler provides timesteps in reverse order
	# Map to trained timestep for g-field lookup
	t_trained = timestep_mapper.map_to_trained(t_inference)
	
	# Get bias strength for this inference step
	# alpha_schedule is indexed by inference step
	# Need to find which index in alpha_schedule corresponds to t_inference
	# Since DDIM counts down, we need to map correctly
	if t_inference < len(alpha_schedule):
		# Map inference timestep to schedule index
		# DDIM scheduler timesteps are ordered [highest, ..., lowest]
		# alpha_schedule[0] is for the last (lowest noise) step
		# So we need to reverse index
		num_steps = len(alpha_schedule)
		schedule_idx = num_steps - 1 - t_inference if t_inference < num_steps else 0
		b_t = float(alpha_schedule[schedule_idx])
	else:
		b_t = 0.0
	
	# Skip if bias is zero (outside injection range)
	if abs(b_t) < 1e-8:
		return eps_t
	
	# Get G_t for this trained timestep
	if t_trained not in gfield_cache:
		return eps_t  # Skip if G_t not cached
	
	G_t = gfield_cache[t_trained]
	
	# Convert to tensor if numpy
	if isinstance(G_t, np.ndarray):
		G_t = torch.from_numpy(G_t).to(eps_t.device).to(eps_t.dtype)
	
	# Convert mask to tensor if numpy
	if isinstance(mask, np.ndarray):
		mask = torch.from_numpy(mask).to(eps_t.device).to(eps_t.dtype)
	
	# Expand dimensions to match batch
	if G_t.dim() == 3:
		G_t = G_t.unsqueeze(0).expand_as(eps_t)
	if mask.dim() == 3:
		mask = mask.unsqueeze(0).expand_as(eps_t)
	
	# Ensure same device and dtype
	G_t = G_t.to(eps_t.device).to(eps_t.dtype)
	mask = mask.to(eps_t.device).to(eps_t.dtype)
	
	# Apply bias
	return apply_bias(
		eps_t,
		G_t,
		mask,
		b_t,
		mode=mode
	)
