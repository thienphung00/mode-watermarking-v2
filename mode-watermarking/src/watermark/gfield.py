"""
G-field construction for per-timestep watermark bias tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from ..config.mode_config import ModeConfig
from ..sd_integration.timestep_mapper import TimestepMapper
from .g_utils import extract_bits_to_gvalues, time_dependent_offset


@dataclass
class GFieldBuilder:
	mapping_mode: str = "binary"  # or "continuous"
	bit_pos: int = 30

	def build_gfield(
		self,
		seed: int,
		shape: Tuple[int, int, int],  # (C,H,W)
		t: int,
		mapping_mode: Optional[str] = None,
		content_mix_fn=None,
		key_stream: Optional[Iterable[int]] = None,
	) -> np.ndarray:
		"""
		Construct a g-field G_t[c,h,w] from a key stream at timestep t.
		"""
		C, H, W = shape
		L = C * H * W
		start = time_dependent_offset(t, L)
		idxs = np.arange(L, dtype=np.int64) + start
		if key_stream is None:
			raise ValueError("key_stream is required to build gfield")
		# Materialize needed window from stream
		h_vals = np.fromiter(key_stream, dtype=np.uint64, count=L)
		mapping = (mapping_mode or self.mapping_mode)
		g_values = extract_bits_to_gvalues(h_vals.tolist(), mapping=mapping, bit_pos=self.bit_pos)
		G = g_values.reshape(C, H, W)
		return G.astype(np.float32)

	def build_g_schedule(
		self,
		timestep_mapper: TimestepMapper,
		latent_shape: Tuple[int, int, int],
		key_stream: Iterable[int],
	) -> Dict[int, np.ndarray]:
		"""
		Build g-fields only for actual DDIM timesteps (50, not 1000).
		
		Args:
			timestep_mapper: Maps inference steps to trained timesteps
			latent_shape: [C, H, W] = [4, 64, 64]
			key_stream: LCG-based key stream
		
		Returns:
			Dict mapping trained_timestep -> G_t tensor
		"""
		schedule: Dict[int, np.ndarray] = {}
		
		# Get all mapped timesteps (50 values between 0-999)
		trained_timesteps = timestep_mapper.get_all_trained_timesteps()
		
		for trained_t in trained_timesteps:
			G_t = self.build_gfield(
				seed=0,
				shape=latent_shape,
				t=trained_t,  # Use trained timestep for offset
				mapping_mode=self.mapping_mode,
				key_stream=key_stream,
			)
			# Verify zero-mean property (for non-distortionary)
			if not self.verify_zero_mean(G_t, tolerance=1e-3):
				print(f"Warning: G_t at timestep {trained_t} has non-zero mean: {np.mean(G_t):.6f}")
			schedule[trained_t] = G_t
		
		return schedule
	
	def verify_zero_mean(self, G_t: np.ndarray, tolerance: float = 1e-6) -> bool:
		"""
		Verify E[G_t] ≈ 0 for non-distortionary property.
		
		Args:
			G_t: G-field tensor [C, H, W]
			tolerance: Acceptable deviation from zero
		
		Returns:
			True if mean is within tolerance of zero
		"""
		mean = np.mean(G_t)
		return abs(mean) < tolerance

