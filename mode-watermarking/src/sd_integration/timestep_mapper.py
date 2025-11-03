"""
Timestep mapping for DDIM inference steps to original training timesteps.
"""
from __future__ import annotations

from typing import Dict, List


class TimestepMapper:
	"""Map DDIM inference steps to original training timesteps"""
	
	def __init__(
		self,
		trained_timesteps: int = 1000,
		inference_timesteps: int = 50,
		discretization: str = "uniform"  # "uniform", "quadratic"
	):
		"""
		Initialize timestep mapper.
		
		Args:
			trained_timesteps: Number of timesteps SD was trained on (typically 1000)
			inference_timesteps: Number of DDIM inference steps (typically 50)
			discretization: "uniform" for uniform spacing, "quadratic" for quadratic spacing
		"""
		self.trained_timesteps = trained_timesteps
		self.inference_timesteps = inference_timesteps
		self.discretization = discretization
		self._mapping: Dict[int, int] = {}
		self._reverse_mapping: Dict[int, int] = {}
		self._build_mapping()
	
	def _build_mapping(self):
		"""Create mapping: inference_step -> trained_timestep"""
		if self.discretization == "uniform":
			# Uniform spacing: [0, 20, 40, ..., 980] for 50 steps with 1000 trained timesteps
			step_size = self.trained_timesteps // self.inference_timesteps
			for i in range(self.inference_timesteps):
				trained_t = i * step_size
				# DDIM counts down: inference_timesteps-1 is the first (highest noise) step
				# So we map reverse order
				inference_t = self.inference_timesteps - 1 - i
				self._mapping[inference_t] = trained_t
				self._reverse_mapping[trained_t] = inference_t
		elif self.discretization == "quadratic":
			# Quadratic spacing: more steps at high noise (beginning of diffusion)
			# Uses sqrt spacing
			for i in range(self.inference_timesteps):
				# Normalize to [0, 1]
				t_normalized = i / (self.inference_timesteps - 1) if self.inference_timesteps > 1 else 0.0
				# Apply quadratic mapping (more steps at high noise)
				t_quad = t_normalized ** 2
				trained_t = int(t_quad * (self.trained_timesteps - 1))
				inference_t = self.inference_timesteps - 1 - i
				self._mapping[inference_t] = trained_t
				self._reverse_mapping[trained_t] = inference_t
		else:
			raise ValueError(f"Unknown discretization: {self.discretization}")
		
	def map_to_trained(self, inference_step: int) -> int:
		"""
		Map DDIM step (0-49) to trained timestep (0-999).
		
		Args:
			inference_step: DDIM inference step index
		
		Returns:
			Corresponding trained timestep
		"""
		# Clamp to valid range
		inference_step = max(0, min(inference_step, self.inference_timesteps - 1))
		return self._mapping.get(inference_step, inference_step)
		
	def map_to_inference(self, trained_step: int) -> int:
		"""
		Reverse mapping: trained timestep -> inference step (for detection).
		
		Args:
			trained_step: Trained timestep index
		
		Returns:
			Corresponding inference step, or -1 if not found
		"""
		return self._reverse_mapping.get(trained_step, -1)
		
	def get_all_trained_timesteps(self) -> List[int]:
		"""
		Return all mapped trained timesteps (50 values) for g-field caching.
		
		Returns:
			Sorted list of all trained timesteps that will be used during inference
		"""
		return sorted(set(self._mapping.values()))

