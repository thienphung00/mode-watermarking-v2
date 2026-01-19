"""
Denoiser hooks for watermark injection (Strategy 1: Pre-Scheduler Injection).

Strategy 1 Pipeline Flow:
    noise_pred = unet(latents, t, encoder_hidden_states=...)
    noise_pred = noise_pred + alpha_t * G_t     # WatermarkHook injects HERE
    latents = scheduler.step(noise_pred, t, latents).prev_sample

The WatermarkHook modifies noise_pred IN-PLACE before the scheduler step.
The scheduler operates on WATERMARKED noise prediction, not clean noise.

Key Components:
    - TimestepMapper: Maps DDIM step_index → trained_timestep
    - WatermarkHook: Performs bias injection via noise_pred.add_(alpha_t * G_t)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class TimestepMapper:
    """
    Maps DDIM inference step indices to original training timesteps.

    Strategy 1 requires correct mapping for G_t and alpha_t lookup:
        step_index → trained_t → g_schedule[trained_t]

    Mapping Logic (for 50 inference steps, 1000 trained timesteps):
        - step_index=0  → trained_t=980 (first step, highest noise)
        - step_index=1  → trained_t=960
        - ...
        - step_index=49 → trained_t=0   (last step, lowest noise)

    The g_schedule and alpha_schedule are keyed by TRAINED timesteps (0, 20, 40, ...),
    NOT scheduler timesteps (999, 979, ...).
    """

    def __init__(
        self,
        trained_timesteps: int = 1000,
        inference_timesteps: int = 50,
        discretization: str = "uniform",
    ):
        """
        Initialize timestep mapper.

        Args:
            trained_timesteps: Number of timesteps SD was trained on (typically 1000)
            inference_timesteps: Number of DDIM inference steps (typically 50)
            discretization: "uniform" for uniform spacing
        """
        self.trained_timesteps = trained_timesteps
        self.inference_timesteps = inference_timesteps
        self.discretization = discretization
        self._mapping: Dict[int, int] = {}
        self._reverse_mapping: Dict[int, int] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        """
        Create mapping: inference_step_index -> trained_timestep.

        For uniform discretization with 50 steps and 1000 trained timesteps:
            step_index  trained_t  (description)
            ---------   ---------  ------------
            0           980        first step, highest noise
            1           960
            ...
            48          20
            49          0          last step, lowest noise

        The trained timesteps are used as keys for g_schedule and alpha_schedule.
        """
        if self.discretization == "uniform":
            # Uniform spacing: trained_t values are [0, 20, 40, ..., 980]
            step_size = self.trained_timesteps // self.inference_timesteps
            for i in range(self.inference_timesteps):
                trained_t = i * step_size
                # DDIM counts down: step_index=0 is first step (highest noise)
                # So step_index=0 → trained_t=980, step_index=49 → trained_t=0
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
        Reverse mapping: trained timestep -> inference step.

        Args:
            trained_step: Trained timestep index

        Returns:
            Corresponding inference step, or -1 if not found
        """
        return self._reverse_mapping.get(trained_step, -1)

    def get_all_trained_timesteps(self) -> List[int]:
        """
        Return all mapped trained timesteps for g-field caching.

        Returns:
            Sorted list of all trained timesteps that will be used during inference
        """
        return sorted(set(self._mapping.values()))


class WatermarkHook:
    """
    Watermark hook for Strategy 1 bias injection during denoising.

    Strategy 1 (Pre-Scheduler Injection):
        Called AFTER UNet predicts noise, BEFORE scheduler step.
        Modifies noise_pred IN-PLACE so scheduler receives watermarked prediction.

    Pipeline flow:
        UNet → WatermarkHook (modify noise_pred) → Scheduler → Next x_t

    Applies additive bias: eps' = eps + alpha_t * (M ⊙ G_t)
    where:
        - eps: Original noise prediction from UNet
        - alpha_t: Bias strength for timestep t (from alpha_schedule)
        - M: Spatial mask (optional)
        - G_t: Watermark tensor for timestep t (from g_schedule)

    Critical Implementation Details:
        - Uses timestep_mapper to convert step_index → trained_t
        - Looks up G_t and alpha_t using trained_t (NOT scheduler timestep)
        - Modifies noise_pred IN-PLACE via .add_() to preserve tensor identity
        - NO cloning/copying of tensors
    """

    def __init__(
        self,
        g_schedule: Dict[int, np.ndarray],
        alpha_schedule: Dict[int, float],
        mask: Optional[np.ndarray],
        timestep_mapper: TimestepMapper,
        device: str = "cuda",
        store_intermediates: bool = False,
        timesteps_to_store: Optional[List[int]] = None,
    ):
        """
        Initialize watermark hook with pre-computed schedules.

        Args:
            g_schedule: Dictionary mapping TRAINED_TIMESTEP → G_t tensor [C, H, W]
                        Keys are trained timesteps (0, 20, 40, ..., 980)
                        NOT scheduler timesteps (999, 979, ...)
            alpha_schedule: Dictionary mapping TRAINED_TIMESTEP → alpha_t scalar
                           Keys match g_schedule
            mask: Spatial mask [C, H, W] (optional, can be None)
            timestep_mapper: Maps step_index → trained_timestep for G_t lookup
            device: Device to preload tensors on
            store_intermediates: If True, store intermediate latents and deltas
            timesteps_to_store: List of step indices to store (None = store all)
        """
        self.g_schedule = g_schedule
        self.alpha_schedule = alpha_schedule
        self.mask = mask
        self.timestep_mapper = timestep_mapper
        self.device = device
        self.store_intermediates = store_intermediates
        self.timesteps_to_store = timesteps_to_store

        # Storage for intermediate latents and deltas
        if self.store_intermediates:
            self.intermediate_latents: Dict[int, torch.Tensor] = {}
            self.intermediate_deltas: Dict[int, torch.Tensor] = {}
            self.cumulative_delta: Optional[torch.Tensor] = None

        # Preload mask tensor to device
        if self.mask is not None:
            self.mask_tensor = torch.from_numpy(self.mask).to(device)
        else:
            self.mask_tensor = None

        # Preload G-field tensors to device for efficiency
        # Keys are trained timesteps: {0: G_0, 20: G_20, ..., 980: G_980}
        self.g_schedule_tensors: Dict[int, torch.Tensor] = {}
        for t, G_t in g_schedule.items():
            self.g_schedule_tensors[t] = torch.from_numpy(G_t).to(device)

    def __call__(
        self,
        step_index: int,
        timestep: int,
        latents: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Hook function for Strategy 1 watermark injection.

        Called by pipeline's hooked UNet forward AFTER noise prediction,
        BEFORE the modified noise_pred is returned to diffusers/scheduler.

        Strategy 1 Flow:
            1. Receive step_index, timestep, latents, noise_pred from pipeline
            2. Map step_index → trained_t using timestep_mapper
            3. Look up G_t = g_schedule[trained_t] and alpha_t = alpha_schedule[trained_t]
            4. Compute bias = alpha_t * (mask ⊙ G_t)
            5. Apply IN-PLACE: noise_pred.add_(bias)
            6. Return {"noise_pred": noise_pred} (same tensor, modified)

        Args:
            step_index: Current DDIM step index (0 = first/highest noise step)
            timestep: Current scheduler timestep value (e.g., 999, 979, ...)
                      NOTE: This is NOT used for G_t lookup; we use step_index
            latents: Current latent tensor [B, C, H, W]
            **kwargs: Must contain "noise_pred" - the UNet output to modify

        Returns:
            Dictionary with "noise_pred" key containing the SAME tensor (modified in-place)
            Returns {} if no modification was made (missing noise_pred or no G_t)
        """
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Extract noise_pred from kwargs
        # ═══════════════════════════════════════════════════════════════════
        noise_pred = kwargs.get("noise_pred")
        if noise_pred is None:
            # Try alternative keys for backwards compatibility
            noise_pred = kwargs.get("pred_original_sample")

        if noise_pred is None:
            # No noise prediction to modify - this shouldn't happen in Strategy 1
            return {}

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Map step_index → trained_t using timestep_mapper
        # CRITICAL: Use step_index, NOT timestep value, for correct mapping
        # ═══════════════════════════════════════════════════════════════════
        trained_t = self.timestep_mapper.map_to_trained(step_index)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: Look up G_t and alpha_t using TRAINED timestep
        # ═══════════════════════════════════════════════════════════════════
        G_t = self.g_schedule_tensors.get(trained_t)
        if G_t is None:
            # No G-field for this timestep (shouldn't happen if schedules match)
            return {}

        alpha_t = self.alpha_schedule.get(trained_t, 0.0)
        if alpha_t == 0.0:
            # No injection at this timestep (alpha is zero)
            return {}

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Prepare G_t tensor (dtype, device, batch expansion)
        # ═══════════════════════════════════════════════════════════════════
        # Ensure G_t has same dtype and device as noise_pred
        G_t = G_t.to(device=noise_pred.device, dtype=noise_pred.dtype)

        # Apply mask if present: G_t = M ⊙ G_t
        if self.mask_tensor is not None:
            mask = self.mask_tensor.to(device=noise_pred.device, dtype=noise_pred.dtype)
            G_t = G_t * mask

        # Expand G_t to match batch size if needed
        # Handle classifier-free guidance: noise_pred may be [2*B, C, H, W]
        # when guidance_scale > 1, or [B, C, H, W] when guidance_scale = 1
        if noise_pred.dim() == 4:
            # Get the FULL batch size of noise_pred (may be 2*B in guidance mode)
            full_batch_size = noise_pred.shape[0]
            noise_pred_channels = noise_pred.shape[1]
            noise_pred_height = noise_pred.shape[2]
            noise_pred_width = noise_pred.shape[3]
            
            # Handle different G_t shapes
            if G_t.dim() == 3:
                # G_t is [C, H, W] - expand to match noise_pred
                G_t = G_t.unsqueeze(0).expand(full_batch_size, -1, -1, -1)
            elif G_t.dim() == 4:
                # G_t already has batch dimension - check if it matches
                if G_t.shape[0] != full_batch_size:
                    # Batch size mismatch - expand or slice as needed
                    if G_t.shape[0] == 1:
                        # Single batch - expand to full batch size
                        G_t = G_t.expand(full_batch_size, -1, -1, -1)
                    elif G_t.shape[0] * 2 == full_batch_size:
                        # G_t is [B, C, H, W] but noise_pred is [2*B, C, H, W] (guidance mode)
                        # Duplicate G_t to match
                        G_t = torch.cat([G_t, G_t], dim=0)
                    else:
                        raise ValueError(
                            f"Cannot match G_t batch size {G_t.shape[0]} to noise_pred batch size {full_batch_size}"
                        )
            
            # Final validation: ensure all dimensions match
            if G_t.shape != noise_pred.shape:
                raise ValueError(
                    f"Shape mismatch after expansion: G_t is {G_t.shape} but noise_pred is {noise_pred.shape}. "
                    f"Expected G_t to be [{full_batch_size}, {noise_pred_channels}, {noise_pred_height}, {noise_pred_width}]"
                )

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Apply bias IN-PLACE to preserve tensor identity
        # CRITICAL: Use .add_() for in-place modification
        # This ensures scheduler receives the SAME tensor object
        # 
        # IMPORTANT: When guidance_scale > 1.0, the pipeline will call chunk(2)
        # on noise_pred after this hook returns. The tensor must remain
        # contiguous and properly structured for chunk() to work correctly.
        # ═══════════════════════════════════════════════════════════════════
        bias = alpha_t * G_t
        
        # Final validation: bias must match noise_pred shape exactly
        if bias.shape != noise_pred.shape:
            raise ValueError(
                f"Bias shape {bias.shape} does not match noise_pred shape {noise_pred.shape}. "
                f"This will cause errors. G_t was expanded incorrectly."
            )
        
        # Ensure bias is contiguous (required for in-place addition)
        if not bias.is_contiguous():
            bias = bias.contiguous()
        
        # Apply bias in-place
        noise_pred.add_(bias)
        
        # Ensure noise_pred remains contiguous after in-place modification
        # This is critical for the pipeline's chunk(2) operation in guidance
        if not noise_pred.is_contiguous():
            # This shouldn't happen with add_, but ensure it anyway
            noise_pred = noise_pred.contiguous()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: Store intermediate latents and deltas if requested
        # ═══════════════════════════════════════════════════════════════════
        if self.store_intermediates:
            # Check if we should store this timestep
            should_store = (
                self.timesteps_to_store is None or step_index in self.timesteps_to_store
            )
            
            if should_store:
                # Store current latent (before scheduler step)
                # Handle guidance case: latents may be duplicated [2*B, C, H, W]
                # Store only the first half (actual batch) to avoid duplication
                if latents.shape[0] > 1 and latents.shape[0] % 2 == 0:
                    # Likely guidance mode - store first half
                    actual_latents = latents[:latents.shape[0] // 2].clone().detach()
                else:
                    actual_latents = latents.clone().detach()
                self.intermediate_latents[step_index] = actual_latents
                
                # Store delta for this step (bias applied to noise_pred)
                # For guidance mode, bias is applied to both halves, so store first half
                if bias.shape[0] > 1 and bias.shape[0] % 2 == 0:
                    # Likely guidance mode - store first half
                    actual_bias = bias[:bias.shape[0] // 2].clone().detach()
                else:
                    actual_bias = bias.clone().detach()
                self.intermediate_deltas[step_index] = actual_bias
                
                # Update cumulative delta (use actual_bias for consistency)
                if self.cumulative_delta is None:
                    self.cumulative_delta = actual_bias.clone().detach()
                else:
                    # Ensure shapes match
                    if self.cumulative_delta.shape != actual_bias.shape:
                        # Reshape cumulative_delta to match if needed
                        if self.cumulative_delta.shape[0] == 1 and actual_bias.shape[0] > 1:
                            self.cumulative_delta = self.cumulative_delta.expand_as(actual_bias)
                    # Note: This is approximate - the actual cumulative effect
                    # depends on how the scheduler processes the biased noise_pred
                    self.cumulative_delta = self.cumulative_delta + actual_bias.clone().detach()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 7: Return modified noise_pred (same tensor reference)
        # ═══════════════════════════════════════════════════════════════════
        return {"noise_pred": noise_pred}
    
    def get_intermediate_latents(self) -> Dict[int, torch.Tensor]:
        """
        Get stored intermediate latents.
        
        Returns:
            Dictionary mapping step_index → latent tensor [B, C, H, W]
        """
        if not self.store_intermediates:
            raise ValueError("store_intermediates was False during initialization")
        return self.intermediate_latents.copy()
    
    def get_intermediate_deltas(self) -> Dict[int, torch.Tensor]:
        """
        Get stored intermediate deltas (bias tensors).
        
        Returns:
            Dictionary mapping step_index → delta tensor [B, C, H, W]
        """
        if not self.store_intermediates:
            raise ValueError("store_intermediates was False during initialization")
        return self.intermediate_deltas.copy()
    
    def get_cumulative_delta(self) -> Optional[torch.Tensor]:
        """
        Get cumulative delta (sum of all bias tensors).
        
        Returns:
            Cumulative delta tensor [B, C, H, W] or None if not stored
        """
        if not self.store_intermediates:
            raise ValueError("store_intermediates was False during initialization")
        return self.cumulative_delta.clone() if self.cumulative_delta is not None else None
    
    def clear_intermediates(self) -> None:
        """Clear stored intermediate latents and deltas."""
        if self.store_intermediates:
            self.intermediate_latents.clear()
            self.intermediate_deltas.clear()
            self.cumulative_delta = None


def apply_bias_to_noise(
    noise_pred: torch.Tensor,
    G_t: torch.Tensor,
    alpha_t: float,
    mask: Optional[torch.Tensor] = None,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Apply additive bias to predicted noise (Strategy 1 utility function).

    Strategy 1 requires IN-PLACE modification to preserve tensor identity.
    The scheduler must receive the SAME tensor object that was modified.

    Formula: eps' = eps + alpha_t * (M ⊙ G_t)

    Args:
        noise_pred: Predicted noise [B, C, H, W]
                    WARNING: Will be modified IN-PLACE if inplace=True
        G_t: Watermark tensor [C, H, W] or [B, C, H, W]
        alpha_t: Bias strength scalar
        mask: Mask tensor [C, H, W] or [B, C, H, W] (optional)
        inplace: If True (default), modify noise_pred in-place via .add_()
                 MUST be True for Strategy 1 to work correctly

    Returns:
        Biased noise prediction (same tensor as noise_pred if inplace=True)
    """
    # Apply mask if present
    if mask is not None:
        G_t = G_t * mask

    # Compute bias
    bias = alpha_t * G_t

    # Apply bias (in-place is CRITICAL for Strategy 1)
    if inplace:
        noise_pred.add_(bias)
        return noise_pred
    else:
        return noise_pred + bias

