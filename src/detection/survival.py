"""
Watermark survival analysis through UNet denoising steps.

Measures how much of an injected watermark perturbation survives through
a single UNet denoising step using Jacobian-vector product approximation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def compute_watermark_survival(
    unet: torch.nn.Module,
    x_t: torch.Tensor,
    x_t_pert: torch.Tensor,
    delta: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Measure how much of a watermark bias survives a single UNet denoising step.

    Uses first-order finite difference approximation to compute the Jacobian-vector
    product without computing the full Jacobian matrix:

        J_t * delta ≈ UNet(x_t + delta) - UNet(x_t)

    The survival factor quantifies signal preservation:

        S_t = ||J_t * delta|| / ||delta||

    Args:
        unet: The Stable Diffusion UNet (torch nn.Module)
        x_t: Original latent at timestep t, shape [B, 4, H, W]
        x_t_pert: Perturbed latent x_t' = x_t + delta, shape [B, 4, H, W]
        delta: Injected perturbation Δx_t = x_t_pert - x_t, shape [B, 4, H, W]
        t: Diffusion timestep tensor, shape [B]
        encoder_hidden_states: Text conditioning embeddings (optional),
                               shape [B, seq_len, hidden_dim]
        device: Torch device for computation

    Returns:
        Tuple containing:
            - survival_factor (float): Survival ratio ||J_t * delta|| / ||delta||
            - jvp (torch.Tensor): Jacobian-vector product, shape [B, 4, H, W]
            - eps_clean (torch.Tensor): UNet(x_t) prediction, shape [B, 4, H, W]
            - eps_pert (torch.Tensor): UNet(x_t_pert) prediction, shape [B, 4, H, W]

    Example:
        >>> delta = alpha_t * (M * G_t)
        >>> x_t_pert = x_t + delta
        >>> survival = compute_watermark_survival(
        ...     unet, x_t, x_t_pert, delta, t, encoder_hidden_states=embeddings
        ... )
        >>> survival_factor, jvp, eps_clean, eps_pert = survival
        >>> print(f"Survival factor: {survival_factor:.4f}")
    """
    # ═══════════════════════════════════════════════════════════════════
    # Input validation
    # ═══════════════════════════════════════════════════════════════════
    if x_t.shape != x_t_pert.shape:
        raise ValueError(
            f"x_t shape {x_t.shape} does not match x_t_pert shape {x_t_pert.shape}"
        )

    if x_t.shape != delta.shape:
        raise ValueError(
            f"x_t shape {x_t.shape} does not match delta shape {delta.shape}"
        )

    if x_t.shape[0] != t.shape[0]:
        raise ValueError(
            f"Batch size mismatch: x_t batch {x_t.shape[0]} != t batch {t.shape[0]}"
        )

    if x_t.shape[1] != 4:
        raise ValueError(
            f"Expected 4 channels in latent, got {x_t.shape[1]}"
        )

    # Verify x_t_pert ≈ x_t + delta (with tolerance for floating point)
    expected_pert = x_t + delta
    if not torch.allclose(x_t_pert, expected_pert, atol=1e-5, rtol=1e-5):
        import warnings
        warnings.warn(
            "x_t_pert does not equal x_t + delta. Using provided x_t_pert as-is.",
            UserWarning
        )

    # Ensure tensors are on correct device
    x_t = x_t.to(device)
    x_t_pert = x_t_pert.to(device)
    delta = delta.to(device)
    t = t.to(device)

    if encoder_hidden_states is not None:
        encoder_hidden_states = encoder_hidden_states.to(device)

    # Ensure UNet is on correct device and in eval mode
    unet = unet.to(device)
    unet.eval()

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Compute UNet(x_t) and UNet(x_t_pert)
    # ═══════════════════════════════════════════════════════════════════
    # Forward pass on clean latent
    eps_clean = unet(
        x_t,
        t,
        encoder_hidden_states=encoder_hidden_states,
    )

    # Handle tuple return (some diffusers versions return tuple)
    if isinstance(eps_clean, tuple):
        eps_clean = eps_clean[0]

    # Forward pass on perturbed latent
    eps_pert = unet(
        x_t_pert,
        t,
        encoder_hidden_states=encoder_hidden_states,
    )

    # Handle tuple return
    if isinstance(eps_pert, tuple):
        eps_pert = eps_pert[0]

    # Validate output shapes
    if eps_clean.shape != x_t.shape:
        raise ValueError(
            f"UNet output shape {eps_clean.shape} does not match input shape {x_t.shape}"
        )

    if eps_pert.shape != x_t_pert.shape:
        raise ValueError(
            f"UNet output shape {eps_pert.shape} does not match input shape {x_t_pert.shape}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Approximate Jacobian-vector product via finite differences
    # J_t * delta ≈ UNet(x_t + delta) - UNet(x_t) = eps_pert - eps_clean
    # ═══════════════════════════════════════════════════════════════════
    jvp = eps_pert - eps_clean

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Compute survival factor S_t = ||J_t * delta|| / ||delta||
    # ═══════════════════════════════════════════════════════════════════
    # Compute norms (Frobenius norm for tensors)
    jvp_norm = torch.norm(jvp)
    delta_norm = torch.norm(delta)

    # Avoid division by zero
    if delta_norm < 1e-10:
        survival_factor = 0.0
    else:
        survival_factor = float(jvp_norm / delta_norm)

    return survival_factor, jvp, eps_clean, eps_pert

