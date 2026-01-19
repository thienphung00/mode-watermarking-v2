"""
Utilities for custom DDIM sampling and latent extraction.

Provides functions for:
- Custom DDIM sampling loop with intermediate latent access
- Text conditioning embedding extraction
- Latent extraction from pipeline results
- DDIM inversion for getting x_t from final image
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm


def get_text_embeddings(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract text conditioning embeddings from pipeline.

    Args:
        pipeline: Stable Diffusion pipeline
        prompt: Text prompt
        negative_prompt: Optional negative prompt (for classifier-free guidance)
        device: Device to place embeddings on (default: pipeline.device)

    Returns:
        Tuple of (prompt_embeddings, negative_embeddings or None)
        prompt_embeddings shape: [1, seq_len, hidden_dim]
    """
    if device is None:
        device = pipeline.device

    # Tokenize prompt
    text_inputs = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = text_inputs.to(device)

    # Get text embeddings
    prompt_embeddings = pipeline.text_encoder(text_inputs.input_ids)[0]

    # Get negative embeddings if provided
    negative_embeddings = None
    if negative_prompt is not None:
        negative_inputs = pipeline.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_inputs = negative_inputs.to(device)
        negative_embeddings = pipeline.text_encoder(negative_inputs.input_ids)[0]

    return prompt_embeddings, negative_embeddings


def custom_ddim_sample(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    latents: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    return_intermediates: bool = False,
    timesteps_to_save: Optional[List[int]] = None,
    hook: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Custom DDIM sampling loop with access to intermediate latents.

    This function manually steps through the DDIM sampling process,
    allowing you to capture latents at any timestep.

    Args:
        pipeline: Stable Diffusion pipeline
        prompt: Text prompt
        num_inference_steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        latents: Initial latents [B, C, H, W] (optional)
        seed: Random seed for initial latents (optional)
        negative_prompt: Negative prompt for guidance (optional)
        return_intermediates: If True, return intermediate latents
        timesteps_to_save: List of step indices to save (None = save all)
        hook: Optional watermark hook to apply (must be compatible with custom loop)

    Returns:
        Dictionary containing:
            - latents: Final latent tensor [B, C, H, W]
            - images: Decoded PIL images
            - intermediate_latents: Dict[step_index, latent] if return_intermediates=True
            - all_latents: List of all latents if return_intermediates=True
    """
    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Set up generator
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Get text embeddings
    prompt_embeddings, negative_embeddings = get_text_embeddings(
        pipeline, prompt, negative_prompt, device
    )

    # Prepare latents
    if latents is None:
        height = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        latent_height = height // pipeline.vae_scale_factor
        latent_width = width // pipeline.vae_scale_factor
        latent_shape = (1, pipeline.unet.config.in_channels, latent_height, latent_width)
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        latents = latents.to(device=device, dtype=dtype)

    # Set up scheduler
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # Storage for intermediate latents
    intermediate_latents: Dict[int, torch.Tensor] = {}
    all_latents: List[torch.Tensor] = []

    # Prepare embeddings for classifier-free guidance
    if guidance_scale > 1.0 and negative_embeddings is not None:
        # Concatenate negative and positive embeddings
        text_embeddings = torch.cat([negative_embeddings, prompt_embeddings])
    else:
        text_embeddings = prompt_embeddings

    # Denoising loop
    for i, t in enumerate(tqdm(timesteps, desc="DDIM sampling")):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        # Predict noise
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        # Apply guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # Apply hook if provided (for watermark injection)
        # Note: Hook expects step_index, timestep, latents, and noise_pred in kwargs
        if hook is not None:
            # Convert timestep to int for hook
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            hook_result = hook(i, t_int, latents, noise_pred=noise_pred)
            if hook_result and isinstance(hook_result, dict) and "noise_pred" in hook_result:
                noise_pred = hook_result["noise_pred"]

        # Scheduler step
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        # Save intermediate latent AFTER scheduler step (this is x_{t-1})
        if return_intermediates:
            if timesteps_to_save is None or i in timesteps_to_save:
                intermediate_latents[i] = latents.clone().detach()
            all_latents.append(latents.clone().detach())

    # Decode final latents
    latents = 1 / pipeline.vae.config.scaling_factor * latents
    with torch.no_grad():
        images = pipeline.vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")

    # Convert to PIL
    from PIL import Image
    pil_images = [Image.fromarray(img) for img in images]

    result = {
        "latents": latents,
        "images": pil_images,
    }

    if return_intermediates:
        result["intermediate_latents"] = intermediate_latents
        result["all_latents"] = all_latents

    return result


def extract_latents_from_pipeline_result(
    result: Any,
    return_dict: bool = True,
) -> Optional[torch.Tensor]:
    """
    Extract latents from pipeline result.

    Args:
        result: Result from pipeline() call
        return_dict: Whether result is a dict or object

    Returns:
        Latent tensor [B, C, H, W] or None if not available
    """
    if return_dict:
        if isinstance(result, dict):
            return result.get("latents")
        elif hasattr(result, "latents"):
            return result.latents
    else:
        if hasattr(result, "latents"):
            return result.latents

    return None


def ddim_invert(
    pipeline: StableDiffusionPipeline,
    image: Any,  # PIL Image or torch.Tensor
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    num_inversion_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Perform DDIM inversion to get x_t from final image.

    Inverts the diffusion process to recover the latent representation
    at timestep t from a final image.

    Args:
        pipeline: Stable Diffusion pipeline
        image: PIL Image or tensor [1, 3, H, W] in range [0, 1]
        prompt: Text prompt (used for conditioning, but inversion is prompt-agnostic)
        num_inference_steps: Number of steps for inversion (default: same as generation)
        guidance_scale: Guidance scale (typically 1.0 for inversion)
        num_inversion_steps: Number of inversion steps (default: num_inference_steps)

    Returns:
        Inverted latent tensor [1, C, H, W] at timestep t=num_inference_steps-1
    """
    from PIL import Image
    import numpy as np

    device = pipeline.device
    dtype = pipeline.unet.dtype

    # Convert image to tensor if needed
    if isinstance(image, Image.Image):
        # Preprocess image
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.max() > 1.0:
            image = (image / 255.0 - 0.5) / 0.5  # Normalize if in [0, 255]
        image = image.to(device)

    # Encode image to latent space
    with torch.no_grad():
        latents = pipeline.vae.encode(image).latent_dist.sample()
        latents = latents * pipeline.vae.config.scaling_factor

    # Get text embeddings (needed for UNet forward, but inversion is prompt-agnostic)
    prompt_embeddings, _ = get_text_embeddings(pipeline, prompt, device=device)

    # Set up scheduler for inversion
    if num_inversion_steps is None:
        num_inversion_steps = num_inference_steps

    pipeline.scheduler.set_timesteps(num_inversion_steps, device=device)
    timesteps = reversed(pipeline.scheduler.timesteps)

    # Inversion loop (reverse denoising)
    for t in tqdm(timesteps, desc="DDIM inversion"):
        # Predict noise
        latent_model_input = pipeline.scheduler.scale_model_input(latents, t)
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeddings,
        ).sample

        # Reverse scheduler step
        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

    return latents

