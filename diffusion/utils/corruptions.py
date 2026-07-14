"""
Structured corruption utilities for stress-testing denoisers.

These functions create evaluation conditions where local cues are degraded,
forcing the denoiser to rely on long-range spatial interactions.
This is the novel experimental component of the project.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import random


def mask_patches(
    x: torch.Tensor,
    patch_size: int = 8,
    mask_ratio: float = 0.5,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask out random contiguous patches from the image.
    
    Args:
        x: Images, (B, C, H, W)
        patch_size: Size of square patches to mask
        mask_ratio: Fraction of patches to mask out
        mask_value: Value to fill masked regions (0.0 = gray in [-1,1] range)
    
    Returns:
        (masked_x, mask): The corrupted image and binary mask (1=visible, 0=masked)
    """
    B, C, H, W = x.shape
    gh = H // patch_size  # grid height
    gw = W // patch_size  # grid width
    num_patches = gh * gw
    num_masked = int(num_patches * mask_ratio)
    
    # Create per-image random masks
    mask = torch.ones(B, 1, gh, gw, device=x.device)
    for b in range(B):
        indices = torch.randperm(num_patches, device=x.device)[:num_masked]
        rows = indices // gw
        cols = indices % gw
        mask[b, 0, rows, cols] = 0.0
    
    # Upsample mask to image resolution
    mask_full = F.interpolate(mask, size=(H, W), mode='nearest')  # (B, 1, H, W)
    
    masked_x = x * mask_full + mask_value * (1 - mask_full)
    return masked_x, mask_full


def mask_center_block(
    x: torch.Tensor,
    block_fraction: float = 0.5,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask out a centered square block from each image.
    
    Args:
        x: Images, (B, C, H, W)
        block_fraction: Size of block as fraction of image size
        mask_value: Fill value
    
    Returns:
        (masked_x, mask)
    """
    B, C, H, W = x.shape
    block_h = int(H * block_fraction)
    block_w = int(W * block_fraction)
    
    start_h = (H - block_h) // 2
    start_w = (W - block_w) // 2
    
    mask = torch.ones(B, 1, H, W, device=x.device)
    mask[:, :, start_h:start_h + block_h, start_w:start_w + block_w] = 0.0
    
    masked_x = x * mask + mask_value * (1 - mask)
    return masked_x, mask


def mask_random_block(
    x: torch.Tensor,
    block_fraction: float = 0.5,
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask out a randomly positioned square block from each image.
    """
    B, C, H, W = x.shape
    block_h = int(H * block_fraction)
    block_w = int(W * block_fraction)
    
    mask = torch.ones(B, 1, H, W, device=x.device)
    for b in range(B):
        sh = random.randint(0, H - block_h)
        sw = random.randint(0, W - block_w)
        mask[b, :, sh:sh + block_h, sw:sw + block_w] = 0.0
    
    masked_x = x * mask + mask_value * (1 - mask)
    return masked_x, mask


def mask_stripes(
    x: torch.Tensor,
    num_stripes: int = 4,
    orientation: str = "horizontal",
    mask_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask alternating horizontal or vertical stripes.
    This creates a structured corruption where every other stripe is missing,
    requiring the model to interpolate across large spatial gaps.
    """
    B, C, H, W = x.shape
    mask = torch.ones(B, 1, H, W, device=x.device)
    
    if orientation == "horizontal":
        stripe_h = H // (num_stripes * 2)
        for i in range(num_stripes):
            start = (2 * i + 1) * stripe_h
            end = min(start + stripe_h, H)
            mask[:, :, start:end, :] = 0.0
    else:
        stripe_w = W // (num_stripes * 2)
        for i in range(num_stripes):
            start = (2 * i + 1) * stripe_w
            end = min(start + stripe_w, W)
            mask[:, :, :, start:end] = 0.0
    
    masked_x = x * mask + mask_value * (1 - mask)
    return masked_x, mask


def evaluate_reconstruction(
    model,
    x_0: torch.Tensor,
    schedule,
    corruption_fn,
    corruption_kwargs: dict,
    t_eval: int = 500,
) -> dict:
    """
    Evaluate denoiser recovery under structured corruption.
    
    Process:
    1. Apply structured corruption to clean image
    2. Add diffusion noise at timestep t_eval
    3. Run single-step denoising
    4. Measure reconstruction quality in masked vs unmasked regions
    
    Returns:
        Dictionary with MSE metrics (overall, masked region, visible region)
    """
    device = x_0.device
    B = x_0.shape[0]
    
    # Apply corruption
    corrupted, mask = corruption_fn(x_0, **corruption_kwargs)
    
    # Add diffusion noise
    t = torch.full((B,), t_eval, device=device, dtype=torch.long)
    sqrt_ab = schedule.sqrt_alpha_bar[t][:, None, None, None]
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
    
    noise = torch.randn_like(corrupted)
    x_t = sqrt_ab * corrupted + sqrt_one_minus_ab * noise
    
    # Predict noise
    model.eval()
    with torch.no_grad():
        noise_pred = model(x_t, t)
    
    # Reconstruct x_0 estimate
    x0_pred = (x_t - sqrt_one_minus_ab * noise_pred) / sqrt_ab
    
    # Compute MSE in different regions
    mse_overall = F.mse_loss(x0_pred, x_0).item()
    
    # MSE in masked (missing) regions
    inv_mask = 1.0 - mask
    if inv_mask.sum() > 0:
        mse_masked = (((x0_pred - x_0) ** 2) * inv_mask).sum() / (inv_mask.sum() * x_0.shape[1])
        mse_masked = mse_masked.item()
    else:
        mse_masked = 0.0
    
    # MSE in visible regions
    if mask.sum() > 0:
        mse_visible = (((x0_pred - x_0) ** 2) * mask).sum() / (mask.sum() * x_0.shape[1])
        mse_visible = mse_visible.item()
    else:
        mse_visible = 0.0
    
    return {
        "mse_overall": mse_overall,
        "mse_masked_region": mse_masked,
        "mse_visible_region": mse_visible,
    }
