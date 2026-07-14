"""
Diffusion process utilities.

Implements the forward noising process and reverse sampling for DDPM/DDIM.
Both architectures share this code — the only thing that changes is the denoiser f_theta.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DiffusionSchedule:
    """
    Linear beta schedule for discrete-time diffusion.
    
    Precomputes all the alpha-bar values and related quantities
    so we don't recompute them every training step.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule (same as DDPM)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bar[:-1]])
        
        # Store everything we need as float32 tensors
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        self.alpha_bar = alpha_bar.float().to(device)
        self.alpha_bar_prev = alpha_bar_prev.float().to(device)
        
        # Precompute useful quantities
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float().to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float().to(device)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas).float().to(device)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        ).float().to(device)
    
    def to(self, device):
        """Move all tensors to a new device."""
        self.device = device
        for attr in [
            'betas', 'alphas', 'alpha_bar', 'alpha_bar_prev',
            'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar', 'sqrt_recip_alpha',
            'posterior_variance', 'posterior_mean_coef1', 'posterior_mean_coef2',
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward diffusion process: sample x_t given x_0 and timestep t.
    
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    Args:
        x_0: Clean images, shape (B, C, H, W)
        t: Timestep indices, shape (B,)
        schedule: Diffusion schedule with precomputed values
        noise: Optional pre-sampled noise (for reproducibility)
    
    Returns:
        (x_t, noise): The noised images and the noise that was added
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    sqrt_ab = schedule.sqrt_alpha_bar[t][:, None, None, None]        # (B, 1, 1, 1)
    sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
    
    x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
    return x_t, noise


def compute_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """
    Compute the simple denoising loss: E[||epsilon - f_theta(x_t, t)||^2].
    
    This is the shared training objective for both U-Net and DiT.
    """
    x_t, noise = q_sample(x_0, t, schedule)
    noise_pred = model(x_t, t)
    loss = nn.functional.mse_loss(noise_pred, noise)
    return loss


@torch.no_grad()
def p_sample_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: int,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """
    Single reverse diffusion step: sample x_{t-1} from x_t.
    
    Uses the DDPM posterior:
        mu = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps_pred)
        x_{t-1} = mu + sqrt(posterior_variance) * z
    """
    B = x_t.shape[0]
    t_tensor = torch.full((B,), t, device=x_t.device, dtype=torch.long)
    
    eps_pred = model(x_t, t_tensor)
    
    # Compute predicted mean
    beta_t = schedule.betas[t]
    sqrt_recip_alpha_t = schedule.sqrt_recip_alpha[t]
    sqrt_one_minus_ab_t = schedule.sqrt_one_minus_alpha_bar[t]
    
    mu = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_ab_t) * eps_pred)
    
    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(schedule.posterior_variance[t])
        x_prev = mu + sigma * noise
    else:
        x_prev = mu
    
    return x_prev


@torch.no_grad()
def sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: Tuple[int, ...],
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """
    Generate samples by running the full reverse diffusion process.
    
    Args:
        model: Trained denoiser
        schedule: Diffusion schedule
        shape: Shape of samples to generate, e.g. (batch, 3, 32, 32)
        device: Device to run on
        verbose: Print progress
    
    Returns:
        Generated samples in [-1, 1] range
    """
    model.eval()
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(schedule.num_timesteps)):
        x = p_sample_step(model, x, t, schedule)
        if verbose and t % 100 == 0:
            print(f"  Sampling step {schedule.num_timesteps - t}/{schedule.num_timesteps}")
    
    # Clamp to valid range
    x = torch.clamp(x, -1.0, 1.0)
    return x


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: Tuple[int, ...],
    num_steps: int = 50,
    eta: float = 0.0,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """
    DDIM sampling for faster generation.
    
    Args:
        num_steps: Number of DDIM steps (fewer = faster, slightly lower quality)
        eta: Controls stochasticity (0 = deterministic, 1 = DDPM-like)
    """
    model.eval()
    
    # Create sub-sequence of timesteps
    step_size = schedule.num_timesteps // num_steps
    timesteps = list(range(0, schedule.num_timesteps, step_size))
    timesteps = list(reversed(timesteps))
    
    x = torch.randn(shape, device=device)
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_tensor)
        
        alpha_bar_t = schedule.alpha_bar[t]
        alpha_bar_prev = schedule.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
        
        # Predicted x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Direction pointing to x_t
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
        
        if verbose and i % 10 == 0:
            print(f"  DDIM step {i+1}/{len(timesteps)}")
    
    return torch.clamp(x, -1.0, 1.0)
