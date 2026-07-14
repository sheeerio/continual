"""
Shared building blocks used by both U-Net and DiT.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding, following the original transformer paper.
    Maps integer timesteps to dense vectors.
    
    This is shared by both architectures — the only difference is how
    the embedding gets injected into the network.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Integer timesteps, shape (B,)
        Returns:
            Embeddings, shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        
        return emb


class TimestepMLP(nn.Module):
    """
    Projects sinusoidal timestep embedding through a small MLP.
    Used by both architectures.
    """
    
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))
