"""
Diffusion Transformer (DiT) Denoiser.

A vision-transformer-based denoiser following the DiT paper (Peebles & Xie, 2023):
- Patchify input images into token sequences
- Transformer blocks with multi-head self-attention
- Timestep conditioning via adaptive layer norm zero (adaLN-Zero)
- Final linear projection to unpatchify

All hyperparameters (patch size, depth, width, heads) are configurable
for the ablation study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from utils.blocks import SinusoidalPositionEmbedding


# ---------------------------------------------------------------------------
# Patchification
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Split image into non-overlapping patches and linearly embed them.
    
    For a 32x32 image with patch_size=4:
        - num_patches = (32/4)^2 = 64 patches
        - each patch is 4x4x3 = 48 pixels, projected to hidden_dim
    """
    
    def __init__(self, image_size: int = 32, patch_size: int = 4, in_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size
        
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  ->  (B, num_patches, hidden_dim)
        """
        x = self.proj(x)                           # (B, hidden_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)           # (B, num_patches, hidden_dim)
        return x


class UnpatchEmbed(nn.Module):
    """
    Reverse of PatchEmbed: project tokens back to image patches and reshape.
    """
    
    def __init__(self, image_size: int = 32, patch_size: int = 4, out_channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        
        self.proj = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_patches, hidden_dim)  ->  (B, C, H, W)
        """
        B, N, D = x.shape
        x = self.proj(x)  # (B, N, patch_size^2 * C)
        
        # Reshape to image
        p = self.patch_size
        c = self.out_channels
        g = self.grid_size
        
        x = x.reshape(B, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)           # (B, C, g, p, g, p)
        x = x.reshape(B, c, g * p, g * p)          # (B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# adaLN-Zero: Adaptive Layer Norm with zero initialization
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (from DiT paper).
    
    Given conditioning vector c, predicts:
        gamma1, beta1  (for pre-attention norm)
        gamma2, beta2  (for pre-MLP norm)  
        alpha1         (gate for attention output)
        alpha2         (gate for MLP output)
    
    All initialized to zero so the block starts as identity.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # Zero-initialize so each block starts as identity
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)
    
    def forward(self, c: torch.Tensor):
        """
        c: (B, cond_dim) -> 6 tensors each (B, hidden_dim)
        """
        params = self.proj(c)
        return params.chunk(6, dim=-1)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    A single DiT transformer block:
        x -> LayerNorm (modulated by adaLN) -> MultiHead SelfAttn -> gate -> residual
        x -> LayerNorm (modulated by adaLN) -> MLP -> gate -> residual
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # adaLN-Zero conditioning
        self.adaln = AdaLNZero(hidden_dim, cond_dim)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) — token sequence
        c: (B, cond_dim) — conditioning (timestep embedding)
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(c)
        
        # --- Attention branch ---
        h = self.norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * h
        
        # --- MLP branch ---
        h = self.norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.mlp(h)
        x = x + alpha2.unsqueeze(1) * h
        
        return x


# ---------------------------------------------------------------------------
# Full DiT Model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """
    Diffusion Transformer denoiser.
    
    Default config for CIFAR-10 aiming at ~8M params:
        hidden_dim=256, depth=8, num_heads=4, patch_size=4
    
    Args:
        image_size: Input image spatial size (e.g. 32 for CIFAR-10)
        patch_size: Size of each image patch (e.g. 4 -> 8x8 grid of patches)
        in_channels: Number of image channels
        hidden_dim: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_dim * mlp_ratio
        time_dim: Dimension of timestep embedding
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Patch embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, time_dim, dropout)
            for _ in range(depth)
        ])
        
        # Final layer: norm + linear projection to pixel patches
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)
        
        self.unpatch = UnpatchEmbed(image_size, patch_size, in_channels, hidden_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy images, (B, C, H, W)
            t: Timestep indices, (B,)
        Returns:
            Predicted noise, (B, C, H, W)
        """
        # Embed timestep
        c = self.time_embed(t)  # (B, time_dim)
        
        # Patchify + position embedding
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, c)
        
        # Final layer with adaLN
        shift_scale = self.final_adaln(c)
        shift, scale = shift_scale.chunk(2, dim=-1)
        tokens = self.final_norm(tokens)
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Unpatchify to image
        return self.unpatch(tokens)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
