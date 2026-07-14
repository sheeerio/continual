"""
U-Net Denoiser for Diffusion Models.

A standard convolutional U-Net following DDPM's design:
- ResNet blocks with GroupNorm + SiLU
- Timestep conditioning via adaptive group normalization
- Self-attention at low resolutions
- Skip connections between encoder and decoder

All hyperparameters are configurable for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from utils.blocks import TimestepMLP


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class AdaptiveGroupNorm(nn.Module):
    """
    Group normalization with scale and shift predicted from timestep embedding.
    This is how the U-Net injects time information.
    """
    
    def __init__(self, num_channels: int, num_groups: int = 32, time_dim: int = 256):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, num_channels), num_channels)
        self.proj = nn.Linear(time_dim, num_channels * 2)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t_emb: (B, time_dim)
        """
        x = self.norm(x)
        scale_shift = self.proj(t_emb)[:, :, None, None]  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """
    Residual block with two convolutions and adaptive group norm for time conditioning.
    
        x -> GroupNorm -> SiLU -> Conv -> [+ time] -> GroupNorm -> SiLU -> Dropout -> Conv -> [+ residual]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int = 256,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_channels, num_groups, time_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = AdaptiveGroupNorm(out_channels, num_groups, time_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        self.act = nn.SiLU()
        
        # Residual projection if channels change
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        h = self.act(self.norm1(x, t_emb))
        h = self.conv1(h)
        
        h = self.act(self.norm2(h, t_emb))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class SelfAttention2D(nn.Module):
    """
    Multi-head self-attention over spatial dimensions.
    Applied at low resolutions in the U-Net (e.g., 8x8, 4x4).
    """
    
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each: (B, heads, d, HW)
        
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, d)
        k = k                        # (B, heads, d, HW)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, d)
        
        attn = torch.matmul(q, k) * self.scale  # (B, heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, HW, d)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)  # (B, C, H, W)
        out = self.proj_out(out)
        
        return out + residual


class Downsample(nn.Module):
    """Spatial downsampling with a strided convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling with nearest-neighbor interpolation + convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net denoiser for diffusion models.
    
    Args:
        in_channels: Number of image channels (3 for RGB)
        base_channels: Base channel width (doubled at each level)
        channel_mults: Multipliers for each resolution level, e.g. (1, 2, 4)
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: At which downsampling factors to apply attention
            e.g., (2, 4) means attention at 16x16 and 8x8 for 32x32 input
        time_dim: Dimension of timestep embedding
        dropout: Dropout rate
        num_heads: Number of attention heads
        num_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (2,),   # attention at base_res // 2
        time_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.time_dim = time_dim
        
        # Timestep embedding
        self.time_embed = TimestepMLP(time_dim, time_dim)
        
        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # ------ Encoder ------
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = base_channels
        encoder_channels = [ch]  # track channels for skip connections
        current_ds = 1  # current downsampling factor
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_dim, dropout, num_groups)]
                if current_ds in attention_resolutions:
                    layers.append(SelfAttention2D(out_ch, num_heads, num_groups))
                self.encoder_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                encoder_channels.append(ch)
            
            # Downsample (except last level)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(ch))
                encoder_channels.append(ch)
                current_ds *= 2
        
        # ------ Bottleneck ------
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        self.mid_attn = SelfAttention2D(ch, num_heads, num_groups)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        
        # ------ Decoder ------
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in reversed(range(len(channel_mults))):
            mult = channel_mults[level]
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):  # +1 for the extra block after concat
                skip_ch = encoder_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_dim, dropout, num_groups)]
                
                ds_at_level = 2 ** level
                if ds_at_level in attention_resolutions:
                    layers.append(SelfAttention2D(out_ch, num_heads, num_groups))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
                ch = out_ch
            
            # Upsample (except first decoder level = last encoder level)
            if level > 0:
                self.upsamplers.append(Upsample(ch))
        
        # ------ Output ------
        self.output_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy images, (B, C, H, W)
            t: Timestep indices, (B,)
        Returns:
            Predicted noise, (B, C, H, W)
        """
        t_emb = self.time_embed(t)
        
        # Initial conv
        h = self.input_conv(x)
        skips = [h]
        
        # Encoder
        ds_idx = 0
        block_idx = 0
        for level, mult in enumerate(self.channel_mults_from_blocks()):
            for _ in range(self._num_res_blocks()):
                for layer in self.encoder_blocks[block_idx]:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                skips.append(h)
                block_idx += 1
            
            if ds_idx < len(self.downsamplers):
                h = self.downsamplers[ds_idx](h)
                skips.append(h)
                ds_idx += 1
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        up_idx = 0
        block_idx = 0
        for block_layers in self.decoder_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for layer in block_layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            block_idx += 1
            
            # Check if we need to upsample
            # Upsample after every (num_res_blocks+1) blocks in each level
            if up_idx < len(self.upsamplers) and block_idx % (self._num_res_blocks() + 1) == 0:
                h = self.upsamplers[up_idx](h)
                up_idx += 1
        
        # Output
        h = F.silu(self.output_norm(h))
        return self.output_conv(h)
    
    def channel_mults_from_blocks(self):
        """Helper — reconstruct channel_mults from init."""
        # We store it to avoid re-parsing
        return self._channel_mults
    
    def _num_res_blocks(self):
        return self._num_res_blocks_val


# ---------------------------------------------------------------------------
# Cleaner forward pass — let's rewrite UNet with a simpler approach
# ---------------------------------------------------------------------------

class UNetDenoiser(nn.Module):
    """
    Clean implementation of the U-Net denoiser.
    
    Default config for CIFAR-10 at ~8M params:
        base_channels=64, channel_mults=(1,2,4), num_res_blocks=2
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (16,),  # apply attention at these spatial sizes
        time_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_groups: int = 32,
        image_size: int = 32,
    ):
        super().__init__()
        self.image_size = image_size
        
        # Timestep embedding
        self.time_embed = TimestepMLP(time_dim, time_dim)
        
        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Build encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.ModuleList()
        
        ch = base_channels
        self.skip_channels = [ch]
        current_res = image_size
        
        for level_idx, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout, num_groups)])
                if current_res in attn_resolutions:
                    block.append(SelfAttention2D(out_ch, num_heads, num_groups))
                level_blocks.append(block)
                ch = out_ch
                self.skip_channels.append(ch)
            
            self.encoder.append(level_blocks)
            
            if level_idx < len(channel_mults) - 1:
                self.pool.append(Downsample(ch))
                self.skip_channels.append(ch)
                current_res //= 2
        
        # Bottleneck
        self.mid1 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        self.mid_attn = SelfAttention2D(ch, num_heads, num_groups)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        
        # Build decoder (mirrors encoder)
        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for level_idx in reversed(range(len(channel_mults))):
            mult = channel_mults[level_idx]
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            
            for i in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                block = nn.ModuleList([
                    ResBlock(ch + skip_ch, out_ch, time_dim, dropout, num_groups)
                ])
                
                # Compute resolution at this level
                level_res = image_size // (2 ** level_idx) if level_idx < len(channel_mults) - 1 else current_res
                # Simpler: just check if the encoder had attention at this level
                enc_res = image_size // (2 ** level_idx)
                if enc_res in attn_resolutions:
                    block.append(SelfAttention2D(out_ch, num_heads, num_groups))
                
                level_blocks.append(block)
                ch = out_ch
            
            self.decoder.append(level_blocks)
            if level_idx > 0:
                self.up.append(Upsample(ch))
        
        # Output
        self.out_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        
        h = self.input_conv(x)
        skips = [h]
        
        # Encoder
        for level_idx, level_blocks in enumerate(self.encoder):
            for block in level_blocks:
                for layer in block:
                    h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
                skips.append(h)
            if level_idx < len(self.pool):
                h = self.pool[level_idx](h)
                skips.append(h)
        
        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        
        # Decoder
        for level_idx, level_blocks in enumerate(self.decoder):
            for block in level_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                for layer in block:
                    h = layer(h, t_emb) if isinstance(layer, ResBlock) else layer(h)
            if level_idx < len(self.up):
                h = self.up[level_idx](h)
        
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
