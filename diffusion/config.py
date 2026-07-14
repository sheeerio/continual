"""
Configuration system for experiments.

Everything is a dataclass so it's easy to modify for ablations.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple
import json


@dataclass
class DiffusionConfig:
    """Diffusion process settings."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class UNetConfig:
    """U-Net architecture settings."""
    in_channels: int = 3
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)  # spatial sizes where attention is applied
    time_dim: int = 256
    dropout: float = 0.1
    num_heads: int = 4
    num_groups: int = 32
    image_size: int = 32


@dataclass
class DiTConfig:
    """DiT architecture settings."""
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    hidden_dim: int = 256
    depth: int = 8
    num_heads: int = 4
    mlp_ratio: float = 4.0
    time_dim: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training settings."""
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_every: int = 100         # log every N steps
    sample_every: int = 5000     # generate samples every N steps
    save_every: int = 10000      # save checkpoint every N steps
    num_sample_images: int = 64  # how many images to generate at each sample step
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation / corruption experiment settings."""
    # Timesteps at which to evaluate corruption recovery
    eval_timesteps: Tuple[int, ...] = (100, 250, 500, 750)
    
    # Patch masking
    patch_sizes: Tuple[int, ...] = (4, 8)
    mask_ratios: Tuple[float, ...] = (0.25, 0.5, 0.75)
    
    # Block masking
    block_fractions: Tuple[float, ...] = (0.25, 0.5)
    
    # Number of batches to average over
    num_eval_batches: int = 10
    eval_batch_size: int = 64


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""
    name: str = "dit_vs_unet"
    model_type: str = "unet"     # "unet" or "dit"
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./outputs"
    device: str = "cuda"
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path) as f:
            d = json.load(f)
        return cls(
            name=d['name'],
            model_type=d['model_type'],
            diffusion=DiffusionConfig(**d['diffusion']),
            unet=UNetConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d['unet'].items()}),
            dit=DiTConfig(**d['dit']),
            train=TrainConfig(**d['train']),
            eval=EvalConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d['eval'].items()}),
            output_dir=d['output_dir'],
            device=d['device'],
        )
