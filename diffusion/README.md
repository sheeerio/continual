# When Do Diffusion Transformers Outperform U-Nets?

**CPSC 440 — Advanced Machine Learning (UBC)**  
Gunbir Baveja, Parvin Aliyeva, Arshvir Sandhu

## Overview

This codebase implements a controlled comparison between two denoising architectures
for diffusion models on CIFAR-10:

1. **U-Net** — Standard convolutional denoiser (ResBlocks + attention at low resolution)
2. **DiT** — Diffusion Transformer (patch-based self-attention with adaLN-Zero)

Both share the same diffusion training pipeline, noise schedule, and training objective.
The only thing that changes is the denoiser architecture.

## Project Structure

```
dit-vs-unet/
├── models/
│   ├── unet.py          # Convolutional U-Net denoiser
│   └── dit.py           # Diffusion Transformer denoiser
├── utils/
│   ├── diffusion.py     # Forward/reverse diffusion, DDPM + DDIM sampling
│   ├── blocks.py        # Shared components (sinusoidal embeddings, etc.)
│   └── corruptions.py   # Structured corruption functions for stress-testing
├── config.py            # Dataclass-based configuration system
├── train.py             # Training script
├── evaluate.py          # Corruption evaluation script
└── README.md
```

## Recommended Matched Configs

For a fair comparison, match models by parameter count:

| Config | U-Net | DiT |
|--------|-------|-----|
| **Small (~8-10M)** | base=64, mults=(1,2,2), res_blocks=2 → **8.4M** | dim=256, depth=8, patch=4 → **9.8M** |
| **Medium (~12-14M)** | base=64, mults=(1,2,3), res_blocks=2 → **12.3M** | dim=256, depth=12, patch=4 → **14.5M** |

## Quick Start

### Train U-Net baseline
```bash
python train.py --model unet --epochs 100 --batch_size 128 \
    --base_channels 64 --num_res_blocks 2 \
    --name unet_small --output_dir ./outputs
```

### Train DiT
```bash
python train.py --model dit --epochs 100 --batch_size 128 \
    --hidden_dim 256 --depth 8 --patch_size 4 \
    --name dit_small --output_dir ./outputs
```

### DiT ablation (varying patch size)
```bash
python train.py --model dit --patch_size 2 --name dit_patch2
python train.py --model dit --patch_size 4 --name dit_patch4
python train.py --model dit --patch_size 8 --name dit_patch8
```

### DiT ablation (varying depth)
```bash
python train.py --model dit --depth 4  --name dit_depth4
python train.py --model dit --depth 8  --name dit_depth8
python train.py --model dit --depth 12 --name dit_depth12
```

### Evaluate structured corruptions
```bash
python evaluate.py \
    --unet_ckpt outputs/unet_small/checkpoints/final.pt \
    --dit_ckpt outputs/dit_small/checkpoints/final.pt \
    --output_dir ./eval_results
```

## Architecture Details

### U-Net
- ResNet blocks with GroupNorm + SiLU activation
- Timestep conditioning via Adaptive Group Normalization (scale & shift)
- Multi-head self-attention at configurable spatial resolutions
- Encoder-decoder with skip connections
- Strided conv downsampling, nearest-neighbor + conv upsampling

### DiT (Diffusion Transformer)
- Patchify layer (non-overlapping Conv2d with stride = patch_size)
- Learnable position embeddings
- Transformer blocks with multi-head self-attention
- Timestep conditioning via adaLN-Zero (all gates initialized to zero)
- GELU activation in MLP
- Linear unpatchify projection

### Diffusion Process (shared)
- Linear beta schedule: β₁ = 1e-4 to β_T = 0.02, T = 1000
- Forward: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
- Training objective: E[||ε - f_θ(x_t, t)||²]
- Sampling: DDPM (1000 steps) or DDIM (50 steps, deterministic)

## Structured Corruption Experiments

The novel component: we evaluate both denoisers under conditions
that weaken local cues, testing whether the transformer benefits more
from its global receptive field.

Corruption types:
- **Patch masking**: Random contiguous patches zeroed out (25%, 50%, 75%)
- **Center block masking**: Large centered square removed (25%, 50% of image)
- **Random block masking**: Large randomly-placed square removed
- **Stripe masking**: Alternating horizontal/vertical stripes removed

For each corruption, we measure reconstruction MSE separately in
masked vs. visible regions across multiple noise levels.

## Key Hypothesis

> Transformer-based denoisers are most helpful when reconstruction
> depends on long-range interactions across the image.

If true, we expect:
- Similar or comparable performance in standard denoising
- DiT advantage grows under structured corruptions (especially in masked regions)
- DiT advantage grows with larger/more structured corruption patterns
