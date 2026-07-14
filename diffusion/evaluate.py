"""
Evaluation script for structured corruption experiments.

Loads trained models and evaluates denoising quality under various
structured corruptions (patch masking, block masking, stripes).

Usage:
    python evaluate.py --unet_ckpt outputs/unet_cifar10/checkpoints/final.pt \
                       --dit_ckpt outputs/dit_cifar10/checkpoints/final.pt
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import ExperimentConfig
from models.unet import UNetDenoiser
from models.dit import DiT
from utils.diffusion import DiffusionSchedule
from utils.corruptions import (
    mask_patches, mask_center_block, mask_random_block,
    mask_stripes, evaluate_reconstruction,
)


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load model + config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.json")
    
    if os.path.exists(config_path):
        config = ExperimentConfig.load(config_path)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Build model
    if config.model_type == "unet":
        c = config.unet
        model = UNetDenoiser(
            in_channels=c.in_channels, base_channels=c.base_channels,
            channel_mults=c.channel_mults, num_res_blocks=c.num_res_blocks,
            attn_resolutions=c.attn_resolutions, time_dim=c.time_dim,
            dropout=c.dropout, num_heads=c.num_heads,
            num_groups=c.num_groups, image_size=c.image_size,
        )
    else:
        c = config.dit
        model = DiT(
            image_size=c.image_size, patch_size=c.patch_size,
            in_channels=c.in_channels, hidden_dim=c.hidden_dim,
            depth=c.depth, num_heads=c.num_heads,
            mlp_ratio=c.mlp_ratio, time_dim=c.time_dim, dropout=c.dropout,
        )
    
    # Load EMA weights (preferred) or model weights
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    
    model.to(device).eval()
    return model, config


def get_test_loader(batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def run_corruption_eval(
    model: nn.Module,
    model_name: str,
    schedule: DiffusionSchedule,
    test_loader: DataLoader,
    device: str,
    num_batches: int = 10,
    output_dir: str = "./eval_results",
) -> dict:
    """Run all corruption experiments on a single model."""
    results = {}
    
    eval_timesteps = [100, 250, 500, 750]
    
    # Corruption configs to test
    corruptions = [
        ("patch_mask_p4_r25", mask_patches, {"patch_size": 4, "mask_ratio": 0.25}),
        ("patch_mask_p4_r50", mask_patches, {"patch_size": 4, "mask_ratio": 0.50}),
        ("patch_mask_p4_r75", mask_patches, {"patch_size": 4, "mask_ratio": 0.75}),
        ("patch_mask_p8_r25", mask_patches, {"patch_size": 8, "mask_ratio": 0.25}),
        ("patch_mask_p8_r50", mask_patches, {"patch_size": 8, "mask_ratio": 0.50}),
        ("center_block_25", mask_center_block, {"block_fraction": 0.25}),
        ("center_block_50", mask_center_block, {"block_fraction": 0.50}),
        ("random_block_25", mask_random_block, {"block_fraction": 0.25}),
        ("random_block_50", mask_random_block, {"block_fraction": 0.50}),
        ("stripes_h_4", mask_stripes, {"num_stripes": 4, "orientation": "horizontal"}),
        ("stripes_v_4", mask_stripes, {"num_stripes": 4, "orientation": "vertical"}),
    ]
    
    # Also run uncorrupted baseline (standard denoising)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    for corr_name, corr_fn, corr_kwargs in corruptions:
        print(f"\n  Corruption: {corr_name}")
        
        for t_eval in eval_timesteps:
            key = f"{corr_name}_t{t_eval}"
            
            # Average over multiple batches
            metrics_accum = {"mse_overall": 0, "mse_masked_region": 0, "mse_visible_region": 0}
            batch_count = 0
            
            for batch_idx, (images, _) in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                
                images = images.to(device)
                batch_metrics = evaluate_reconstruction(
                    model, images, schedule, corr_fn, corr_kwargs, t_eval
                )
                
                for k, v in batch_metrics.items():
                    metrics_accum[k] += v
                batch_count += 1
            
            # Average
            avg_metrics = {k: v / batch_count for k, v in metrics_accum.items()}
            results[key] = avg_metrics
            
            print(f"    t={t_eval:>4d}: MSE_overall={avg_metrics['mse_overall']:.5f} | "
                  f"MSE_masked={avg_metrics['mse_masked_region']:.5f} | "
                  f"MSE_visible={avg_metrics['mse_visible_region']:.5f}")
    
    # Also run standard (no corruption) baseline
    print(f"\n  Standard denoising (no corruption):")
    for t_eval in eval_timesteps:
        key = f"standard_t{t_eval}"
        metrics_accum = {"mse": 0}
        batch_count = 0
        
        for batch_idx, (images, _) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            B = images.shape[0]
            t = torch.full((B,), t_eval, device=device, dtype=torch.long)
            
            noise = torch.randn_like(images)
            sqrt_ab = schedule.sqrt_alpha_bar[t][:, None, None, None]
            sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
            x_t = sqrt_ab * images + sqrt_one_minus_ab * noise
            
            with torch.no_grad():
                noise_pred = model(x_t, t)
            
            mse = nn.functional.mse_loss(noise_pred, noise).item()
            metrics_accum["mse"] += mse
            batch_count += 1
        
        results[key] = {"mse": metrics_accum["mse"] / batch_count}
        print(f"    t={t_eval:>4d}: Denoising MSE={results[key]['mse']:.5f}")
    
    return results


def save_corruption_visualizations(
    model: nn.Module,
    images: torch.Tensor,
    schedule: DiffusionSchedule,
    output_dir: str,
    model_name: str,
    device: str,
):
    """Save visualization grids showing corruption -> denoising -> reconstruction."""
    os.makedirs(output_dir, exist_ok=True)
    images = images[:8].to(device)  # Just 8 images for visualization
    
    t_eval = 250
    B = images.shape[0]
    t = torch.full((B,), t_eval, device=device, dtype=torch.long)
    
    corruptions = [
        ("patch_mask_50", mask_patches, {"patch_size": 4, "mask_ratio": 0.5}),
        ("center_block_50", mask_center_block, {"block_fraction": 0.5}),
    ]
    
    for corr_name, corr_fn, corr_kwargs in corruptions:
        corrupted, mask = corr_fn(images, **corr_kwargs)
        
        noise = torch.randn_like(corrupted)
        sqrt_ab = schedule.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        x_t = sqrt_ab * corrupted + sqrt_one_minus_ab * noise
        
        with torch.no_grad():
            noise_pred = model(x_t, t)
        x0_pred = (x_t - sqrt_one_minus_ab * noise_pred) / sqrt_ab
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # Stack: original | corrupted | noised | reconstructed
        grid = torch.cat([
            (images + 1) / 2,
            (corrupted + 1) / 2,
            (x_t + 1) / 2,  # will look noisy
            (x0_pred + 1) / 2,
        ], dim=0)
        
        save_image(
            grid,
            os.path.join(output_dir, f"{model_name}_{corr_name}_vis.png"),
            nrow=B,
        )
        print(f"  Saved visualization: {model_name}_{corr_name}_vis.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_ckpt", type=str, required=True)
    parser.add_argument("--dit_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading U-Net...")
    unet_model, unet_config = load_model_from_checkpoint(args.unet_ckpt, str(device))
    
    print("Loading DiT...")
    dit_model, dit_config = load_model_from_checkpoint(args.dit_ckpt, str(device))
    
    # Diffusion schedule (same for both)
    schedule = DiffusionSchedule(device=str(device))
    
    # Test data
    test_loader = get_test_loader(batch_size=64)
    
    # Run evaluations
    unet_results = run_corruption_eval(
        unet_model, "UNet", schedule, test_loader, str(device),
        args.num_batches, args.output_dir
    )
    
    dit_results = run_corruption_eval(
        dit_model, "DiT", schedule, test_loader, str(device),
        args.num_batches, args.output_dir
    )
    
    # Save results
    all_results = {"unet": unet_results, "dit": dit_results}
    with open(os.path.join(args.output_dir, "corruption_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/corruption_results.json")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    test_iter = iter(test_loader)
    vis_images = next(test_iter)[0]
    
    save_corruption_visualizations(
        unet_model, vis_images, schedule, args.output_dir, "unet", str(device)
    )
    save_corruption_visualizations(
        dit_model, vis_images, schedule, args.output_dir, "dit", str(device)
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
