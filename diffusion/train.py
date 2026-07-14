"""
Training script for diffusion models.

Usage:
    python train.py --model unet --epochs 100 --batch_size 128
    python train.py --model dit --epochs 100 --batch_size 128
    python train.py --model dit --patch_size 2 --depth 12    # ablation
"""

import os
import argparse
import copy
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import ExperimentConfig, DiffusionConfig, UNetConfig, DiTConfig, TrainConfig
from models.unet import UNetDenoiser
from models.dit import DiT
from utils.diffusion import DiffusionSchedule, compute_loss, ddim_sample


def get_cifar10_loader(batch_size: int, train: bool = True) -> DataLoader:
    """Load CIFAR-10 with standard preprocessing (normalize to [-1, 1])."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=2, pin_memory=True, drop_last=train,
    )


def build_model(config: ExperimentConfig) -> nn.Module:
    """Instantiate the denoiser based on config."""
    if config.model_type == "unet":
        c = config.unet
        model = UNetDenoiser(
            in_channels=c.in_channels,
            base_channels=c.base_channels,
            channel_mults=c.channel_mults,
            num_res_blocks=c.num_res_blocks,
            attn_resolutions=c.attn_resolutions,
            time_dim=c.time_dim,
            dropout=c.dropout,
            num_heads=c.num_heads,
            num_groups=c.num_groups,
            image_size=c.image_size,
        )
    elif config.model_type == "dit":
        c = config.dit
        model = DiT(
            image_size=c.image_size,
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            hidden_dim=c.hidden_dim,
            depth=c.depth,
            num_heads=c.num_heads,
            mlp_ratio=c.mlp_ratio,
            time_dim=c.time_dim,
            dropout=c.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    
    def update(self, model: nn.Module):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def train(config: ExperimentConfig):
    """Main training loop."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
    
    # Save config
    config.save(os.path.join(config.output_dir, "config.json"))
    
    # Seed
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)
    
    # Data
    train_loader = get_cifar10_loader(config.train.batch_size, train=True)
    
    # Model
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.model_type} | Parameters: {num_params:,}")
    
    # Diffusion schedule
    schedule = DiffusionSchedule(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=str(device),
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    
    # EMA
    ema = EMA(model, decay=config.train.ema_decay)
    
    # Training loop
    global_step = 0
    log_loss = 0.0
    log_count = 0
    start_time = time.time()
    
    # Log file
    log_file = open(os.path.join(config.output_dir, "train_log.csv"), "w")
    log_file.write("step,epoch,loss,lr,time_elapsed\n")
    
    print(f"\nStarting training for {config.train.num_epochs} epochs...")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: ~{len(train_loader) * config.train.num_epochs}")
    print("-" * 60)
    
    for epoch in range(config.train.num_epochs):
        model.train()
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Sample random timesteps
            t = torch.randint(
                0, config.diffusion.num_timesteps,
                (images.shape[0],), device=device
            )
            
            # Compute loss
            loss = compute_loss(model, images, t, schedule)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if config.train.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            
            optimizer.step()
            ema.update(model)
            
            # Logging
            log_loss += loss.item()
            log_count += 1
            global_step += 1
            
            if global_step % config.train.log_every == 0:
                avg_loss = log_loss / log_count
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                
                print(f"Step {global_step:>7d} | Epoch {epoch+1:>3d} | "
                      f"Loss: {avg_loss:.5f} | "
                      f"Steps/s: {steps_per_sec:.1f} | "
                      f"Time: {elapsed/60:.1f}m")
                
                log_file.write(f"{global_step},{epoch},{avg_loss:.6f},"
                               f"{config.train.learning_rate},{elapsed:.1f}\n")
                log_file.flush()
                
                log_loss = 0.0
                log_count = 0
            
            # Generate samples
            if global_step % config.train.sample_every == 0:
                print("  Generating samples...")
                n = config.train.num_sample_images
                samples = ddim_sample(
                    ema.shadow, schedule,
                    shape=(n, 3, 32, 32),
                    num_steps=50, device=str(device), verbose=False
                )
                # Rescale from [-1,1] to [0,1]
                samples = (samples + 1) / 2
                save_image(
                    samples,
                    os.path.join(config.output_dir, "samples", f"step_{global_step:07d}.png"),
                    nrow=int(math.sqrt(n)),
                )
                print(f"  Saved samples at step {global_step}")
            
            # Save checkpoint
            if global_step % config.train.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": config.__dict__,
                }
                path = os.path.join(
                    config.output_dir, "checkpoints", f"step_{global_step:07d}.pt"
                )
                torch.save(ckpt, path)
                print(f"  Saved checkpoint at step {global_step}")
    
    # Final save
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": config.train.num_epochs,
    }
    torch.save(ckpt, os.path.join(config.output_dir, "checkpoints", "final.pt"))
    log_file.close()
    print(f"\nTraining complete! Total steps: {global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion models")
    
    # Model selection
    parser.add_argument("--model", type=str, default="unet", choices=["unet", "dit"])
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # U-Net specific
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    
    # DiT specific
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    
    # Other
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Build config
    config = ExperimentConfig(
        name=args.name or f"{args.model}_cifar10",
        model_type=args.model,
        train=TrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            seed=args.seed,
        ),
        output_dir=os.path.join(args.output_dir, args.name or f"{args.model}_cifar10"),
        device=args.device,
    )
    
    # Override architecture settings from CLI
    if args.model == "unet":
        config.unet.base_channels = args.base_channels
        config.unet.num_res_blocks = args.num_res_blocks
    elif args.model == "dit":
        config.dit.patch_size = args.patch_size
        config.dit.hidden_dim = args.hidden_dim
        config.dit.depth = args.depth
        config.dit.num_heads = args.num_heads
    
    train(config)


if __name__ == "__main__":
    main()
