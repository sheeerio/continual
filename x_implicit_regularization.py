import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import wandb
import matplotlib.pyplot as plt
import math
from collections import deque
from config import get_parser
from utils import misc, schedulers, optimizers
from models import mlp, cnn
from datasets import data_loader
from utils.optimizers import PerLayerLyapunovScheduler

activations = {}


def compute_use_for_activation(h):
    with torch.no_grad():
        pos = (h > 0).float().mean(dim=0)
        eps = 1e-12
        ent = -(pos * (pos + eps).log() + (1 - pos) * (1 - pos + eps).log())
        return ent.mean().item()


def save_activations(name):
    def hook(m, inp, out):
        activations[name] = out

    return hook


parser = get_parser()
config = parser.parse_args()
if not hasattr(config, "snr_margin"):
    config.snr_margin = 0.0
if not hasattr(config, "snr_pred_window"):
    config.snr_pred_window = 20
LENGTH_CHOICES = [100, 300, 50, 150]

rho = config.sam_rho
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
noise_power_full = 0.0
noise_power_mb = 0.0


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)


set_seed(config.seed)

task_lengths = [random.choice(LENGTH_CHOICES) for _ in range(config.runs)]

train_dataset, test_dataset, in_ch, input_size, DATA_MEAN, DATA_STD = (
    data_loader.get_dataset(config)
)

config.alpha = 0.01 if config.activation == "leaky_relu" else config.alpha
hidden = 256
if config.model == "MLP":
    model = mlp.MLP(input_size, hidden, 10).to(device)
elif config.model == "BatchNormMLP":
    model = mlp.BatchNormMLP(input_size, hidden, 10).to(device)
elif config.model == "LinearNet":
    model = nn.Sequential(nn.Flatten(), nn.Identity()).to(device)
elif config.model == "CNN":
    model = cnn.CNN(in_ch).to(device)
elif config.model == "BatchNormCNN":
    model = cnn.BatchNormCNN(in_ch).to(device)
else:
    model = mlp.MLP(input_size, hidden, 10).to(device)

init_params = {
    n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
}
if config.reg == "wass":
    for n, p0 in init_params.items():
        init_params[n] = torch.sort(p0.view(-1))[0].to(device)

activations = {}


def save_activations(name):
    def hook(m, inp, out):
        activations[name] = out

    return hook


if config.model in [
    "MLP",
    "LayerNormMLP",
    "BatchNormMLP",
    "LeakyLayerNormMLP",
    "LeakyKaimingLayerNormMLP",
]:
    model.fc1.register_forward_hook(save_activations("l1"))
    model.fc2.register_forward_hook(save_activations("l2"))
    # model.fc3.register_forward_hook(save_activations("l3"))
else:
    model.fc1.register_forward_hook(save_activations("l1"))


def make_layer_groups(model, base_lr):
    layer_map = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            layer = name.split(".")[0]
            layer_map.setdefault(layer, []).append(p)
    layer_groups = [
        {"params": params, "lr": base_lr, "layer": layer}
        for layer, params in layer_map.items()
    ]
    return layer_map, layer_groups


layer_map, layer_groups = make_layer_groups(model, config.lr)
# ── one EMAState per layer ─────────────────────────────────────────────
layer_states = {
    layer: misc.EMAState(alphas=(0.01, 0.05, 0.5))  # same alphas you used globally
    for layer in layer_map
}
act_layer_states = {
    layer: misc.EMAState(alphas=(0.01, 0.05, 0.5))  # same alphas you used globally
    for layer in layer_map
}

criterion = nn.CrossEntropyLoss(reduction="mean")
criterion_nored = nn.CrossEntropyLoss(reduction="none")  # for within-batch σ² only
wd = config.l2_lambda if config.reg == "l2" else 0.0
if config.optimizer == "adam":
    optimizer = optim.Adam(
        layer_groups, lr=config.lr, weight_decay=wd, betas=(0.9, config.beta2)
    )
elif config.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9
    )
elif config.optimizer == "clamped_adam":
    optimizer = optimizers.ClampedAdam(
        layer_groups, lr=config.lr, lr_min=1e-3, lr_max=1.1
    )
#
# base_optimizer = optimizer
# optimizer = optimizers.CVSharpnessController(
#     base_optimizer,
#     target = 3.0,     # keep CV≈1
#     k_lr   = 0.3,
#     k_wd   = 0.15,
#     band   = 0.05,
#     window = 100
# )

run = wandb.init(
    project=f"workshop_{config.dataset}",
    entity="sheerio",
    group=config.exp_name,
    name=config.name,
    config={
        "model": config.model,
        "dataset": config.dataset,
        "activation": config.activation,
        "reg": config.reg,
        "batch_size": config.batch_size,
        "runs": config.runs,
        "random_seed": config.seed,
        "l2_lambda": config.l2_lambda,
        "spectral_lambda": config.spectral_lambda,
        "spectral_k": config.spectral_k,
        "wass_lambda": config.wass_lambda,
    },
)
wandb.define_metric("gradient_noise", hidden=False)
wandb.define_metric("gradient_noise_mb", hidden=False)
wandb.define_metric("true_grad_norm_sq", hidden=False)
wandb.define_metric("task_lam_var", hidden=False)
# SNR metrics
wandb.define_metric("snr_T", hidden=False)
wandb.define_metric("snr_sigma2_hat", hidden=False)
wandb.define_metric("mb_sigma2_hat", hidden=False)
wandb.define_metric("mb_snr_T", hidden=False)

# tracker and series for plotting later
snr_tracker = optimizers.GradSNR()
snr_T_series = []  # list of (update_idx, T_t)
snr_predictor = optimizers.SNRProgressPredictor(
    margin=config.snr_margin, window=config.snr_pred_window
)
wandb.define_metric("snr_pred", hidden=False)
wandb.define_metric("snr_pred_conf", hidden=True)
wandb.define_metric("snr_T_mean", hidden=True)
wandb.define_metric("snr_T_thresh", hidden=True)

results = {
    config.activation: {
        "batch_error": [],
        "param_norm": [],
        "update_norm": [],
        "effective_rank": [],
        "dormancy": [],
    }
}

sharp_state = misc.EMAState(alphas=(0.01, 0.05, 0.5))
r_sharp_state = misc.EMAState(alphas=(0.01, 0.05, 0.5))
lambda_state = misc.EMAState(alphas=(0.01, 0.05, 0.5))

init_sigma_min = {}
print_val = 0.0
for name, p in model.named_parameters():
    if p.requires_grad and p.ndim >= 2:
        # 1–3 inverse-power iterations are plenty at t=0
        s0 = optimizers.power_iteration_sigma_min(p.detach(), iters=3).item()
        init_sigma_min[name] = s0
        print_val += s0
        print(s0)
print(print_val)

task_lengths = [300, 50, 300, 300, 100] * 4
ns = [1] + 9 * [0]
for task in range(config.runs):
    snr_sum = 0.0
    # k_snr_sum = 0.0
    # s_snr_sum = 0.0
    # sharp**2 / mean for both r_sharp_log and sharp_log
    k_ss_sum1 = 0.0
    k_ss_sum10 = 0.0
    k_rs_sum1 = 0.0
    k_rs_sum10 = 0.0
    s_scv_sum10 = 0.0
    s_svar_sum10 = 0.0
    s_rcv_sum10 = 0.0
    s_rvar_sum10 = 0.0
    s_scv_sum1 = 0.0
    s_sqm_sum1 = 0.0
    s_sqm_sum10 = 0.0
    s_svar_sum1 = 0.0
    s_rcv_sum1 = 0.0
    s_rvar_sum1 = 0.0
    s_rsqm_sum1 = 0.0
    s_rsqm_sum10 = 0.0
    ly_snr_sum = 0.0
    ly_union_sum = 0.0
    ly_snr_2_sum = 0.0
    eff_acrit_scv1_union_sum = 0.0
    eff_acrit_ss1_union_sum = 0.0
    eff_acrit_svar1_union_sum = 0.0
    eff_acrit_ssqm1_union_sum = 0.0
    eff_acrit_scv10_union_sum = 0.0
    eff_acrit_ss10_union_sum = 0.0
    eff_acrit_svar10_union_sum = 0.0
    eff_acrit_ssqm10_union_sum = 0.0
    eff_acrit_rcv1_union_sum = 0.0
    eff_acrit_rs1_union_sum = 0.0
    eff_acrit_rvar1_union_sum = 0.0
    eff_acrit_rsqm1_union_sum = 0.0
    eff_acrit_rcv10_union_sum = 0.0
    eff_acrit_rs10_union_sum = 0.0
    eff_acrit_rvar10_union_sum = 0.0
    eff_acrit_rsqm10_union_sum = 0.0
    config.epochs = task_lengths[task] if config.random_length else config.epochs
    print(config.epochs)

    sharpness_volatility = []
    sharpness_var = []
    if config.reset_model:
        if config.model == "MLP":
            model = mlp.MLP(input_size, hidden, 10).to(device)
        elif config.model == "CNN":
            model = cnn.CNN(in_ch).to(device)
        elif config.model == "BatchNormCNN":
            model = cnn.BatchNormCNN(in_ch).to(device)
        else:
            model = mlp.BatchNormMLP(input_size, hidden, 10).to(device)
        if config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=wd,
                betas=(config.beta1, config.beta2),
            )
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(
                layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9
            )
    if config.reset_optimizer:
        layer_map, layer_groups = make_layer_groups(model, config.lr)
        # Recreate optimizer using layer_groups (not model.parameters())
        if config.optimizer == "adam":
            optimizer = optim.Adam(
                layer_groups,
                lr=config.lr,
                weight_decay=wd,
                betas=(config.beta1, config.beta2),
            )
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(
                layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9
            )
        elif config.optimizer == "clamped_adam":
            optimizer = optimizers.ClampedAdam(
                layer_groups, lr=config.lr, lr_min=1e-2, lr_max=0.8
            )

    if config.dataset == "PermutedMNIST":
        perm_tf = make_perm_tf(task)
        train_dataset.dataset.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD), perm_tf]
        )
        loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
    elif config.dataset == "Shuffle_CIFAR":
        mapping = torch.randperm(10).tolist()
        remapped = [mapping[orig] for orig in orig_labels]
        for idx, new_lbl in zip(subset_indices, remapped):
            train_subset.dataset.targets[idx] = new_lbl
        train_dataset = train_subset
        loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
    else:
        # -x-x-x-x smooth non-stationary experiment -x-x-x-x
        train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
        # if task == 0:
        #     train_dataset = optimizers.randomize_targets(train_dataset, 0.0)
        # -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
        loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )

    total_updates = 0
    if total_updates % config.log_interval == 0:
        hessian_rank = optimizers.empirical_fischer_rank(
            model, train_dataset, device, cfg=config
        )

    model.train()
    ly_sched = None
    scheduler = None
    total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
    if config.lr_schedule == "linear":
        initial_lr = config.lr
        final_lr = config.final_lr
        decay_range = initial_lr - final_lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(
                0.0,
                (initial_lr - (decay_range * step / total_steps)) / initial_lr,
            ),
        )
    elif config.lr_schedule == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.lr_schedule == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    elif config.lr_schedule == "polynomial":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / total_steps) ** config.power,
        )
    elif config.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs
        )
    elif config.lr_schedule == "wsd":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=schedulers.wsd_lambda
        )
    elif config.lr_schedule == "power":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=schedulers.power_lambda
        )
    elif config.lr_schedule == "skew":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=schedulers.skew_lambda
        )
    elif config.lr_schedule == "lyapunov":
        # hyper-params exposed to the CLI for convenience
        ly_safety = config.safety
        ly_cool = config.cool
        ly_warm = config.warm
        ly_sched = optimizers.LyapunovScheduler(
            optimizer,
            ema_state=sharp_state,  # <<– share!
            safety=ly_safety,
            cool=ly_cool,
            warm=ly_warm,
            cfg=config,
        )
        scheduler = None
    elif config.lr_schedule == "pl_lyapunov":
        pl_scheduler = PerLayerLyapunovScheduler(
            optimizer=optimizer,
            layer_states={
                layer: layer_states.get(layer, misc.EMAState(alphas=(0.01, 0.05, 0.5)))
                for layer in layer_map
            },
            safety=config.safety,
            cool=config.cool,
            warm=config.warm,
            cfg=config,
        )
    else:
        scheduler = None

    sum_up = 0.0
    this_task_acc = 0.0
    this_normalized_sharp = 0.0
    # if task == config.runs - 1:
    #     config.epochs = 1500
    cached_sigma_min = {}
    for epoch in range(config.epochs):
        for x, y in loader:
            if config.model in ["CNN", "BatchNormCNN"]:
                inputs = x.to(device)
            else:
                inputs = x.view(x.size(0), -1).to(device)
            labels = y.to(device)

            optimizer.zero_grad()
            out = model(inputs)
            preds = out.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            this_task_acc += acc
            base = criterion(out, labels)
            reg = torch.tensor(0.0, device=device)
            if config.reg == "l2_init":
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += (p - init_params[n]).pow(2).sum()
                reg *= config.l2_lambda
            elif config.reg == "wass":
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += (torch.sort(p.view(-1))[0] - init_params[n]).pow(2).sum()
                reg *= config.wass_lambda
            elif config.reg == "spectral":
                for n, p in model.named_parameters():
                    if p.requires_grad and p.ndim >= 2:
                        reg += (
                            optimizers.power_iteration(p, 1).pow(config.spectral_k)
                            - 1.0
                        ).pow(2)
                reg *= config.spectral_lambda
            elif config.reg == "ortho":
                frac = config.ortho_frac
                for name, p in model.named_parameters():
                    if p.ndim < 2 or not p.requires_grad:
                        continue
                    if total_updates % config.ortho_interval == 0:
                        sigma_now = optimizers.power_iteration_sigma_min(
                            p, iters=1
                        ).detach()
                        cached_sigma_min[name] = sigma_now
                    else:
                        sigma_now = cached_sigma_min.get(
                            name,
                            optimizers.power_iteration_sigma_min(p, iters=1).detach(),
                        )
                    target = frac * init_sigma_min[name]
                    reg += (sigma_now - target).pow(2)
                reg *= config.ortho_lambda
            elif config.reg == "orthofrob":
                for name, p in model.named_parameters():
                    if p.ndim >= 2 and p.requires_grad:
                        W = p.view(p.shape[0], -1)
                        k = W.shape[1]
                        I = torch.eye(k, device=W.device, dtype=W.dtype)
                        reg += (W.t() @ W - I).pow(2).sum()
                reg *= config.ortho_lambda

            loss = base + reg
            params = [p for p in model.parameters() if p.requires_grad]
            old = [p.data.clone() for p in params]

            # nus = [
            #     optimizer.state[p]["exp_avg_sq"]
            #     for p in optimizer.param_groups[0]["params"]
            #     if "exp_avg_sq" in optimizer.state[p]
            # ]

            # if total_updates % TRACE_INTERVAL == 0:
            #     trace_val = optimizers.hessian_trace(loss, params, n_samples=10)

            if total_updates % config.log_interval == 0:
                #     if config.optimizer == "adam":
                #         layer_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                #     else:  # SGD (+ momentum)
                #         layer_eff_lrs = optimizers.per_layer_sgd_lr(
                #             model, optimizer, step=total_updates, sgd_mode="time"
                #         )

                #     layer_sigma2_mb = optimizers.grad_variance_within_batch_by_layer(
                #         model, criterion_nored, inputs, labels, layer_map
                #     )

                #     # union across layers for collapse_pred (NO pred2)
                #     union_pred_step = 0
                #     union_eff_gt_acrit_step_scv1 = 0
                #     union_eff_gt_acrit_step_ss1 = 0
                #     union_eff_gt_acrit_step_svar1 = 0
                #     union_eff_gt_acrit_step_scv10 = 0
                #     union_eff_gt_acrit_step_ss10 = 0
                #     union_eff_gt_acrit_step_svar10 = 0
                #     union_eff_gt_acrit_step_sqm1 = 0
                #     union_eff_gt_acrit_step_sqm10 = 0
                #     union_eff_gt_acrit_step_rcv1 = 0
                #     union_eff_gt_acrit_step_rs1 = 0
                #     union_eff_gt_acrit_step_rvar1 = 0
                #     union_eff_gt_acrit_step_rsqm1 = 0
                #     union_eff_gt_acrit_step_rcv10 = 0
                #     union_eff_gt_acrit_step_rs10 = 0
                #     union_eff_gt_acrit_step_rvar10 = 0
                #     union_eff_gt_acrit_step_rsqm10 = 0

                #     for layer, params in layer_map.items():
                #         lam = optimizers.estimate_hessian_topk(model, loss, params, k=1)[0]
                #         norm_lam = optimizers.get_norm_sharpness(optimizer, lam, config)
                #         eff_lr = layer_eff_lrs.get(layer, optimizer.param_groups[0]["lr"])
                #         state, scalars = misc.update_stat(norm_lam, layer_states[layer], eff_lr)
                #         act_state, act_scalars = misc.update_stat(lam, act_layer_states[layer], eff_lr)

                #         # strictly use collapse_pred for the union
                #         layer_pred = int(scalars["collapse_pred2"])
                #         union_pred_step = max(union_pred_step, layer_pred)

                #         # ---- per-layer alpha_crit_s (unchanged) ----
                #         gi = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
                #         g_layer_sq = float(torch.cat([g.contiguous().view(-1) for g in gi]).pow(2).sum().item())

                #         sigma2_l = float(layer_sigma2_mb.get(layer, 0.0))
                #         s_scv_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * scalars["lam_cv"]
                #         s_ss_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * norm_lam
                #         s_svar_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * scalars["lam_var"]
                #         s_sqm_sigma2_l1 = sigma2_l + 75.0 * g_layer_sq * scalars["sq_mean"]
                #         s_scv_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["lam_cv"]
                #         s_ss_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * norm_lam
                #         s_svar_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["lam_var"]
                #         s_sqm_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["sq_mean"]
                #         # actual sharpness metric
                #         s_rcv_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * act_scalars["lam_cv"]
                #         s_rs_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * lam
                #         s_rvar_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * act_scalars["lam_var"]
                #         s_rsqm_sigma2_l1 = sigma2_l + 75.0 * g_layer_sq * act_scalars["sq_mean"]
                #         s_rcv_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["lam_cv"]
                #         s_rs_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * lam
                #         s_rvar_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["lam_var"]
                #         s_rsqm_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["sq_mean"]

                #         r = (sigma2_l / max(1, inputs.size(0))) / (g_layer_sq + 1e-12) + 1e-12
                #         eta_crit = min(0.8/r, 2/(norm_lam * (1 + r)))

                #         B = int(inputs.size(0))
                #         alpha_crit_scv_layer1 = (B * g_layer_sq) / max(s_scv_sigma2_l1, 1e-12)
                #         alpha_crit_ss_layer1 = (B * g_layer_sq) / max(s_ss_sigma2_l1, 1e-12)
                #         alpha_crit_svar_layer1 = (B * g_layer_sq) / max(s_svar_sigma2_l1, 1e-12)
                #         alpha_crit_sqm_layer1 = (B * g_layer_sq) / max(s_sqm_sigma2_l1, 1e-12)
                #         alpha_crit_scv_layer10 = (B * g_layer_sq) / max(s_scv_sigma2_l10, 1e-12)
                #         alpha_crit_ss_layer10 = (B * g_layer_sq) / max(s_ss_sigma2_l10, 1e-12)
                #         alpha_crit_svar_layer10 = (B * g_layer_sq) / max(s_svar_sigma2_l10, 1e-12)
                #         alpha_crit_sqm_layer10 = (B * g_layer_sq) / max(s_sqm_sigma2_l10, 1e-12)
                #         alpha_crit_rcv_layer1 = (B * g_layer_sq) / max(s_rcv_sigma2_l1, 1e-12)
                #         alpha_crit_rs_layer1 = (B * g_layer_sq) / max(s_rs_sigma2_l1, 1e-12)
                #         alpha_crit_rvar_layer1 = (B * g_layer_sq) / max(s_rvar_sigma2_l1, 1e-12)
                #         alpha_crit_rsqm_layer1 = (B * g_layer_sq) / max(s_rsqm_sigma2_l1, 1e-12)
                #         alpha_crit_rcv_layer10 = (B * g_layer_sq) / max(s_rcv_sigma2_l10, 1e-12)
                #         alpha_crit_rs_layer10 = (B * g_layer_sq) / max(s_rs_sigma2_l10, 1e-12)
                #         alpha_crit_rvar_layer10 = (B * g_layer_sq) / max(s_rvar_sigma2_l10, 1e-12)
                #         alpha_crit_rsqm_layer10 = (B * g_layer_sq) / max(s_rsqm_sigma2_l10, 1e-12)
                #         eff_gt_acrit_scv1 = int(eff_lr > alpha_crit_scv_layer1)
                #         eff_gt_acrit_ss1 = int(eff_lr > alpha_crit_ss_layer1)
                #         eff_gt_acrit_svar1 = int(eff_lr > alpha_crit_svar_layer1)
                #         eff_gt_acrit_sqm1 = int(eff_lr > alpha_crit_sqm_layer1)
                #         eff_gt_acrit_scv10 = int(eff_lr > alpha_crit_scv_layer10)
                #         eff_gt_acrit_ss10 = int(eff_lr > alpha_crit_ss_layer10)
                #         eff_gt_acrit_svar10 = int(eff_lr > alpha_crit_svar_layer10)
                #         eff_gt_acrit_sqm10 = int(eff_lr > alpha_crit_sqm_layer10)
                #         eff_gt_acrit_rcv1 = int(eff_lr > alpha_crit_rcv_layer1)
                #         eff_gt_acrit_rs1 = int(eff_lr > alpha_crit_rs_layer1)
                #         eff_gt_acrit_rvar1 = int(eff_lr > alpha_crit_rvar_layer1)
                #         eff_gt_acrit_rsqm1 = int(eff_lr > alpha_crit_rsqm_layer1)
                #         eff_gt_acrit_rcv10 = int(eff_lr > alpha_crit_rcv_layer10)
                #         eff_gt_acrit_rs10 = int(eff_lr > alpha_crit_rs_layer10)
                #         eff_gt_acrit_rvar10 = int(eff_lr > alpha_crit_rvar_layer10)
                #         eff_gt_acrit_rsqm10 = int(eff_lr > alpha_crit_rsqm_layer10)

                #         union_eff_gt_acrit_step_scv1 = max(union_eff_gt_acrit_step_scv1, eff_gt_acrit_scv1)
                #         union_eff_gt_acrit_step_ss1 = max(union_eff_gt_acrit_step_ss1, eff_gt_acrit_ss1)
                #         union_eff_gt_acrit_step_svar1 = max(union_eff_gt_acrit_step_svar1, eff_gt_acrit_svar1)
                #         union_eff_gt_acrit_step_sqm1 = max(union_eff_gt_acrit_step_sqm1, eff_gt_acrit_sqm1)
                #         union_eff_gt_acrit_step_scv10 = max(union_eff_gt_acrit_step_scv10, eff_gt_acrit_scv10)
                #         union_eff_gt_acrit_step_ss10 = max(union_eff_gt_acrit_step_ss10, eff_gt_acrit_ss10)
                #         union_eff_gt_acrit_step_svar10 = max(union_eff_gt_acrit_step_svar10, eff_gt_acrit_svar10)
                #         union_eff_gt_acrit_step_sqm10 = max(union_eff_gt_acrit_step_sqm10, eff_gt_acrit_sqm10)
                #         union_eff_gt_acrit_step_rcv1 = max(union_eff_gt_acrit_step_rcv1, eff_gt_acrit_rcv1)
                #         union_eff_gt_acrit_step_rs1 = max(union_eff_gt_acrit_step_rs1, eff_gt_acrit_rs1)
                #         union_eff_gt_acrit_step_rvar1 = max(union_eff_gt_acrit_step_rvar1, eff_gt_acrit_rvar1)
                #         union_eff_gt_acrit_step_rsqm1 = max(union_eff_gt_acrit_step_rsqm1, eff_gt_acrit_rsqm1)
                #         union_eff_gt_acrit_step_rcv10 = max(union_eff_gt_acrit_step_rcv10, eff_gt_acrit_rcv10)
                #         union_eff_gt_acrit_step_rs10 = max(union_eff_gt_acrit_step_rs10, eff_gt_acrit_rs10)
                #         union_eff_gt_acrit_step_rvar10 = max(union_eff_gt_acrit_step_rvar10, eff_gt_acrit_rvar10)
                #         union_eff_gt_acrit_step_rsqm10 = max(union_eff_gt_acrit_step_rsqm10, eff_gt_acrit_rsqm10)

                #         wandb.log({
                #             f"{layer}/sharp"       : norm_lam,
                #             f"{layer}/mu"          : scalars["lam_mean"],
                #             f"{layer}/tau"         : scalars["tau"],
                #             f"{layer}/cv"          : scalars["lam_cv"],
                #             f"{layer}/eff_lr"      : eff_lr,
                #             f"{layer}/predict"     : layer_pred,
                #             # f"{layer}/alpha_crit_s": float(alpha_crit_s_layer),
                #             f"{layer}/alpha_crit_scv1": float(alpha_crit_scv_layer1),
                #             f"{layer}/alpha_crit_ss1": float(alpha_crit_ss_layer1),
                #             f"{layer}/alpha_crit_svar1": float(alpha_crit_svar_layer1),
                #             f"{layer}/alpha_crit_scv10": float(alpha_crit_scv_layer10),
                #             f"{layer}/alpha_crit_ss10": float(alpha_crit_ss_layer10),
                #             f"{layer}/alpha_crit_svar10": float(alpha_crit_svar_layer10),
                #             f"{layer}/alpha_crit_sqm1": float(alpha_crit_sqm_layer1),
                #             f"{layer}/alpha_crit_sqm10": float(alpha_crit_sqm_layer10),
                #             f"{layer}/alpha_crit_rcv1": float(alpha_crit_rcv_layer1),
                #             f"{layer}/alpha_crit_rs1": float(alpha_crit_rs_layer1),
                #             f"{layer}/alpha_crit_rvar1": float(alpha_crit_rvar_layer1),
                #             f"{layer}/alpha_crit_rsqm1": float(alpha_crit_rsqm_layer1),
                #             f"{layer}/alpha_crit_rcv10": float(alpha_crit_rcv_layer10),
                #             f"{layer}/alpha_crit_rs10": float(alpha_crit_rs_layer10),
                #             f"{layer}/alpha_crit_rvar10": float(alpha_crit_rvar_layer10),
                #             f"{layer}/alpha_crit_rsqm10": float(alpha_crit_rsqm_layer10),
                #             "reg"                  : reg
                #         })

                #         if config.lr_schedule == "pl_lyapunov":
                #             # if task <= 2:
                #             if config.param == "sqm10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_sqm_layer10, total_updates, total_steps)
                #             elif config.param == "sqm1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_sqm_layer1, total_updates, total_steps)
                #             elif config.param == "scv10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_scv_layer10, total_updates, total_steps)
                #             elif config.param == "scv1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_scv_layer1, total_updates, total_steps)
                #             elif config.param == "svar10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_svar_layer10, total_updates, total_steps)
                #             elif config.param == "svar1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_svar_layer1, total_updates, total_steps)
                #             elif config.param == "ss10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_ss_layer10, total_updates, total_steps)
                #             elif config.param == "ss1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_ss_layer1, total_updates, total_steps)
                #             elif config.param == "rcv1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rcv_layer1, total_updates, total_steps)
                #             elif config.param == "rs1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rs_layer1, total_updates, total_steps)
                #             elif config.param == "rvar1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rvar_layer1, total_updates, total_steps)
                #             elif config.param == "rsqm1":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rsqm_layer1, total_updates, total_steps)
                #             elif config.param == "rcv10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rcv_layer10, total_updates, total_steps)
                #             elif config.param == "rs10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rs_layer10, total_updates, total_steps)
                #             elif config.param == "rvar10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rvar_layer10, total_updates, total_steps)
                #             elif config.param == "rsqm10":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rsqm_layer10, total_updates, total_steps)
                #             elif config.param == "s_tau":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, scalars["tau"], total_updates, total_steps)
                #             elif config.param == "r_tau":
                #                 lr_star = pl_scheduler.step(layer, eff_lr, act_scalars["tau"], total_updates, total_steps)
                #             # lr_star = pl_scheduler.step(layer, eff_lr, scalars["tau"], config)

                #     # accumulate union across layers for this log step
                #     ly_union_sum += union_pred_step
                #     eff_acrit_scv1_union_sum += union_eff_gt_acrit_step_scv1
                #     eff_acrit_ss1_union_sum += union_eff_gt_acrit_step_ss1
                #     eff_acrit_svar1_union_sum += union_eff_gt_acrit_step_svar1
                #     eff_acrit_ssqm1_union_sum += union_eff_gt_acrit_step_sqm1
                #     eff_acrit_scv10_union_sum += union_eff_gt_acrit_step_scv10
                #     eff_acrit_ss10_union_sum += union_eff_gt_acrit_step_ss10
                #     eff_acrit_svar10_union_sum += union_eff_gt_acrit_step_svar10
                #     eff_acrit_ssqm10_union_sum += union_eff_gt_acrit_step_sqm10
                #     eff_acrit_rcv1_union_sum += union_eff_gt_acrit_step_rcv1
                #     eff_acrit_rs1_union_sum += union_eff_gt_acrit_step_rs1
                #     eff_acrit_rvar1_union_sum += union_eff_gt_acrit_step_rvar1
                #     eff_acrit_rsqm1_union_sum += union_eff_gt_acrit_step_rsqm1
                #     eff_acrit_rcv10_union_sum += union_eff_gt_acrit_step_rcv10
                #     eff_acrit_rs10_union_sum += union_eff_gt_acrit_step_rs10
                #     eff_acrit_rvar10_union_sum += union_eff_gt_acrit_step_rvar10
                #     eff_acrit_rsqm10_union_sum += union_eff_gt_acrit_step_rsqm10

                eigs = optimizers.estimate_hessian_topk(model, loss, params, k=1)
                sharpness = eigs[0]
            #     lambda_min = optimizers.estimate_hessian_min_eig(model, loss, params, iters=20)

            #     norm_sharpness = optimizers.get_norm_sharpness(optimizer, sharpness, config)
            #     # lambda_min_norm = optimizers.get_norm_sharpness(optimizer, lambda_min, config)
            #     this_normalized_sharp += norm_sharpness

            #     effective_lr = optimizers.compute_effective_lr(
            #         optimizer, cfg=config, step=total_updates, sgd_mode="time"
            #     )

            #     sharp_state,  sharp_log  = misc.update_stat(norm_sharpness,  sharp_state,  effective_lr)
            #     r_sharp_state,  r_sharp_log  = misc.update_stat(sharpness, r_sharp_state,  effective_lr)
            #     # lambda_state, lambda_log = misc.update_stat(lambda_min_norm, lambda_state, effective_lr)
            #     ly_snr_sum += sharp_log["collapse_pred"]
            #     ly_snr_2_sum += sharp_log["collapse_pred2"]

            # Sharpness Aware Minimization
            if config.sam:
                loss.backward(create_graph=True)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                grad_flat = torch.cat([g.view(-1) for g in grads])
                grad_norm = grad_flat.norm() + 1e-12
                epsilons = [(rho / grad_norm) * g for g in grads]

                for p, e in zip(params, epsilons):
                    p.data.add_(e)
                out_adv = model(inputs)
                loss_adv = criterion(out_adv, labels) + reg
                optimizer.zero_grad()
                loss_adv.backward()

                for p, e in zip(params, epsilons):
                    p.data.sub_(e)

            else:
                loss.backward(retain_graph=True)
                mark = {"sharpness": sharpness}
                # if total_updates % config.log_interval == 0:
                #     params = [p for p in model.parameters() if p.requires_grad]
                #     grad_flat = torch.cat([p.grad.view(-1) for p in params])

                #     # 1) use the *current* grads to get ||g||^2 BEFORE any SNR math
                #     true_grad_norm_sq = float(grad_flat.pow(2).sum().item())

                #     # 2) temporal proxy SNR (unchanged)
                #     eta_eff = effective_lr
                #     batch_B = inputs.size(0)
                #     T_t, sigma2_hat = snr_tracker.update(grad_flat.detach(), eta_eff, int(batch_B))
                #     if T_t is not None:
                #         snr_T_series.append((total_updates, float(T_t)))

                #     # 3) within-minibatch σ² (Mark’s definition)
                #     sigma2_hat_mb = optimizers.grad_variance_within_batch(model, criterion_nored, inputs, labels)
                #     k_rs_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharpness
                #     k_ss_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*norm_sharpness
                #     k_rs_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharpness
                #     k_ss_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*norm_sharpness
                #     s_scv_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["lam_cv"]
                #     s_svar_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["lam_var"]
                #     s_rcv_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["lam_cv"]
                #     s_rvar_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["lam_var"] # decent contender w/ a smaller coeff
                #     s_scv_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharp_log["lam_cv"]
                #     s_svar_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharp_log["lam_var"]
                #     s_rcv_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["lam_cv"]
                #     s_rvar_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["lam_var"]
                #     s_ssqm_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["sq_mean"]
                #     s_rsqm_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["sq_mean"]
                #     s_ssqm_sigma2_hat_mb1 = sigma2_hat_mb + 75.0 * true_grad_norm_sq*sharp_log["sq_mean"]
                #     s_rsqm_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["sq_mean"]

                #     r = (sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq + 1e-12
                #     eta_crit = min(0.8/r, 2/(norm_sharpness * (1 + r)))
                #     T_t_mb = eta_eff * (sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                #     # S_t_mb = eta_eff * (s_sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                #     # K_t_mb = eta_eff * (k_sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                #     K_rs_t_mb1 = eta_eff * (k_rs_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     K_ss_t_mb1 = eta_eff * (k_ss_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     K_rs_t_mb10 = eta_eff * (k_rs_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     K_ss_t_mb10 = eta_eff * (k_ss_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq

                #     S_scv_t_mb10 = eta_eff * (s_scv_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     S_svar_t_mb10 = eta_eff * (s_svar_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rcv_t_mb10 = eta_eff * (s_rcv_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rvar_t_mb10 = eta_eff * (s_rvar_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq

                #     S_scv_t_mb1 = eta_eff * (s_scv_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     S_svar_t_mb1 = eta_eff * (s_svar_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rcv_t_mb1 = eta_eff * (s_rcv_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rvar_t_mb1 = eta_eff * (s_rvar_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq

                #     S_ssqm_t_mb1 = eta_eff * (s_ssqm_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rsqm_t_mb1 = eta_eff * (s_rsqm_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                #     S_ssqm_t_mb10 = eta_eff * (s_ssqm_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     S_rsqm_t_mb10 = eta_eff * (s_rsqm_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                #     # --- SNR-based progress prediction (no scheduling) -----------------------
                #     # pred, pred_real, meanT, thresh, conf = snr_predictor.update(T_t_mb, T_t)
                #     # K_pred, K_pred_real, K_meanT, K_thresh, K_conf = snr_predictor.update(K_t_mb, T_t)
                #     # S_pred, S_pred_real, S_meanT, S_thresh, S_conf = snr_predictor.update(S_t_mb, T_t)
                #     pred, pred_real, meanT, thresh, conf = snr_predictor.update(T_t_mb, T_t)
                #     K_rs_pred1, K_rs_pred_real1, K_rs_meanT1, K_rs_thresh1, K_rs_conf1 = snr_predictor.update(K_rs_t_mb1, T_t)
                #     K_ss_pred1, K_ss_pred_real1, K_ss_meanT1, K_ss_thresh1, K_ss_conf1 = snr_predictor.update(K_ss_t_mb1, T_t)
                #     K_rs_pred10, K_rs_pred_real10, K_rs_meanT10, K_rs_thresh10, K_rs_conf10 = snr_predictor.update(K_rs_t_mb10, T_t)
                #     K_ss_pred10, K_ss_pred_real10, K_ss_meanT10, K_ss_thresh10, K_ss_conf10 = snr_predictor.update(K_ss_t_mb10, T_t)
                #     S_scv_pred10, S_scv_pred_real10, S_scv_meanT10, S_scv_thresh10, S_scv_conf10 = snr_predictor.update(S_scv_t_mb10, T_t)
                #     S_svar_pred10, S_svar_pred_real10, S_svar_meanT10, S_svar_thresh10, S_svar_conf10 = snr_predictor.update(S_svar_t_mb10, T_t)
                #     S_rcv_pred10, S_rcv_pred_real10, S_rcv_meanT10, S_rcv_thresh10, S_rcv_conf10 = snr_predictor.update(S_rcv_t_mb10, T_t)
                #     S_rvar_pred10, S_rvar_pred_real10, S_rvar_meanT10, S_rvar_thresh10, S_rvar_conf10 = snr_predictor.update(S_rvar_t_mb10, T_t)
                #     S_scv_pred1, S_scv_pred_real1, S_scv_meanT1, S_scv_thresh1, S_scv_conf1 = snr_predictor.update(S_scv_t_mb1, T_t)
                #     S_svar_pred1, S_svar_pred_real1, S_svar_meanT1, S_svar_thresh1, S_svar_conf1 = snr_predictor.update(S_svar_t_mb1, T_t)
                #     S_rcv_pred1, S_rcv_pred_real1, S_rcv_meanT1, S_rcv_thresh1, S_rcv_conf1 = snr_predictor.update(S_rcv_t_mb1, T_t)
                #     S_rvar_pred1, S_rvar_pred_real1, S_rvar_meanT1, S_rvar_thresh1, S_rvar_conf1 = snr_predictor.update(S_rvar_t_mb1, T_t)
                #     S_ssqm_pred1, S_ssqm_pred_real1, S_ssqm_meanT1, S_ssqm_thresh1, S_ssqm_conf1 = snr_predictor.update(S_ssqm_t_mb1, T_t)
                #     S_rsqm_pred1, S_rsqm_pred_real1, S_rsqm_meanT1, S_rsqm_thresh1, S_rsqm_conf1 = snr_predictor.update(S_rsqm_t_mb1, T_t)
                #     S_ssqm_pred10, S_ssqm_pred_real10, S_ssqm_meanT10, S_ssqm_thresh10, S_ssqm_conf10 = snr_predictor.update(S_ssqm_t_mb10, T_t)
                #     S_rsqm_pred10, S_rsqm_pred_real10, S_rsqm_meanT10, S_rsqm_thresh10, S_rsqm_conf10 = snr_predictor.update(S_rsqm_t_mb10, T_t)

                #     # Critical effective LR from Mark's 2nd trade-off
                #     # alpha_crit_t = (batch_B * true_grad_norm_sq) / max(sigma2_hat_mb, 1e-12)
                #     # alpha_crit_k = (batch_B * true_grad_norm_sq) / max(k_sigma2_hat_mb, 1e-12)
                #     # alpha_crit_s = (batch_B * true_grad_norm_sq) / max(s_sigma2_hat_mb, 1e-12)
                #     alpha_crit_t = (batch_B * true_grad_norm_sq) / max(sigma2_hat_mb, 1e-12)
                #     alpha_crit_k_rs1 = (batch_B * true_grad_norm_sq) / max(k_rs_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_k_ss1 = (batch_B * true_grad_norm_sq) / max(k_ss_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_k_rs10 = (batch_B * true_grad_norm_sq) / max(k_rs_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_k_ss10 = (batch_B * true_grad_norm_sq) / max(k_ss_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_scv10 = (batch_B * true_grad_norm_sq) / max(s_scv_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_svar10 = (batch_B * true_grad_norm_sq) / max(s_svar_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_rcv10 = (batch_B * true_grad_norm_sq) / max(s_rcv_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_rvar10 = (batch_B * true_grad_norm_sq) / max(s_rvar_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_scv1 = (batch_B * true_grad_norm_sq) / max(s_scv_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_svar1 = (batch_B * true_grad_norm_sq) / max(s_svar_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_rcv1 = (batch_B * true_grad_norm_sq) / max(s_rcv_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_rvar1 = (batch_B * true_grad_norm_sq) / max(s_rvar_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_sqm1 = (batch_B * true_grad_norm_sq) / max(s_ssqm_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_rsqm1 = (batch_B * true_grad_norm_sq) / max(s_rsqm_sigma2_hat_mb1, 1e-12)
                #     alpha_crit_s_sqm10 = (batch_B * true_grad_norm_sq) / max(s_ssqm_sigma2_hat_mb10, 1e-12)
                #     alpha_crit_s_rsqm10 = (batch_B * true_grad_norm_sq) / max(s_rsqm_sigma2_hat_mb10, 1e-12)
                #     alphas = {
                #         "alpha_crit_t": alpha_crit_t,
                #         "alpha_crit_k_rs1": alpha_crit_k_rs1,
                #         "alpha_crit_k_ss1": alpha_crit_k_ss1,
                #         "alpha_crit_k_rs10": alpha_crit_k_rs10,
                #         "alpha_crit_k_ss10": alpha_crit_k_ss10,
                #         "alpha_crit_s_scv10": alpha_crit_s_scv10,
                #         "alpha_crit_s_svar10": alpha_crit_s_svar10,
                #         "alpha_crit_s_rcv10": alpha_crit_s_rcv10,
                #         "alpha_crit_s_rvar10": alpha_crit_s_rvar10,
                #         "alpha_crit_s_scv1": alpha_crit_s_scv1,
                #         "alpha_crit_s_svar1": alpha_crit_s_svar1,
                #         "alpha_crit_s_rcv1": alpha_crit_s_rcv1,
                #         "alpha_crit_s_rvar1": alpha_crit_s_rvar1,
                #         "alpha_crit_s_sqm1": alpha_crit_s_sqm1,
                #         "alpha_crit_s_rsqm1": alpha_crit_s_rsqm1,
                #         "alpha_crit_s_sqm10": alpha_crit_s_sqm10,
                #         "alpha_crit_s_rsqm10": alpha_crit_s_rsqm10,
                #     }

                #     # Prediction: 1 if we are above the critical LR
                #     # snr_sum += pred_real
                #     # k_snr_sum += K_pred_real
                #     # s_snr_sum += S_pred_real
                #     snr_sum += pred_real
                #     k_rs_sum1 += K_rs_pred_real1
                #     k_ss_sum1 += max(K_ss_pred_real1, union_eff_gt_acrit_step_ss1)
                #     k_rs_sum10 += K_rs_pred_real10
                #     k_ss_sum10 += max(K_ss_pred_real10, union_eff_gt_acrit_step_ss10)
                #     s_scv_sum10 += max(S_scv_pred_real10, union_eff_gt_acrit_step_scv10)
                #     s_svar_sum10 += max(S_svar_pred_real10, union_eff_gt_acrit_step_svar10)
                #     s_rcv_sum10 += S_rcv_pred_real10
                #     s_rvar_sum10 += S_rvar_pred_real10
                #     s_scv_sum1 += max(S_scv_pred_real1, union_eff_gt_acrit_step_scv1)
                #     s_svar_sum1 += max(S_svar_pred_real1,union_eff_gt_acrit_step_svar1)
                #     s_rcv_sum1 += S_rcv_pred_real1
                #     s_rvar_sum1 += S_rvar_pred_real1
                #     s_sqm_sum1 += max(S_ssqm_pred_real1, union_eff_gt_acrit_step_sqm1)
                #     s_rsqm_sum1 += S_rsqm_pred_real1
                #     s_sqm_sum10 += max(S_ssqm_pred_real10, union_eff_gt_acrit_step_sqm10)
                #     s_rsqm_sum10 += S_rsqm_pred_real10

                #     n_crit = min(0.8/r, 2/(norm_sharpness * (1 + r)))

                # mark = {
                #         # "snr_T":            float(T_t) if T_t is not None else float("nan"),
                #         # "snr_sigma2_hat":   float(sigma2_hat) if sigma2_hat is not None else float("nan"),
                #         "mb_sigma2_hat":    float(sigma2_hat_mb),
                #         "mb_snr_T":         float(T_t_mb),
                #         # "S_t_mb":           float(S_t_mb),
                # "sharpness":        sharpness,
                #         # "rho":              effective_lr * sharpness / 2,
                #         # "rho_norm":      effective_lr * norm_sharpness / 2,
                #         "n_crit_nois":  0.8/r,
                #         "n_crit_curv":  2/(norm_sharpness * (1 + r)),
                #         "n_crit":       min(0.8/r, 2/(norm_sharpness * (1 + r))),
                #         "lbo":          2/(sharpness+1e-12),
                #         "lbo_norm":     2/(norm_sharpness+1e-12),
                #         # "K_t_mb":           float(K_t_mb),
                #         **alphas,  # critical LRs
                #         # "alpha_crit_t":     float(alpha_crit_t),
                #         # "alpha_crit_k":     float(alpha_crit_k),
                #         # "alpha_crit_s":     float(alpha_crit_s),

                #         # "snr_pred":         int(pred) if pred is not None else -1,
                #         # "snr_pred_real":    int(pred_real) if pred_real is not None else -1,
                #         # "snr_pred_conf":    float(conf) if conf is not None else float("nan"),
                #         # "snr_T_mean":       float(meanT) if meanT is not None else float("nan"),
                #         # "snr_T_thresh":     float(thresh) if thresh is not None else float("nan"),
                #         # "snr_K_real":       int(K_pred_real) if K_pred_real is not None else -1,
                #         # "snr_S_real":       int(S_pred_real) if S_pred_real is not None else -1,
                # }
                #     # ------------------------------------------------------------------------
                #     if ly_sched is not None:
                #         if config.param == "sqm10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_sqm10, total_updates, total_steps)
                #         elif config.param == "sqm1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_sqm1, total_updates, total_steps)
                #         elif config.param == "svar10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_svar10, total_updates, total_steps)
                #         elif config.param == "svar1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_svar1, total_updates, total_steps)
                #         elif config.param == "ss10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_ss10, total_updates, total_steps)
                #         elif config.param == "ss1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_ss1, total_updates, total_steps)
                #         elif config.param == "scv10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_scv10, total_updates, total_steps)
                #         elif config.param == "scv1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_scv1, total_updates, total_steps)
                #         elif config.param == "rsqm10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rsqm10, total_updates, total_steps)
                #         elif config.param == "rsqm1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rsqm1, total_updates, total_steps)
                #         elif config.param == "rcv10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rcv10, total_updates, total_steps)
                #         elif config.param == "rvar10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rvar10, total_updates, total_steps)
                #         elif config.param == "rcv1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rcv1, total_updates, total_steps)
                #         elif config.param == "rvar1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rvar1, total_updates, total_steps)
                #         elif config.param == "rs10":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_rs10, total_updates, total_steps)
                #         elif config.param == "rs1":
                #             lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_rs1, total_updates, total_steps)
                #         elif config.param == "s_tau":
                #             lr_star, _ = ly_sched.step(effective_lr, sharp_log["tau"], total_updates, total_steps)
                #         elif config.param == "r_tau":
                #             lr_star, _ = ly_sched.step(effective_lr, r_sharp_log["tau"], total_updates, total_steps)
                #         log_extra  = {"ly_lr_star": lr_star}
                #     else:
                #         log_extra  = {}
                optimizer.step()

                # betas scheduling
                if config.optimizer == "adam":
                    optimizer.param_groups[0]["betas"] = optimizers.get_betas(
                        config, epoch
                    )
                # lr schedule step
                if config.lr_schedule != "constant" and scheduler is not None:
                    scheduler.step()
                # if ly_sched is not None:
                #     lr_star, _ = ly_sched.step(effective_lr, lambda_log["tau"])
                #     log_extra  = {"ly_lr_star": lr_star}
                # else:
                #     log_extra  = {}

            # shrink perturb
            if config.reg == "shrink_perturb":
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.mul_(1.0 - config.sp_weight_decay)
                        p.data.add_(config.sp_noise_std * torch.randn_like(p.data))

            delta = torch.cat(
                [(p.data - o).view(-1).abs() for p, o in zip(params, old)]
            )
            update_norm = delta.mean().item()
            sum_up += update_norm
            total_updates += 1

            if total_updates % config.log_interval == 0:
                # use_vals = { f"use_{name}": optimizers.compute_use_for_activation(h) for name, h in activations.items() }
                # avg_use_val = sum(use_vals.values()) / len(use_vals)
                # gg, x = 0., 0.
                # for n, p in model.named_parameters():
                #     if p.requires_grad and p.ndim >= 2:
                #         x += 1
                #         gg += optimizers.power_iteration(p, 1).pow(config.spectral_k)
                # gg = gg / x
                wn = torch.cat([p.data.view(-1).abs() for p in params]).mean().item()
                # use_vals = {
                #     f"use_{name}": compute_use_for_activation(h)
                #     for name, h in activations.items()
                # }
                # avg_use_val = sum(use_vals.values()) / len(use_vals)
                log = {
                    "acc": acc,
                    "loss": loss.item(),
                    "update_norm": update_norm,
                    "weight_norm": wn,
                    "ratio": update_norm / (wn + 1e-12),
                    # "average_use_val": avg_use_val,
                    "hessian_rank": hessian_rank,
                    # "sharpness": sharpness,
                    # "preconditioned_sharpness": precond,
                    # "task_lam_var": avg_lam_var,
                    "lr": optimizer.param_groups[0]["lr"],
                    # "avg_use_val": avg_use_val,
                    # "true_grad_norm_sq": true_grad_norm_sq,
                    # "effective_lr": effective_lr,
                    # **init_sigma_min,
                    # "max_singular": gg,
                    # **sharp_log,                                   # sharpness stats
                    # **{f"lam_{k}": v for k, v in lambda_log.items()},
                    # **log_extra,               # only when ly_sched is active
                    **mark,
                    # "gradient_noise": noise_power_full,
                    # "gradient_noise_mb": noise_power_mb,
                    # "beta2": optimizer.param_groups[0]["betas"][1],
                    # **use_vals,
                    # **hess_avgs,
                }
                # if config.model not in ["CNN","BatchNormCNN"]:
                #     h = activations["l1"]
                #     use1 = compute_use_for_activation(h); log["use_l1"]=use1
                run.log(log)
        if scheduler is not None:
            scheduler.step()

    model.eval()
    eval_loader = data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False
    )
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in eval_loader:
            if config.model in ["CNN", "BatchNormCNN"]:
                inp = x.to(device)
            else:
                inp = x.view(x.size(0), -1).to(device)
            out = model(inp)
            l = criterion(out, y.to(device))
            bs = y.size(0)
            total += l.item() * bs
            count += bs
    J = total / count
    pn = (
        torch.cat(
            [p.data.view(-1).abs() for p in model.parameters() if p.requires_grad]
        )
        .mean()
        .item()
    )
    aun = sum_up / total_updates
    inputs, _ = next(iter(eval_loader))
    inputs = inputs.view(inputs.size(0), -1).to(device)
    h = model(inputs)
    s = torch.linalg.svdvals(h)
    cut = s.sum() * 0.99
    j = (torch.cumsum(s, 0) >= cut).nonzero()[0].item() + 1
    effective_rank = -j / float(h.shape[1])
    steps = config.epochs * len(loader) / config.log_interval
    run.log(
        {
            "J": J,
            "param_norm": pn,
            "avg_norm_sharp": this_normalized_sharp / (config.epochs * len(loader)),
            "average_update_norm": aun,
            "effective_rank": effective_rank,
            "task_acc": this_task_acc / (config.epochs * len(loader)),
            # "snr_pct": 1 - (snr_sum / steps),
            # "k_rs_pct1": 1 - (k_rs_sum1 / steps),
            # "k_ss_pct1": 1 - (k_ss_sum1 / steps),
            # "k_rs_pct10": 1 - (k_rs_sum10 / steps),
            # "k_ss_pct10": 1 - (k_ss_sum10 / steps),
            # "s_scv_pct10": 1 - (s_scv_sum10 / steps),
            # "s_svar_pct10": 1 - (s_svar_sum10 / steps),
            # "s_rcv_pct10": 1 - (s_rcv_sum10 / steps),
            # "s_rvar_pct10": 1 - (s_rvar_sum10 / steps),
            # "s_scv_pct1": 1 - (s_scv_sum1 / steps),
            # "s_svar_pct1": 1 - (s_svar_sum1 / steps),
            # "s_rcv_pct1": 1 - (s_rcv_sum1 / steps),
            # "s_rvar_pct1": 1 - (s_rvar_sum1 / steps),
            # "s_sqm_pct1": 1 - (s_sqm_sum1 / steps),
            # "s_rsqm_pct1": 1 - (s_rsqm_sum1 / steps),
            # "s_sqm_pct10": 1 - (s_sqm_sum10 / steps),
            # "s_rsqm_pct10": 1 - (s_rsqm_sum10 / steps),
            # # "k_snr_pct": 1 - (k_snr_sum / steps),
            # # "s_snr_pct": 1 - (s_snr_sum / steps),
            # "ly_snr_pct": 1 - (ly_snr_sum / steps),
            # "ly_snr_pct2": 1 - (ly_snr_2_sum / steps),
            # "ly_snr_pct_union": 1 - (ly_union_sum / steps),
            # "eff_acrit_scv1_pct_union": 1 - (eff_acrit_scv1_union_sum / steps),
            # "eff_acrit_ss1_pct_union": 1 - (eff_acrit_ss1_union_sum / steps),
            # "eff_acrit_svar1_pct_union": 1 - (eff_acrit_svar1_union_sum / steps),
            # "eff_acrit_scv10_pct_union": 1 - (eff_acrit_scv10_union_sum / steps),
            # "eff_acrit_ss10_pct_union": 1 - (eff_acrit_ss10_union_sum / steps),
            # "eff_acrit_svar10_pct_union": 1 - (eff_acrit_svar10_union_sum / steps),
            # "eff_acrit_ssqm1_pct_union": 1 - (eff_acrit_ssqm1_union_sum / steps),
            # "eff_acrit_ssqm10_pct_union": 1 - (eff_acrit_ssqm10_union_sum / steps),
            # "eff_acrit_rcv1_pct_union": 1 - (eff_acrit_rcv1_union_sum / steps),
            # "eff_acrit_rs1_pct_union": 1 - (eff_acrit_rs1_union_sum / steps),
            # "eff_acrit_rvar1_pct_union": 1 - (eff_acrit_rvar1_union_sum / steps),
            # "eff_acrit_rsqm1_pct_union": 1 - (eff_acrit_rsqm1_union_sum / steps),
            # "eff_acrit_rcv10_pct_union": 1 - (eff_acrit_rcv10_union_sum / steps),
            # "eff_acrit_rs10_pct_union": 1 - (eff_acrit_rs10_union_sum / steps),
            # "eff_acrit_rvar10_pct_union": 1 - (eff_acrit_rvar10_union_sum / steps),
            # "eff_acrit_rsqm10_pct_union": 1 - (eff_acrit_rsqm10_union_sum / steps),
        }
    )
    res = results[config.activation]
    res["batch_error"].append(J)
    res["param_norm"].append(pn)
    res["update_norm"].append(aun)

wandb.finish()

tasks = np.arange(1, len(results[config.activation]["batch_error"]) + 1)
for m in ["batch_error", "param_norm", "update_norm"]:
    plt.figure()
    plt.plot(tasks, results[config.activation][m], label=config.activation)
    plt.xlabel("Task")
    plt.ylabel(m)
    plt.legend()
    plt.show()
