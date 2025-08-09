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

parser = get_parser()
config = parser.parse_args()
LENGTH_CHOICES = [100, 300, 50, 150]

rho = config.sam_rho
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
noise_power_full   = 0.0
noise_power_mb     = 0.0

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
set_seed(config.seed)

task_lengths = [random.choice(LENGTH_CHOICES) for _ in range(config.runs)]

train_dataset, test_dataset, in_ch, input_size, DATA_MEAN, DATA_STD = data_loader.get_dataset(config)

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

layer_map = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        layer = name.split('.')[0]          # "fc1.weight" -> "fc1"
        layer_map.setdefault(layer, []).append(p)

layer_groups = [
    {'params': params, 'lr': config.lr, 'layer': layer}   # keep default wd, betas...
    for layer, params in layer_map.items()
]

# ── one EMAState per layer ─────────────────────────────────────────────
layer_states = {
    layer : misc.EMAState(alphas=(0.01, 0.05, 0.5))   # same alphas you used globally
    for layer in layer_map
}

criterion = nn.CrossEntropyLoss(reduction="mean")
wd = config.l2_lambda if config.reg == "l2" else 0.0
if config.optimizer == "adam":
    optimizer = optim.Adam(layer_groups, lr=config.lr, weight_decay=wd, betas=(0.9, config.beta2))
elif config.optimizer == "sgd": 
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=wd)

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
    project=f"rand_label_{config.dataset}",
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
wandb.define_metric("gradient_noise",   hidden=False)
wandb.define_metric("gradient_noise_mb",hidden=False)
wandb.define_metric("true_grad_norm_sq",hidden=False)
wandb.define_metric("task_lam_var",     hidden=False)

results = {
    config.activation: {
        "batch_error": [],
        "param_norm": [],
        "update_norm": [],
        "effective_rank": [],
        "dormancy": [],
    }
}

sharp_state  = misc.EMAState(alphas=(0.01, 0.05, 0.5))
lambda_state = misc.EMAState(alphas=(0.01, 0.05, 0.5))

init_sigma_min = {}
print_val = 0.
for name, p in model.named_parameters():
    if p.requires_grad and p.ndim >= 2:
        # 1–3 inverse-power iterations are plenty at t=0
        s0 = optimizers.power_iteration_sigma_min(p.detach(), iters=3).item()
        init_sigma_min[name] = s0
        print_val += s0
        print(s0)
print(print_val)

task_lengths = [300, 50, 300, 300, 100, 300, 50, 150, 300, 100, 300, 50, 150, 300, 100, 300, 50, 150, 300, 100]
for task in range(config.runs):
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
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=wd, betas=(config.beta1, config.beta2))
    if config.optimizer == "adam" and config.reset_optimizer:
        optimizer.state.clear()

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
        # if task == config.runs:
        #     train_dataset = optimizers.randomize_targets(train_dataset, 1.0)
        # -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
        loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )

    total_updates = 0
    if total_updates % config.log_interval == 0:
        hessian_rank = optimizers.empirical_fischer_rank(model, train_dataset, device, cfg=config)

    model.train()
    ly_sched = None
    scheduler = None
    if config.lr_schedule == "linear":
        total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
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
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.gamma
        )
    elif config.lr_schedule == "polynomial":
        total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / total_steps) ** config.power,
        )
    elif config.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs
        )
    elif config.lr_schedule == "wsd":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.wsd_lambda)
    elif config.lr_schedule == "power":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.power_lambda)
    elif config.lr_schedule == "skew":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.skew_lambda)
    elif config.lr_schedule == "lyapunov":
        # hyper-params exposed to the CLI for convenience
        ly_safety = config.ly_safety
        ly_cool   = config.ly_cool
        ly_warm   = config.ly_warm
        ly_sched  = optimizers.LyapunovScheduler(optimizer,
                                    ema_state = sharp_state,   # <<– share!
                                    safety    = ly_safety,
                                    cool      = ly_cool,
                                    warm      = ly_warm)
        scheduler = None
    elif config.lr_schedule == "pl_lyapunov":
        pl_scheduler = PerLayerLyapunovScheduler(
            optimizer      = optimizer,
            layer_states   = layer_states,
            safety         = config.pl_lyap_safety,
            cool           = config.pl_lyap_cool,
            warm           = config.pl_lyap_warm,
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
                        reg += (optimizers.power_iteration(p, 1).pow(config.spectral_k) - 1.0).pow(2)
                reg *= config.spectral_lambda
            elif config.reg == "ortho":
                frac = config.ortho_frac
                for name, p in model.named_parameters():
                    if p.ndim < 2 or not p.requires_grad:
                        continue

                    if total_updates % config.ortho_interval == 0:
                        sigma_now = optimizers.power_iteration_sigma_min(p, iters=1).detach()
                        cached_sigma_min[name] = sigma_now
                    else:
                        sigma_now = cached_sigma_min.get(
                            name,
                            optimizers.power_iteration_sigma_min(p, iters=1).detach()
                        )

                    target = frac * init_sigma_min[name]
                    reg   += (sigma_now - target).pow(2)
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
                layer_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                for layer, params in layer_map.items():
                    # (a) top eigen-value of this layer’s Hessian block
                    lam = optimizers.estimate_hessian_topk(model, loss, params, k=1)[0]

                    # (b) normalize the sharpness exactly like you do globally
                    norm_lam = optimizers.get_norm_sharpness(optimizer, lam, config)

                    # (c) update EMA statistics and grab scalars
                    eff_lr = layer_eff_lrs.get(layer, optimizer.param_groups[0]["lr"])
                    state, scalars   = misc.update_stat(norm_lam, layer_states[layer], eff_lr)

                    if config.lr_schedule == "pl_lyapunov":
                        lr_star = pl_scheduler.step(layer, eff_lr, scalars["tau"])

                    # (d) log everything with a nice prefix
                    wandb.log({
                        f"{layer}/sharp" : norm_lam,
                        f"{layer}/mu"    : scalars["lam_mean"],
                        f"{layer}/tau"   : scalars["tau"],
                        f"{layer}/cv"    : scalars["lam_cv"],
                        f"{layer}/eff_lr": eff_lr,
                        f"{layer}/predict": scalars["collapse_pred"],
                        "reg":              reg
                    })

                eigs = optimizers.estimate_hessian_topk(model, loss, params, k=1)
                sharpness = eigs[0]
                lambda_min = optimizers.estimate_hessian_min_eig(model, loss, params, iters=20)

                norm_sharpness = optimizers.get_norm_sharpness(optimizer, sharpness, config)
                lambda_min_norm = optimizers.get_norm_sharpness(optimizer, lambda_min, config)
                this_normalized_sharp += norm_sharpness 

                effective_lr = optimizers.compute_adam_effective_lr(optimizer)

                sharp_state,  sharp_log  = misc.update_stat(norm_sharpness,  sharp_state,  effective_lr)
                lambda_state, lambda_log = misc.update_stat(lambda_min_norm, lambda_state, effective_lr) 
                if ly_sched is not None:
                    lr_star, _ = ly_sched.step(effective_lr, lambda_log["tau"])
                    log_extra  = {"ly_lr_star": lr_star}
                else:
                    log_extra  = {}

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
                if total_updates % config.log_interval == 0:
                    params = [p for p in model.parameters() if p.requires_grad]
                    grad_flat = torch.cat([p.grad.view(-1) for p in params])
                    true_grad_norm_sq = (grad_flat.norm()**2).item()
                    var_grad = grad_flat.var(unbiased=False).item()
                    noise_power_full = (config.lr**2) * var_grad
                    noise_power_mb = (config.lr**2) * var_grad / config.batch_size
                optimizer.step()
                
                # betas scheduling
                optimizer.param_groups[0]['betas'] = optimizers.get_betas(config, epoch)
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
                gg, x = 0., 0.
                for name, p in model.named_parameters():
                    if p.requires_grad and p.dim() >= 2:
                        # 1–3 inverse-power iterations are plenty at t=0
                        s0 = optimizers.power_iteration_sigma_min(p.detach(), iters=1).item()
                        gg += s0
                # print(print_val)
                gg, x = 0., 0.
                for n, p in model.named_parameters():
                    if p.requires_grad and p.ndim >= 2:
                        x += 1
                        gg += optimizers.power_iteration(p, 1).pow(config.spectral_k)
                gg = gg / x
                wn = torch.cat([p.data.view(-1).abs() for p in params]).mean().item()
                log = {
                    "acc": acc,
                    "loss": loss.item(),
                    "update_norm": update_norm,
                    "weight_norm": wn,
                    "ratio": update_norm / (wn + 1e-12),
                    # "average_use_val": avg_use_val,
                    "hessian_rank": hessian_rank,
                    "sharpness": sharpness,
                    # "preconditioned_sharpness": precond,
                    # "task_lam_var": avg_lam_var,
                    "lr": optimizer.param_groups[0]["lr"],
                    "true_grad_norm_sq": var_grad,
                    "effective_lr": effective_lr,
                    **init_sigma_min,
                    "max_singular": gg,
                    **sharp_log,                                   # sharpness stats
                    # **{f"lam_{k}": v for k, v in lambda_log.items()},
                    **log_extra,               # only when ly_sched is active
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
    run.log(
        {
            "J": J,
            "param_norm": pn,
            "avg_norm_sharp": this_normalized_sharp / (config.epochs * len(loader)),
            "average_update_norm": aun,
            "effective_rank": effective_rank,
            "task_acc": this_task_acc / (config.epochs * len(loader)),
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
