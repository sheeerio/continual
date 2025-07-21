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

W = 20  # Window size for sharpness tracking
K = 20
TRACE_INTERVAL = 200

parser = get_parser()
config = parser.parse_args()

rho = config.sam_rho
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
noise_power_full   = 0.0
noise_power_mb     = 0.0

misc.set_seed(config.seed)

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
    model.fc3.register_forward_hook(save_activations("l3"))
else:
    model.fc1.register_forward_hook(save_activations("l1"))

criterion = nn.CrossEntropyLoss(reduction="mean")
wd = config.l2_lambda if config.reg == "l2" else 0.0
if config.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=wd, betas=(0.9, config.beta2))
elif config.optimizer == "sgd": 
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=wd)

#
run = wandb.init(
    project=f"random_label_{config.dataset}",
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
wandb.define_metric("task_lam_std",     hidden=False)

results = {
    config.activation: {
        "batch_error": [],
        "param_norm": [],
        "update_norm": [],
        "effective_rank": [],
        "dormancy": [],
    }
}
for task in range(config.runs+1):
    if config.reset_model:
        model = mlp.MLP(input_size, hidden, 10).to(device)
        if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=wd, betas=(config.beta1, config.beta2))
    if config.optimizer == "adam" and config.reset_optimizer:
        optimizer.state.clear()

    sharp_queue = deque(maxlen=W)
    avg_sharp_queue = deque(maxlen=config.epochs)
    current_runlen = 0
    step_within_task = 0
    collapse_step_within_task = None

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
        hessian_rank = optimizers.empirical_fischer_rank(model, train_dataset, device)

    model.train()
    
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
    else:
        scheduler = None

    sum_up = 0.0
    this_task_acc = 0.0
    this_normalized_sharp = 0.0
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
            
            loss = base + reg
            params = [p for p in model.parameters() if p.requires_grad]
            old = [p.data.clone() for p in params]

            nus = [
                optimizer.state[p]["exp_avg_sq"]
                for p in optimizer.param_groups[0]["params"]
                if "exp_avg_sq" in optimizer.state[p]
            ]

            # if total_updates % TRACE_INTERVAL == 0:
            #     trace_val = optimizers.hessian_trace(loss, params, n_samples=10)

            if total_updates % config.log_interval == 0:
                # nus = [
                # optimizer.state[p]["exp_avg_sq"]
                # for p in optimizer.param_groups[0]["params"]
                # if "exp_avg_sq" in optimizer.state[p]
                # ]
                # precond = ( optimizers.preconditioned_sharpness(loss, params, nus, epsilon=optimizer.param_groups[0].get("eps",1e-8)) 
                #             if len(nus) else None )
                eigs = optimizers.estimate_hessian_topk(model, loss, params, k=1)
                sharpness = eigs[0]
                # eig_ratio = eigs[0] / (eigs[4] + 1e-12)

                norm_sharpness = optimizers.get_norm_sharpness(optimizer, sharpness)
                this_normalized_sharp += norm_sharpness

                sharp_queue.append(norm_sharpness)
                avg_sharp_queue.append(norm_sharpness)

                lam_mean = sum(sharp_queue) / len(sharp_queue)
                avg_lam_mean = sum(avg_sharp_queue) / len(avg_sharp_queue)
                lam_std = (
                    sum((x - lam_mean) ** 2 for x in sharp_queue) / len(sharp_queue)
                ) ** 0.5 if len(sharp_queue) == W else 0.0
                avg_lam_std = (
                    sum((x - avg_lam_mean) ** 2 for x in avg_sharp_queue)
                    / len(avg_sharp_queue)
                )


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
                
                optimizer.step()
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
                if config.optimizer == "adam":
                    end = 2 * config.epochs
                    if epoch < end:
                        beta1 = config.beta1 + (0.99 - config.beta1) * (epoch / (end))
                        beta2 = config.beta2 + (0.75 - config.beta2) * (epoch / (end))
                    else:
                        beta1 = 0.99
                        beta2 = 0.75
                    if config.beta_schedule:
                        optimizer.param_groups[0]['betas'] = (beta1, beta2)
                if config.lr_schedule != "constant":
                    scheduler.step()
            

            # shrink perturb
            if config.reg == "shrink_perturb":
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.mul_(1.0 - config.sp_weight_decay)
                        p.data.add_(config.sp_noise_std * torch.randn_like(p.data))

            delta = torch.cat(
                [(p.data - o).view(-1).abs() for p, o in zip(params, old)]
            )
            up_norm = delta.mean().item()
            sum_up += up_norm
            total_updates += 1
            if total_updates % config.log_interval == 0:
                # use_vals = { f"use_{name}": optimizers.compute_use_for_activation(h) for name, h in activations.items() }
                # avg_use_val = sum(use_vals.values()) / len(use_vals)
                wn = torch.cat([p.data.view(-1).abs() for p in params]).mean().item()
                log = {
                    "acc": acc,
                    "loss": loss.item(),
                    "update_norm": up_norm,
                    "weight_norm": wn,
                    "ratio": up_norm / (wn + 1e-12),
                    # "average_use_val": avg_use_val,
                    "hessian_rank": hessian_rank,
                    "sharpness": sharpness,
                    # "preconditioned_sharpness": precond,
                    # "norm_sharpness": norm_sharpness,
                    "lam_mean": lam_mean,
                    "lam_std": lam_std,
                    "lam_cv": lam_std / (lam_mean + 1e-12),
                    "task_lam_std": avg_lam_std,
                    "lr": optimizer.param_groups[0]["lr"],
                    "true_grad_norm_sq": var_grad,
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
