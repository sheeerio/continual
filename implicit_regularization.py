import os
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from config import get_parser
from utils import misc, schedulers, optimizers
from models import mlp, cnn
from datasets import data_loader
from utils.optimizers import PerLayerLyapunovScheduler

def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def register_activation_hooks(model: nn.Module, model_name: str, store: Dict[str, torch.Tensor]) -> None:
    def save_activations(name):
        def hook(_m, _inp, out):
            store[name] = out
        return hook

    if model_name in ["MLP", "LayerNormMLP", "BatchNormMLP", "LeakyLayerNormMLP", "LeakyKaimingLayerNormMLP"]:
        model.fc1.register_forward_hook(save_activations("l1"))
        model.fc2.register_forward_hook(save_activations("l2"))
    else:
        if hasattr(model, "fc1"):
            model.fc1.register_forward_hook(save_activations("l1"))

def make_layer_groups(model: nn.Module, base_lr: float):
    layer_map: Dict[str, List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            layer = name.split('.')[0]
            layer_map.setdefault(layer, []).append(p)
    layer_groups = [{'params': params, 'lr': base_lr, 'layer': layer}
                    for layer, params in layer_map.items()]
    return layer_map, layer_groups

def build_model(config, input_size: int, in_ch: int) -> nn.Module:
    hidden = 256
    if config.model == "MLP":
        return mlp.MLP(input_size, hidden, 10)
    if config.model == "BatchNormMLP":
        return mlp.BatchNormMLP(input_size, hidden, 10)
    if config.model == "LinearNet":
        return nn.Sequential(nn.Flatten(), nn.Identity())
    if config.model == "CNN":
        return cnn.CNN(in_ch)
    if config.model == "BatchNormCNN":
        return cnn.BatchNormCNN(in_ch)
    return mlp.MLP(input_size, hidden, 10)

def build_optimizer(config, layer_groups, weight_decay: float):
    if config.optimizer == "adam":
        return optim.Adam(layer_groups, lr=config.lr, weight_decay=weight_decay, betas=(0.9, config.beta2))
    if config.optimizer == "sgd":
        return optim.SGD(layer_groups, lr=config.lr, weight_decay=weight_decay, momentum=0.9)
    if config.optimizer == "clamped_adam":
        return optimizers.ClampedAdam(layer_groups, lr=config.lr, lr_min=1e-3, lr_max=1.1)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")

def build_scheduler(config, optimizer, sharp_state, layer_states, act_layer_states, layer_map):
    total_steps = None 
    if config.lr_schedule in {"linear","step","exponential","polynomial","cosine","wsd","power","skew"}:
        return ("pytorch", None, None)
    if config.lr_schedule == "lyapunov":
        ly_sched = optimizers.LyapunovScheduler(
            optimizer,
            ema_state=sharp_state,
            safety=config.safety, cool=config.cool, warm=config.warm,
            cfg=config
        )
        return (None, ly_sched, None)
    if config.lr_schedule == "pl_lyapunov":
        pl_sched = PerLayerLyapunovScheduler(
            optimizer=optimizer,
            layer_states={layer: layer_states.get(layer, misc.EMAState(alphas=(0.01,0.05,0.5)))
                          for layer in layer_map},
            safety=config.safety, cool=config.cool, warm=config.warm, cfg=config
        )
        return (None, None, pl_sched)
    return (None, None, None)

class Regularizer:
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)

class L2InitReg(Regularizer):
    def __init__(self, config, init_params): self.cfg, self.init = config, init_params
    def __call__(self, model: nn.Module, device) -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                reg += (p - self.init[n]).pow(2).sum()
        return self.cfg.l2_lambda * reg

class WassersteinReg(Regularizer):
    def __init__(self, config, init_params): self.cfg, self.init = config, init_params
    def __call__(self, model: nn.Module, device) -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                reg += (torch.sort(p.view(-1))[0] - self.init[n]).pow(2).sum()
        return self.cfg.wass_lambda * reg

class SpectralReg(Regularizer):
    def __init__(self, config): self.cfg = config
    def __call__(self, model: nn.Module, device) -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        for _n, p in model.named_parameters():
            if p.requires_grad and p.ndim >= 2:
                reg += (optimizers.power_iteration(p, 1).pow(self.cfg.spectral_k) - 1.0).pow(2)
        return self.cfg.spectral_lambda * reg

class OrthoSigmaMinReg(Regularizer):
    def __init__(self, config, init_sigma_min, cached_sigma_min, total_updates_ref):
        self.cfg = config
        self.init_sigma_min = init_sigma_min
        self.cached = cached_sigma_min
        self.total_updates_ref = total_updates_ref
    def __call__(self, model: nn.Module, device) -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        frac = self.cfg.ortho_frac
        for name, p in model.named_parameters():
            if p.ndim < 2 or not p.requires_grad:
                continue
            if self.total_updates_ref() % self.cfg.ortho_interval == 0:
                sigma_now = optimizers.power_iteration_sigma_min(p, iters=1).detach()
                self.cached[name] = sigma_now
            else:
                sigma_now = self.cached.get(name, optimizers.power_iteration_sigma_min(p, iters=1).detach())
            target = frac * self.init_sigma_min[name]
            reg   += (sigma_now - target).pow(2)
        return self.cfg.ortho_lambda * reg

class OrthoFrobReg(Regularizer):
    def __init__(self, config): self.cfg = config
    def __call__(self, model: nn.Module, device) -> torch.Tensor:
        reg = torch.tensor(0.0, device=device)
        for _name, p in model.named_parameters():
            if p.ndim >= 2 and p.requires_grad:
                W = p.view(p.shape[0], -1)
                k = W.shape[1]
                I = torch.eye(k, device=W.device, dtype=W.dtype)
                reg += (W.t() @ W - I).pow(2).sum()
        return self.cfg.ortho_lambda * reg

def make_regularizer(config, init_params, init_sigma_min, cached_sigma_min, total_updates_ref):
    if config.reg == "l2_init": return L2InitReg(config, init_params)
    if config.reg == "wass":    return WassersteinReg(config, init_params)
    if config.reg == "spectral":return SpectralReg(config)
    if config.reg == "ortho":   return OrthoSigmaMinReg(config, init_sigma_min, cached_sigma_min, total_updates_ref)
    if config.reg == "orthofrob": return OrthoFrobReg(config)
    return Regularizer()

@dataclass
class TrainingState:
    device: torch.device
    criterion: nn.Module
    criterion_nored: nn.Module
    sharp_state: misc.EMAState = field(default_factory=lambda: misc.EMAState(alphas=(0.01,0.05,0.5)))
    r_sharp_state: misc.EMAState = field(default_factory=lambda: misc.EMAState(alphas=(0.01,0.05,0.5)))
    layer_states: Dict[str, misc.EMAState] = field(default_factory=dict)
    act_layer_states: Dict[str, misc.EMAState] = field(default_factory=dict)
    snr_tracker: optimizers.GradSNR = field(default_factory=optimizers.GradSNR)
    snr_predictor: optimizers.SNRProgressPredictor = None
    total_updates: int = 0

class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.activations: Dict[str, torch.Tensor] = {}

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_nored = nn.CrossEntropyLoss(reduction="none")

        self.train_dataset, self.test_dataset, self.in_ch, self.input_size, self.DATA_MEAN, self.DATA_STD = data_loader.get_dataset(config)

        self.model = build_model(config, self.input_size, self.in_ch).to(self.device)
        register_activation_hooks(self.model, config.model, self.activations)
        self.layer_map, self.layer_groups = make_layer_groups(self.model, config.lr)
        self.layer_states = {layer: misc.EMAState(alphas=(0.01,0.05,0.5)) for layer in self.layer_map}
        self.act_layer_states = {layer: misc.EMAState(alphas=(0.01,0.05,0.5)) for layer in self.layer_map}

        wd = config.l2_lambda if config.reg == "l2" else 0.0
        self.optimizer = build_optimizer(config, self.layer_groups, wd)
        self.sharp_state = misc.EMAState(alphas=(0.01,0.05,0.5))
        self.r_sharp_state = misc.EMAState(alphas=(0.01,0.05,0.5))
        sch_kind, self.ly_sched, self.pl_sched = build_scheduler(
            config, self.optimizer, self.sharp_state, self.layer_states, self.act_layer_states, self.layer_map
        )
        self.pt_scheduler = None
        self.sch_kind = sch_kind 

        self.init_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        if config.reg == "wass":
            for n, p0 in self.init_params.items():
                self.init_params[n] = torch.sort(p0.view(-1))[0].to(self.device)
        self.init_sigma_min = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.ndim >= 2:
                    self.init_sigma_min[name] = optimizers.power_iteration_sigma_min(p.detach(), iters=3).item()
        self.cached_sigma_min = {}

        total_updates_ref = lambda: self.state.total_updates
        self.regularizer = make_regularizer(config, self.init_params, self.init_sigma_min, self.cached_sigma_min, total_updates_ref)

        self.state = TrainingState(
            device=self.device,
            criterion=self.criterion,
            criterion_nored=self.criterion_nored,
            snr_predictor=optimizers.SNRProgressPredictor(margin=config.snr_margin, window=config.snr_pred_window),
            layer_states=self.layer_states,
            act_layer_states=self.act_layer_states
        )

        self.run = wandb.init(
            project=f"workshop_{config.dataset}",
            entity="sheerio",
            group=config.exp_name,
            name=config.name,
            config=vars(config),
        )
        for m in ["gradient_noise","gradient_noise_mb","true_grad_norm_sq","task_lam_var",
                  "snr_T","snr_sigma2_hat","mb_sigma2_hat","mb_snr_T",
                  "snr_pred","snr_pred_conf","snr_T_mean","snr_T_thresh"]:
            wandb.define_metric(m, hidden=False if "mb" in m or "snr" in m else True)

    def _reset_task_accumulators(self):
        for n in [
            "snr_sum","k_rs_sum1","k_ss_sum1","k_rs_sum10","k_ss_sum10",
            "s_scv_sum10","s_svar_sum10","s_rcv_sum10","s_rvar_sum10",
            "s_scv_sum1","s_svar_sum1","s_rcv_sum1","s_rvar_sum1",
            "s_sqm_sum1","s_rsqm_sum1","s_sqm_sum10","s_rsqm_sum10",
            "ly_snr_sum","ly_snr_2_sum","ly_union_sum",
            "eff_acrit_scv1_union_sum","eff_acrit_ss1_union_sum","eff_acrit_svar1_union_sum","eff_acrit_ssqm1_union_sum",
            "eff_acrit_scv10_union_sum","eff_acrit_ss10_union_sum","eff_acrit_svar10_union_sum","eff_acrit_ssqm10_union_sum",
            "eff_acrit_rcv1_union_sum","eff_acrit_rs1_union_sum","eff_acrit_rvar1_union_sum","eff_acrit_rsqm1_union_sum",
            "eff_acrit_rcv10_union_sum","eff_acrit_rs10_union_sum","eff_acrit_rvar10_union_sum","eff_acrit_rsqm10_union_sum",
        ]:
            setattr(self, n, 0.0)

    def _build_loader_for_task(self, task_idx: int):
        train_ds = optimizers.randomize_targets(self.train_dataset, self.cfg.ns)
        loader = data.DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=4)
        return train_ds, loader
    
    def _metrics_schedules_and_logging(self, inputs, labels, loss, epoch: int, total_steps: int):
        def safe_div(a, b): return a / (b + 1e-12)
        def batch_B(): return int(inputs.size(0))
        def init_acc(name): 
            if not hasattr(self, name): setattr(self, name, 0.0)

        for n in [
            "snr_sum","k_rs_sum1","k_ss_sum1","k_rs_sum10","k_ss_sum10",
            "s_scv_sum10","s_svar_sum10","s_rcv_sum10","s_rvar_sum10",
            "s_scv_sum1","s_svar_sum1","s_rcv_sum1","s_rvar_sum1",
            "s_sqm_sum1","s_rsqm_sum1","s_sqm_sum10","s_rsqm_sum10",
            "ly_snr_sum","ly_snr_2_sum","ly_union_sum",
            "eff_acrit_scv1_union_sum","eff_acrit_ss1_union_sum","eff_acrit_svar1_union_sum","eff_acrit_ssqm1_union_sum",
            "eff_acrit_scv10_union_sum","eff_acrit_ss10_union_sum","eff_acrit_svar10_union_sum","eff_acrit_ssqm10_union_sum",
            "eff_acrit_rcv1_union_sum","eff_acrit_rs1_union_sum","eff_acrit_rvar1_union_sum","eff_acrit_rsqm1_union_sum",
            "eff_acrit_rcv10_union_sum","eff_acrit_rs10_union_sum","eff_acrit_rvar10_union_sum","eff_acrit_rsqm10_union_sum",
            "this_task_acc","this_normalized_sharp"
        ]: init_acc(n)

        if self.cfg.optimizer == "adam":
            layer_eff_lrs = optimizers.per_layer_effective_lr(self.model, self.optimizer)
        else:
            layer_eff_lrs = optimizers.per_layer_sgd_lr(self.model, self.optimizer, step=self.state.total_updates, sgd_mode="time")

        layer_sigma2_mb = optimizers.grad_variance_within_batch_by_layer(
            self.model, self.criterion_nored, inputs, labels, self.layer_map
        )

        def sigma2_mix_specs(norm_lam, sc, act_sc, lam_val, g2, sig2_l):
            return {
                ("s_scv", 1):  sig2_l + 10.0 * g2 * sc["lam_cv"],
                ("s_ss",  1):  sig2_l + 10.0 * g2 * norm_lam,
                ("s_svar",1):  sig2_l + 10.0 * g2 * sc["lam_var"],
                ("s_sqm", 1):  sig2_l + 75.0 * g2 * sc["sq_mean"],
                ("s_scv",10):  sig2_l + 20.0 * g2 * sc["lam_cv"],
                ("s_ss", 10):  sig2_l + 20.0 * g2 * norm_lam,
                ("s_svar",10): sig2_l + 20.0 * g2 * sc["lam_var"],
                ("s_sqm",10):  sig2_l + 20.0 * g2 * sc["sq_mean"],
                ("r_rcv", 1):  sig2_l + 10.0 * g2 * act_sc["lam_cv"],
                ("r_rs",  1):  sig2_l + 10.0 * g2 * lam_val,
                ("r_rvar",1):  sig2_l + 10.0 * g2 * act_sc["lam_var"],
                ("r_rsqm",1):  sig2_l + 75.0 * g2 * act_sc["sq_mean"],
                ("r_rcv",10):  sig2_l + 20.0 * g2 * act_sc["lam_cv"],
                ("r_rs", 10):  sig2_l + 20.0 * g2 * lam_val,
                ("r_rvar",10): sig2_l + 20.0 * g2 * act_sc["lam_var"],
                ("r_rsqm",10): sig2_l + 20.0 * g2 * act_sc["sq_mean"],
            }

        union = {k:0 for k in [
            "pred","eff_gt_scv1","eff_gt_ss1","eff_gt_svar1","eff_gt_sqm1",
            "eff_gt_scv10","eff_gt_ss10","eff_gt_svar10","eff_gt_sqm10",
            "eff_gt_rcv1","eff_gt_rs1","eff_gt_rvar1","eff_gt_rsqm1",
            "eff_gt_rcv10","eff_gt_rs10","eff_gt_rvar10","eff_gt_rsqm10"
        ]}

        for layer, params in self.layer_map.items():
            lam = optimizers.estimate_hessian_topk(self.model, loss, params, k=1)[0]
            norm_lam = optimizers.get_norm_sharpness(self.optimizer, lam, self.cfg)
            eff_lr = layer_eff_lrs.get(layer, self.optimizer.param_groups[0]["lr"])

            _, sc      = misc.update_stat(norm_lam, self.layer_states[layer], eff_lr)
            _, act_sc  = misc.update_stat(lam,       self.act_layer_states[layer], eff_lr)

            gi = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            gi = [g for g in gi if g is not None]
            if len(gi) == 0:
                g2 = 0.0
            else:
                g2 = float(torch.cat([g.contiguous().view(-1) for g in gi]).pow(2).sum().item())
            sig2_l = float(layer_sigma2_mb.get(layer, 0.0))
            B = batch_B()

            mixes = sigma2_mix_specs(norm_lam, sc, act_sc, lam, g2, sig2_l)

            alpha = {}
            for (tag, mult), sig2 in mixes.items():
                alpha[(tag, mult)] = (B * g2) / max(sig2, 1e-12)

            def over(k, tag, mult):
                ok = int(eff_lr > alpha[(tag, mult)])
                union[k] = max(union[k], ok)

            over("eff_gt_scv1",  "s_scv", 1);  over("eff_gt_ss1",   "s_ss", 1)
            over("eff_gt_svar1", "s_svar",1);  over("eff_gt_sqm1",  "s_sqm",1)
            over("eff_gt_scv10", "s_scv",10);  over("eff_gt_ss10",  "s_ss",10)
            over("eff_gt_svar10","s_svar",10); over("eff_gt_sqm10","s_sqm",10)
            over("eff_gt_rcv1",  "r_rcv", 1);  over("eff_gt_rs1",   "r_rs", 1)
            over("eff_gt_rvar1", "r_rvar",1);  over("eff_gt_rsqm1","r_rsqm",1)
            over("eff_gt_rcv10", "r_rcv",10);  over("eff_gt_rs10",  "r_rs",10)
            over("eff_gt_rvar10","r_rvar",10); over("eff_gt_rsqm10","r_rsqm",10)
            union["pred"] = max(union["pred"], int(sc["collapse_pred2"]))

            if self.pl_sched is not None:
                pmap = {
                    "sqm10":("s_sqm",10),"sqm1":("s_sqm",1),"scv10":("s_scv",10),"scv1":("s_scv",1),
                    "svar10":("s_svar",10),"svar1":("s_svar",1),"ss10":("s_ss",10),"ss1":("s_ss",1),
                    "rcv1":("r_rcv",1),"rs1":("r_rs",1),"rvar1":("r_rvar",1),"rqm1":("r_rsqm",1),
                    "rcv10":("r_rcv",10),"rs10":("r_rs",10),"rvar10":("r_rvar",10),"rqm10":("r_rsqm",10),
                    "s_tau":("s_tau",1),"r_tau":("r_tau",1),
                }
                if self.cfg.param in ("s_tau","r_tau"):
                    target = sc["tau"] if self.cfg.param=="s_tau" else act_sc["tau"]
                else:
                    t, m = pmap[self.cfg.param]
                    target = alpha[(t, m)]
                self.pl_sched.step(layer, eff_lr, target, self.state.total_updates, total_steps)

            wandb.log({
                f"{layer}/sharp": float(norm_lam),
                f"{layer}/mu": float(sc["lam_mean"]),
                f"{layer}/tau": float(sc["tau"]),
                f"{layer}/cv": float(sc["lam_cv"]),
                f"{layer}/eff_lr": float(eff_lr),
                f"{layer}/predict": int(sc["collapse_pred2"]),
                f"{layer}/alpha_crit_scv1":  float(alpha[("s_scv",1)]),
                f"{layer}/alpha_crit_ss1":   float(alpha[("s_ss",1)]),
                f"{layer}/alpha_crit_svar1": float(alpha[("s_svar",1)]),
                f"{layer}/alpha_crit_sqm1":  float(alpha[("s_sqm",1)]),
                f"{layer}/alpha_crit_scv10":  float(alpha[("s_scv",10)]),
                f"{layer}/alpha_crit_ss10":   float(alpha[("s_ss",10)]),
                f"{layer}/alpha_crit_svar10": float(alpha[("s_svar",10)]),
                f"{layer}/alpha_crit_sqm10":  float(alpha[("s_sqm",10)]),
                f"{layer}/alpha_crit_rcv1":  float(alpha[("r_rcv",1)]),
                f"{layer}/alpha_crit_rs1":   float(alpha[("r_rs",1)]),
                f"{layer}/alpha_crit_rvar1": float(alpha[("r_rvar",1)]),
                f"{layer}/alpha_crit_rsqm1": float(alpha[("r_rsqm",1)]),
                f"{layer}/alpha_crit_rcv10":  float(alpha[("r_rcv",10)]),
                f"{layer}/alpha_crit_rs10":   float(alpha[("r_rs",10)]),
                f"{layer}/alpha_crit_rvar10": float(alpha[("r_rvar",10)]),
                f"{layer}/alpha_crit_rsqm10": float(alpha[("r_rsqm",10)]),
                "reg": float(loss.detach().item())
            })

        params_all = [p for p in self.model.parameters() if p.requires_grad]
        sharp = optimizers.estimate_hessian_topk(self.model, loss, params_all, k=1)[0]
        norm_sharp = optimizers.get_norm_sharpness(self.optimizer, sharp, self.cfg)
        eff_lr = optimizers.compute_effective_lr(self.optimizer, cfg=self.cfg, step=self.state.total_updates, sgd_mode="time")

        self.sharp_state,  sharp_stats   = misc.update_stat(norm_sharp, self.sharp_state, eff_lr)
        self.r_sharp_state, rsharp_stats = misc.update_stat(sharp,      self.r_sharp_state, eff_lr)

        self.ly_snr_sum   += sharp_stats["collapse_pred"]
        self.ly_snr_2_sum += sharp_stats["collapse_pred2"]
        self.this_normalized_sharp += norm_sharp

        grads_all = [p.grad for p in params_all if p.grad is not None]
        if len(grads_all) == 0:
            grad_flat = torch.zeros(1, device=self.device)
            g2_true = 0.0
        else:
            grad_flat = torch.cat([g.view(-1) for g in grads_all])
            g2_true = float(grad_flat.pow(2).sum().item())

        sigma2_hat_mb = optimizers.grad_variance_within_batch(self.model, self.criterion_nored, inputs, labels)
        B = int(inputs.size(0))
        r = (sigma2_hat_mb / max(1, B)) / (g2_true + 1e-12) + 1e-12
        T_t, _ = self.state.snr_tracker.update(grad_flat.detach(), eff_lr, B)
        T_t_mb = eff_lr * (sigma2_hat_mb / max(1, B)) / (g2_true + 1e-12)

        def mixK(coef, val): return sigma2_hat_mb + coef * g2_true * val
        mixes_global = {
            "K_rs_1":  mixK(10.0, sharp),       "K_ss_1":  mixK(10.0, norm_sharp),
            "K_rs_10": mixK(20.0, sharp),       "K_ss_10": mixK(20.0, norm_sharp),
            "S_scv_10":  mixK(20.0, sharp_stats["lam_cv"]),    "S_svar_10": mixK(20.0, sharp_stats["lam_var"]),
            "S_rcv_10":  mixK(20.0, rsharp_stats["lam_cv"]),   "S_rvar_10": mixK(20.0, rsharp_stats["lam_var"]),
            "S_scv_1":   mixK(10.0, sharp_stats["lam_cv"]),    "S_svar_1":  mixK(10.0, sharp_stats["lam_var"]),
            "S_rcv_1":   mixK(10.0, rsharp_stats["lam_cv"]),   "S_rvar_1":  mixK(10.0, rsharp_stats["lam_var"]),
            "S_ssqm_10": mixK(20.0, sharp_stats["sq_mean"]),   "S_rsqm_10": mixK(20.0, rsharp_stats["sq_mean"]),
            "S_ssqm_1":  mixK(75.0, sharp_stats["sq_mean"]),   "S_rsqm_1":  mixK(10.0, rsharp_stats["sq_mean"]),
        }
        ratios = {k: eff_lr * (v / max(1, B)) / (g2_true + 1e-12) for k, v in mixes_global.items()}

        def upd(x): return self.state.snr_predictor.update(x, T_t)[1]
        K_rs_pred_real1  = upd(ratios["K_rs_1"])
        K_ss_pred_real1  = upd(ratios["K_ss_1"])
        K_rs_pred_real10 = upd(ratios["K_rs_10"])
        K_ss_pred_real10 = upd(ratios["K_ss_10"])
        S_scv_pred_real10  = upd(ratios["S_scv_10"])
        S_svar_pred_real10 = upd(ratios["S_svar_10"])
        S_rcv_pred_real10  = upd(ratios["S_rcv_10"])
        S_rvar_pred_real10 = upd(ratios["S_rvar_10"])
        S_scv_pred_real1   = upd(ratios["S_scv_1"])
        S_svar_pred_real1  = upd(ratios["S_svar_1"])
        S_rcv_pred_real1   = upd(ratios["S_rcv_1"])
        S_rvar_pred_real1  = upd(ratios["S_rvar_1"])
        S_ssqm_pred_real1  = upd(ratios["S_ssqm_1"])
        S_rsqm_pred_real1  = upd(ratios["S_rsqm_1"])
        S_ssqm_pred_real10 = upd(ratios["S_ssqm_10"])
        S_rsqm_pred_real10 = upd(ratios["S_rsqm_10"])
        _, pred_real, *_   = self.state.snr_predictor.update(T_t_mb, T_t)

        alpha_crit_t = (B * g2_true) / max(sigma2_hat_mb, 1e-12)
        getA = lambda key: (B * g2_true) / max(mixes_global[key], 1e-12)
        alpha = {
            "k_rs1":  getA("K_rs_1"),   "k_ss1":  getA("K_ss_1"),
            "k_rs10": getA("K_rs_10"),  "k_ss10": getA("K_ss_10"),
            "s_scv10": getA("S_scv_10"), "s_svar10": getA("S_svar_10"),
            "s_rcv10": getA("S_rcv_10"), "s_rvar10": getA("S_rvar_10"),
            "s_scv1":  getA("S_scv_1"),  "s_svar1":  getA("S_svar_1"),
            "s_rcv1":  getA("S_rcv_1"),  "s_rvar1":  getA("S_rvar_1"),
            "s_sqm1":  getA("S_ssqm_1"), "s_rsqm1":  getA("S_rsqm_1"),
            "s_sqm10": getA("S_ssqm_10"),"s_rsqm10": getA("S_rsqm_10"),
        }

        self.snr_sum     += pred_real
        self.k_rs_sum1   += K_rs_pred_real1
        self.k_ss_sum1   += max(K_ss_pred_real1,  union["eff_gt_ss1"])
        self.k_rs_sum10  += K_rs_pred_real10
        self.k_ss_sum10  += max(K_ss_pred_real10, union["eff_gt_ss10"])
        self.s_scv_sum10 += max(S_scv_pred_real10,  union["eff_gt_scv10"])
        self.s_svar_sum10+= max(S_svar_pred_real10, union["eff_gt_svar10"])
        self.s_rcv_sum10 += S_rcv_pred_real10
        self.s_rvar_sum10+= S_rvar_pred_real10
        self.s_scv_sum1  += max(S_scv_pred_real1,  union["eff_gt_scv1"])
        self.s_svar_sum1 += max(S_svar_pred_real1, union["eff_gt_svar1"])
        self.s_rcv_sum1  += S_rcv_pred_real1
        self.s_rvar_sum1 += S_rvar_pred_real1
        self.s_sqm_sum1  += max(S_ssqm_pred_real1,  union["eff_gt_sqm1"])
        self.s_rsqm_sum1 += S_rsqm_pred_real1
        self.s_sqm_sum10 += max(S_ssqm_pred_real10, union["eff_gt_sqm10"])
        self.s_rsqm_sum10+= S_rsqm_pred_real10

        self.ly_union_sum += union["pred"]
        self.eff_acrit_scv1_union_sum  += union["eff_gt_scv1"]
        self.eff_acrit_ss1_union_sum   += union["eff_gt_ss1"]
        self.eff_acrit_svar1_union_sum += union["eff_gt_svar1"]
        self.eff_acrit_ssqm1_union_sum += union["eff_gt_sqm1"]
        self.eff_acrit_scv10_union_sum  += union["eff_gt_scv10"]
        self.eff_acrit_ss10_union_sum   += union["eff_gt_ss10"]
        self.eff_acrit_svar10_union_sum += union["eff_gt_svar10"]
        self.eff_acrit_ssqm10_union_sum += union["eff_gt_sqm10"]
        self.eff_acrit_rcv1_union_sum  += union["eff_gt_rcv1"]
        self.eff_acrit_rs1_union_sum   += union["eff_gt_rs1"]
        self.eff_acrit_rvar1_union_sum += union["eff_gt_rvar1"]
        self.eff_acrit_rsqm1_union_sum += union["eff_gt_rsqm1"]
        self.eff_acrit_rcv10_union_sum  += union["eff_gt_rcv10"]
        self.eff_acrit_rs10_union_sum   += union["eff_gt_rs10"]
        self.eff_acrit_rvar10_union_sum += union["eff_gt_rvar10"]
        self.eff_acrit_rsqm10_union_sum += union["eff_gt_rsqm10"]

        log_extra = {}
        if self.ly_sched is not None:
            pmap = {
                "sqm10":"s_sqm10","sqm1":"s_sqm1","svar10":"s_svar10","svar1":"s_svar1",
                "ss10":"k_ss10","ss1":"k_ss1","scv10":"s_scv10","scv1":"s_scv1",
                "rqm10":"s_rsqm10","rqm1":"s_rsqm1","rcv10":"s_rcv10","rvar10":"s_rvar10",
                "rcv1":"s_rcv1","rvar1":"s_rvar1","rs10":"k_rs10","rs1":"k_rs1"
            }
            if self.cfg.param == "s_tau":
                target = sharp_stats["tau"]
            elif self.cfg.param == "r_tau":
                target = rsharp_stats["tau"]
            else:
                target = alpha[pmap[self.cfg.param]]
            lr_star, _ = self.ly_sched.step(eff_lr, target, self.state.total_updates, total_steps)
            log_extra["ly_lr_star"] = lr_star

        acc = float((self.model(inputs).argmax(dim=1) == labels).float().mean().item())
        self.this_task_acc += acc

        wandb.log({
            "acc": acc,
            "loss": float(loss.item()),
            "sharpness": float(sharp),
            "effective_lr": float(eff_lr),
            "true_grad_norm_sq": float(g2_true),
            "mb_sigma2_hat": float(sigma2_hat_mb),
            "mb_snr_T": float(T_t_mb),
            "n_crit_nois": 0.8 / r,
            "n_crit_curv": 2 / (norm_sharp * (1 + r)),
            "n_crit": min(0.8 / r, 2 / (norm_sharp * (1 + r))),
            "lbo": 2 / (sharp + 1e-12),
            "lbo_norm": 2 / (norm_sharp + 1e-12),
            "alpha_crit_t": float(alpha_crit_t),
            "alpha_crit_k_rs1": float(alpha["k_rs1"]),
            "alpha_crit_k_ss1": float(alpha["k_ss1"]),
            "alpha_crit_k_rs10": float(alpha["k_rs10"]),
            "alpha_crit_k_ss10": float(alpha["k_ss10"]),
            "alpha_crit_s_scv10": float(alpha["s_scv10"]),
            "alpha_crit_s_svar10": float(alpha["s_svar10"]),
            "alpha_crit_s_rcv10": float(alpha["s_rcv10"]),
            "alpha_crit_s_rvar10": float(alpha["s_rvar10"]),
            "alpha_crit_s_scv1": float(alpha["s_scv1"]),
            "alpha_crit_s_svar1": float(alpha["s_svar1"]),
            "alpha_crit_s_rcv1": float(alpha["s_rcv1"]),
            "alpha_crit_s_rvar1": float(alpha["s_rvar1"]),
            "alpha_crit_s_sqm1": float(alpha["s_sqm1"]),
            "alpha_crit_s_rsqm1": float(alpha["s_rsqm1"]),
            "alpha_crit_s_sqm10": float(alpha["s_sqm10"]),
            "alpha_crit_s_rsqm10": float(alpha["s_rsqm10"]),
            **sharp_stats,   
            **log_extra,
        })


    def _maybe_build_pt_scheduler(self, total_steps: int):
        if self.sch_kind != "pytorch":
            return
        c = self.cfg
        if c.lr_schedule == "linear":
            initial_lr, final_lr = c.lr, c.final_lr
            decay_range = initial_lr - final_lr
            self.pt_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: max(0.0, (initial_lr - (decay_range * step / total_steps)) / initial_lr)
            )
        elif c.lr_schedule == "step":
            self.pt_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=c.step_size, gamma=c.gamma)
        elif c.lr_schedule == "exponential":
            self.pt_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=c.gamma)
        elif c.lr_schedule == "polynomial":
            self.pt_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: (1 - step / total_steps) ** c.power)
        elif c.lr_schedule == "cosine":
            self.pt_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=c.epochs)
        elif c.lr_schedule == "wsd":
            self.pt_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=schedulers.wsd_lambda)
        elif c.lr_schedule == "power":
            self.pt_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=schedulers.power_lambda)
        elif c.lr_schedule == "skew":
            self.pt_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=schedulers.skew_lambda)

    def train(self):
        LENGTH_CHOICES = [100, 300, 50, 150]
        task_lengths = [random.choice(LENGTH_CHOICES) for _ in range(self.cfg.runs)] if self.cfg.random_length else [self.cfg.epochs]*self.cfg.runs

        for task in range(self.cfg.runs):
            self._reset_task_accumulators()

            epochs = task_lengths[task]
            train_ds, loader = self._build_loader_for_task(task)
            total_steps = epochs * math.ceil(len(train_ds) / self.cfg.batch_size)
            self._maybe_build_pt_scheduler(total_steps)

            this_task_acc = 0.0
            this_normalized_sharp = 0.0
            sum_update_norm = 0.0

            self.model.train()
            for epoch in range(epochs):
                for x, y in loader:
                    if self.cfg.model in ["CNN", "BatchNormCNN"]:
                        inputs = x.to(self.device)
                    else:
                        inputs = x.view(x.size(0), -1).to(self.device)
                    labels = y.to(self.device)

                    self.optimizer.zero_grad()
                    out = self.model(inputs)
                    preds = out.argmax(dim=1)
                    acc = (preds == labels).float().mean().item()
                    this_task_acc += acc

                    base = self.criterion(out, labels)
                    reg  = self.regularizer(self.model, self.device)
                    loss = base + reg

                    if self.cfg.sam:
                        params = [p for p in self.model.parameters() if p.requires_grad]
                        loss.backward(create_graph=True)
                        grads = torch.autograd.grad(loss, params, create_graph=True)
                        grad_flat = torch.cat([g.view(-1) for g in grads]); grad_norm = grad_flat.norm() + 1e-12
                        epsilons = [(self.cfg.sam_rho / grad_norm) * g for g in grads]
                        for p, e in zip(params, epsilons): p.data.add_(e)
                        out_adv = self.model(inputs); loss_adv = self.criterion(out_adv, labels) + reg
                        self.optimizer.zero_grad(); loss_adv.backward()
                        for p, e in zip(params, epsilons): p.data.sub_(e)
                    else:
                        loss.backward(retain_graph=True)

                    if self.state.total_updates % self.cfg.log_interval == 0:
                        self._metrics_schedules_and_logging(inputs, labels, loss, epoch, total_steps)

                    self.optimizer.step()

                    if self.cfg.optimizer == "adam":
                        self.optimizer.param_groups[0]['betas'] = optimizers.get_betas(self.cfg, epoch)

                    if self.pt_scheduler is not None:
                        self.pt_scheduler.step()

                    with torch.no_grad():
                        params = [p for p in self.model.parameters() if p.requires_grad]
                        wn = torch.cat([p.data.view(-1).abs() for p in params]).mean().item()
                        update_norm = torch.cat([p.grad.view(-1).abs() for p in params]).mean().item() if all(p.grad is not None for p in params) else 0.0
                        sum_update_norm += update_norm

                    self.state.total_updates += 1

            self.model.eval()
            eval_loader = data.DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=False)
            total, count = 0.0, 0
            with torch.no_grad():
                for x, y in eval_loader:
                    inp = x.to(self.device) if self.cfg.model in ["CNN","BatchNormCNN"] else x.view(x.size(0), -1).to(self.device)
                    out = self.model(inp); l = self.criterion(out, y.to(self.device))
                    bs = y.size(0); total += l.item() * bs; count += bs
            J = total / count

            pn = torch.cat([p.data.view(-1).abs() for p in self.model.parameters() if p.requires_grad]).mean().item()
            aun = sum_update_norm / max(1, (epochs * len(loader)))

            inputs, _ = next(iter(eval_loader))
            inputs = inputs.view(inputs.size(0), -1).to(self.device) if self.cfg.model not in ["CNN","BatchNormCNN"] else inputs.to(self.device)
            h = self.model(inputs)
            s = torch.linalg.svdvals(h)
            cut = s.sum() * 0.99
            j = (torch.cumsum(s, 0) >= cut).nonzero()[0].item() + 1
            effective_rank = -j / float(h.shape[1])

            steps = (epochs * len(loader) / self.cfg.log_interval) if self.cfg.log_interval else 1
            wandb.log({
                "J": J,
                "param_norm": pn,
                "average_update_norm": aun,
                "effective_rank": effective_rank,
                "task_acc": this_task_acc / max(1, (epochs * len(loader))),
                "snr_pct": 1 - (self.snr_sum / steps),
                "k_rs_pct1": 1 - (self.k_rs_sum1 / steps),
                "k_ss_pct1": 1 - (self.k_ss_sum1 / steps),
                "k_rs_pct10": 1 - (self.k_rs_sum10 / steps),
                "k_ss_pct10": 1 - (self.k_ss_sum10 / steps),
                "s_scv_pct10": 1 - (self.s_scv_sum10 / steps),
                "s_svar_pct10": 1 - (self.s_svar_sum10 / steps),
                "s_rcv_pct10": 1 - (self.s_rcv_sum10 / steps),
                "s_rvar_pct10": 1 - (self.s_rvar_sum10 / steps),
                "s_scv_pct1": 1 - (self.s_scv_sum1 / steps),
                "s_svar_pct1": 1 - (self.s_svar_sum1 / steps),
                "s_rcv_pct1": 1 - (self.s_rcv_sum1 / steps),
                "s_rvar_pct1": 1 - (self.s_rvar_sum1 / steps),
                "s_sqm_pct1": 1 - (self.s_sqm_sum1 / steps),
                "s_rsqm_pct1": 1 - (self.s_rsqm_sum1 / steps),
                "s_sqm_pct10": 1 - (self.s_sqm_sum10 / steps),
                "s_rsqm_pct10": 1 - (self.s_rsqm_sum10 / steps),
                "ly_snr_pct": 1 - (self.ly_snr_sum / steps),
                "ly_snr_pct2": 1 - (self.ly_snr_2_sum / steps),
                "ly_snr_pct_union": 1 - (self.ly_union_sum / steps),
                "eff_acrit_scv1_pct_union": 1 - (self.eff_acrit_scv1_union_sum / steps),
                "eff_acrit_ss1_pct_union": 1 - (self.eff_acrit_ss1_union_sum / steps),
                "eff_acrit_svar1_pct_union": 1 - (self.eff_acrit_svar1_union_sum / steps),
                "eff_acrit_scv10_pct_union": 1 - (self.eff_acrit_scv10_union_sum / steps),
                "eff_acrit_ss10_pct_union": 1 - (self.eff_acrit_ss10_union_sum / steps),
                "eff_acrit_svar10_pct_union": 1 - (self.eff_acrit_svar10_union_sum / steps),
                "eff_acrit_ssqm1_pct_union": 1 - (self.eff_acrit_ssqm1_union_sum / steps),
                "eff_acrit_ssqm10_pct_union": 1 - (self.eff_acrit_ssqm10_union_sum / steps),
                "eff_acrit_rcv1_pct_union": 1 - (self.eff_acrit_rcv1_union_sum / steps),
                "eff_acrit_rs1_pct_union": 1 - (self.eff_acrit_rs1_union_sum / steps),
                "eff_acrit_rvar1_pct_union": 1 - (self.eff_acrit_rvar1_union_sum / steps),
                "eff_acrit_rsqm1_pct_union": 1 - (self.eff_acrit_rsqm1_union_sum / steps),
                "eff_acrit_rcv10_pct_union": 1 - (self.eff_acrit_rcv10_union_sum / steps),
                "eff_acrit_rs10_pct_union": 1 - (self.eff_acrit_rs10_union_sum / steps),
                "eff_acrit_rvar10_pct_union": 1 - (self.eff_acrit_rvar10_union_sum / steps),
                "eff_acrit_rsqm10_pct_union": 1 - (self.eff_acrit_rsqm10_union_sum / steps),
            })

        wandb.finish()

def main():
    parser = get_parser()
    config = parser.parse_args()
    if not hasattr(config, "snr_margin"):      config.snr_margin = 0.0
    if not hasattr(config, "snr_pred_window"): config.snr_pred_window = 20
    if getattr(config, "activation", None) == "leaky_relu":
        config.alpha = 0.01

    set_seed(config.seed)
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()