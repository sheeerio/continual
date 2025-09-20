import torch
import random
import numpy as np
from torch.utils.data import Subset
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data
from collections import deque
from typing import Deque, Dict, Tuple, List, Optional
from utils.misc import EMAState
import torch.optim as optim
from torch.optim import Adam

# def power_iteration_sigma_min(W: torch.Tensor, iters: int = 3, eps: float = 1e-6):
#     WT_W = W.T @ W 
#     v = torch.randn(WT_W.shape[0], device=W.device)
#     v = v / v.norm()
#     I = torch.eye(WT_W.shape[0], device=W.device)
#     for _ in range(iters):
#         v = torch.linalg.solve(WT_W + eps * I, v)
#         v = v / (v.norm() + 1e-12)
#     sigma_min_sq = torch.dot(v, WT_W @ v)
#     return torch.sqrt(sigma_min_sq + 1e-12)
def power_iteration_sigma_min(W: torch.Tensor,
                               iters: int = 3,
                               shift_mult: float = 1e-3) -> torch.Tensor:
    m, n = W.shape
    use_WtW = (n <= m)
    A = W.T @ W if use_WtW else W @ W.T 

    diag_mean = A.diagonal().mean()
    ridge     = shift_mult * diag_mean + 1e-6 
    A_shift   = A + ridge * torch.eye(A.size(0), device=W.device, dtype=W.dtype)

    v = torch.randn(A_shift.shape[0], device=W.device, dtype=W.dtype)
    v.div_(v.norm() + 1e-12)

    for _ in range(iters):
        x = torch.linalg.lstsq(A_shift, v.unsqueeze(-1)).solution.squeeze(-1)
        v = x / (x.norm() + 1e-12)

    sigma_min_sq = torch.dot(v, A @ v)
    return torch.sqrt(sigma_min_sq.clamp(min=0.0))


def randomize_targets(dataset, p):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        if isinstance(original_dataset.targets, torch.Tensor):
            targets_list = original_dataset.targets.tolist()
        else:
            targets_list = original_dataset.targets
        n = len(subset_indices)
        k = int(p * n)
        random_subset_idx = torch.randperm(n)[:k]
        for i in random_subset_idx:
            original_idx = subset_indices[i]
            targets_list[original_idx] = random.randint(0, 9)

        if isinstance(original_dataset.targets, torch.Tensor):
            original_dataset.targets = torch.tensor(targets_list)
        else:
            original_dataset.targets = targets_list

        return dataset
    else:
        if isinstance(dataset.targets, torch.Tensor):
            targets_list = dataset.targets.tolist()
        else:
            targets_list = dataset.targets

        n = len(targets_list)
        k = int(p * n)
        idx = torch.randperm(n)[:k]
        for i in idx:
            targets_list[i] = random.randint(0, 9)

        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(targets_list)
        else:
            dataset.targets = targets_list
        return dataset

def preconditioned_sharpness(loss, params, nu, epsilon=1e-8, iters=20):
    nu_flat = torch.cat([n.detach().view(-1) for n in nu])
    P_inv_diag = 1.0 / (torch.sqrt(nu_flat) + epsilon)
    dim = P_inv_diag.numel()

    v = torch.randn(dim, device=loss.device)
    v /= v.norm()

    for _ in range(iters):
        u = P_inv_diag * v
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
        Hu = torch.autograd.grad(grad_flat.dot(u), params, retain_graph=True)
        Hu_flat = torch.cat([h.contiguous().view(-1) for h in Hu]).detach()

        w = P_inv_diag * Hu_flat
        v = w / (w.norm() + 1e-12)

    u = P_inv_diag * v
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    Hu = torch.autograd.grad(grad_flat.dot(u), params, retain_graph=False)
    Hu_flat = torch.cat([h.contiguous().view(-1) for h in Hu]).detach()
    w = P_inv_diag * Hu_flat

    return v.dot(w).item()

def empirical_fischer_rank(model, dataset, device, thresh=0.99, max_m=100, cfg=None):
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    m = 0
    for x, y in loader:
        if m >= max_m:
            break
        if cfg.model in ["CNN", "BatchNormCNN"]:
            x = x.to(device)
        else:
            x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        model.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss(reduction='mean')(output, y)
        loss.backward(retain_graph=True)
        g = torch.cat([p.grad.view(-1) for p in params])
        grads.append(g)
        m += 1
    G = torch.stack(grads)
    M = G @ G.T
    sig = torch.linalg.svdvals(M)
    cumsum = torch.cumsum(sig, dim=0)
    total = cumsum[-1]
    j = (cumsum / total >= thresh).nonzero()[0].item() + 1 if total !=0 else 0
    return j / float(m)


def estimate_hessian_topk(model, loss, params, k=1, iters=100):
    grads = torch.autograd.grad(loss, params, create_graph=True)  # retain
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
    n = flat_grad.numel()

    def hvp(v):
        g_v = (flat_grad * v).sum()
        hv = torch.autograd.grad(g_v, params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hv]).detach()

    eigs = []
    vs = []
    for _ in range(k):
        v = torch.randn(n, device=flat_grad.device)
        v = v / (v.norm() + 1e-12)
        
        for _ in range(iters):
            w = hvp(v)
            for j, u in enumerate(vs):
                w = w - eigs[j] * (u.dot(w)) * u
            v = w / (w.norm() + 1e-12)
        
        Hv = hvp(v)
        lam = v.dot(Hv).item()
        eigs.append(lam)
        vs.append(v)
    
    return eigs

def compute_use_for_activation(h):
    with torch.no_grad():
        pos = (h > 0).float().mean(dim=0)
        eps = 1e-12
        ent = -(pos * (pos + eps).log() + (1 - pos) * (1 - pos + eps).log())
        return ent.mean().item()


def power_iteration(W, iters=1):
    v = torch.randn(W.shape[1], device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W.t() @ (W @ v)
        v = v / (v.norm() + 1e-12)
    return (W @ v).norm()


def hessian_trace(loss, params, n_samples=1):
    trace_est = 0.0
    for _ in range(n_samples):
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        v = torch.randn(flat_grad.size(), device=flat_grad.device)
        Hv = torch.autograd.grad(flat_grad.dot(v), params, retain_graph=True)
        Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])
        trace_est += (flat_grad * Hv_flat).sum().item()
    return trace_est / n_samples

def get_norm_sharpness(optimizer, sharpness, cfg):
    v_squares = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if "exp_avg_sq" in state:
                v_sq = state["exp_avg_sq"].detach()
                v_squares.append(v_sq.view(-1))
    if len(v_squares) > 0:
        v_cat = torch.cat(v_squares)
        rms = torch.sqrt(v_cat.mean() + 1e-16)
        lr0 = optimizer.param_groups[0]["lr"]
        alpha_agg = lr0 / (rms + optimizer.param_groups[0]["eps"])
    else:
        alpha_agg = optimizer.param_groups[0]["lr"]

    if cfg.optimizer == "adam":
        norm_sharpness = sharpness * alpha_agg
    else:
        norm_sharpness = sharpness

    return norm_sharpness

def get_betas(cfg, epoch):
    if cfg.optimizer == "adam":
        end = 2 * cfg.epochs
        if epoch < end:
            beta1 = cfg.beta1 + (0.99 - cfg.beta1) * (epoch / (end))
            beta2 = cfg.beta2 + (0.75 - cfg.beta2) * (epoch / (end))
        else:
            beta1 = 0.99
            beta2 = 0.75
        return (beta1, beta2)
    else:
        raise ValueError("optimizer is not adam, cannot get betas.")


class LyapunovScheduler:
    def __init__(self,
                opt,
                ema_state: EMAState,
                safety: float = 0.9,
                cool:   float = 0.999,
                warm:   float = 1.00001,
                cfg =  None):
        self.opt      = opt
        self.state    = ema_state    
        self.safety   = cfg.safety
        self.cool     = cool
        self.warm     = warm

    def step(self, effective_lr: float, tau: float, current_step: int, total_steps: int) -> Tuple[float, float]:
        if not self.state.ema_variation or self.state.ema_variation == 0.0:
            return float('inf'), effective_lr

        lr_star = tau
        if effective_lr > 0.12 and effective_lr > self.safety * lr_star:
            for g in self.opt.param_groups:
                g['lr'] *= self.cool
        elif current_step < total_steps * 0.1 and effective_lr < 0.3 * self.safety * lr_star:
            for g in self.opt.param_groups:
                g['lr'] *= self.warm

        return lr_star, effective_lr

def estimate_hessian_min_eig(model, loss, params, iters=100):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    d = g_flat.numel()

    def hvp_minus(v):
        g_v = (g_flat * v).sum()
        Hv  = torch.autograd.grad(g_v, params, retain_graph=True)
        return -torch.cat([h.contiguous().view(-1) for h in Hv]).detach()  # -H v

    v = torch.randn(d, device=g_flat.device);  v /= v.norm() + 1e-12
    for _ in range(iters):
        w = hvp_minus(v)
        v = w / (w.norm() + 1e-12)

    Hv  = hvp_minus(v) 
    lam = v.dot(Hv).item()  
    return -lam

import math
from typing import Optional

def compute_effective_lr(optimizer: torch.optim.Optimizer,
                         eps: float = None,
                         cfg: Optional[None] = None,
                         *,
                         step: int | None = None,
                         sgd_mode: str = "time") -> float:
    group = optimizer.param_groups[0]
    base_lr = float(group.get('lr', 0.0))

    if cfg is not None and getattr(cfg, "optimizer", "").lower() == "sgd":
        momentum = float(group.get('momentum', 0.0))

        if step is None:
            inferred = 0
            for p in group.get("params", []):
                st = optimizer.state.get(p, None)
                if st is not None and "step" in st:
                    inferred = max(inferred, int(st["step"]))
            step = inferred if inferred > 0 else None

        mode = "time" if sgd_mode == "time" else "asymptotic"
        return sgd_momentum_eta_eff(base_lr, momentum, step, mode=mode)

    eps = group.get('eps', 1e-8) if eps is None else eps

    effs = []
    for p in group.get("params", []):
        st = optimizer.state.get(p, None)
        if not st:
            continue
        v = st.get("max_exp_avg_sq") if group.get("amsgrad", False) and "max_exp_avg_sq" in st \
            else st.get("exp_avg_sq")
        if v is None or v.numel() == 0:
            continue

        beta1, beta2 = group.get("betas", (0.9, 0.999))
        step_p = int(st.get("step", 0))
        if step_p < 1:
            continue
        bc1 = 1.0 - (beta1 ** step_p)
        bc2 = 1.0 - (beta2 ** step_p)
        v_hat_mean = v.float().mean().item() / bc2
        denom = math.sqrt(max(v_hat_mean, 0.0)) + eps

        raw = base_lr / (bc1 * denom)

        lr_min = getattr(optimizer, "lr_min", -float("inf"))
        lr_max = getattr(optimizer, "lr_max",  float("inf"))
        effs.append(max(lr_min, min(raw, lr_max)))
    return float(np.mean(effs)) if effs else base_lr



def per_layer_effective_lr(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    include_ndim1: bool = False,
    eps: float | None = None
) -> dict[str, float]:
    import math
    id2name = {id(p): n for n, p in model.named_parameters()}
    layer_sum: dict[str, float]   = {}
    layer_cnt: dict[str, int]     = {}

    LR_MIN = getattr(optimizer, "lr_min", float("-inf"))
    LR_MAX = getattr(optimizer, "lr_max", float("inf"))

    for group in optimizer.param_groups:
        base_lr   = float(group.get("lr", 0.0))
        if base_lr == 0.0:
            continue
        g_eps     = float(group.get("eps", 1e-8) if eps is None else eps)
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        use_ams   = bool(group.get("amsgrad", False))

        group_param_ids = {id(p) for p in group.get("params", [])}

        for p in group.get("params", []):
            if id(p) not in group_param_ids or not getattr(p, "requires_grad", False):
                continue
            if (not include_ndim1) and (p.ndim <= 1):
                continue

            st = optimizer.state.get(p)
            if not st:
                continue
            t = int(st.get("step", 0))
            if t < 1:
                continue 

            v = st.get("max_exp_avg_sq") if (use_ams and "max_exp_avg_sq" in st) else st.get("exp_avg_sq")
            if v is None or v.numel() == 0:
                continue

            bc1 = 1.0 - (beta1 ** t)
            bc2 = 1.0 - (beta2 ** t)
            if bc1 <= 0.0 or bc2 <= 0.0:
                continue

            v_hat_mean = float(v.float().mean().item()) / bc2
            denom = math.sqrt(max(v_hat_mean, 0.0)) + g_eps

            raw_step = base_lr / (bc1 * denom)    
            step_clamped = max(LR_MIN, min(raw_step, LR_MAX))

            name = id2name.get(id(p))
            if name is None:
                continue
            layer = name.split(".", 1)[0]
            layer_sum[layer] = layer_sum.get(layer, 0.0) + step_clamped
            layer_cnt[layer] = layer_cnt.get(layer, 0) + 1

    return {k: (layer_sum[k] / layer_cnt[k]) for k in layer_sum}

def per_layer_sgd_lr(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    include_ndim1: bool = False,
    step: int | None = None,
    sgd_mode: str = "time"
) -> dict[str, float]:
    id2name = {id(p): name for name, p in model.named_parameters()}
    layer_lrs: dict[str, float] = {}

    group = optimizer.param_groups[0]
    base_lr = float(group.get("lr", 0.0))
    mu      = float(group.get("momentum", 0.0))

    if step is None:
        inferred = 0
        for p in group.get("params", []):
            st = optimizer.state.get(p, None)
            if st is not None and "step" in st:
                inferred = max(inferred, int(st["step"]))
        step = inferred if inferred > 0 else None

    eff = sgd_momentum_eta_eff(base_lr, mu, step,
                               mode=("time" if sgd_mode == "time" else "asymptotic"))

    group_param_ids = {id(q) for q in group.get("params", [])}
    for q in group.get("params", []):
        if id(q) not in group_param_ids:
            continue
        if not getattr(q, "requires_grad", False):
            continue
        if (not include_ndim1) and (q.ndim <= 1):
            continue

        name = id2name.get(id(q), None)
        if name is None:
            continue
        layer = name.split('.', 1)[0] if '.' in name else name
        layer_lrs[layer] = eff
    return layer_lrs

import math
from typing import Optional, Literal

def sgd_momentum_eta_eff(
    lr: float,
    momentum: float,
    step: int | None,
    *,
    mode: Literal["time", "asymptotic"] = "time"
) -> float:
    mu = float(momentum)
    if mu == 0.0:
        return float(lr)
    if mode == "asymptotic":
        return float(lr) / (1.0 - mu)

    t = max(int(step or 0), 1)
    return float(lr) * (1.0 - (mu ** t)) / (1.0 - mu)


def sgd_weight_decay_factor(
    lr: float,
    weight_decay: float,
    *,
    decoupled: bool
) -> float:
    lam = float(weight_decay)
    if lam <= 0.0:
        return 1.0
    if decoupled:
        return max(0.0, 1.0 - float(lr) * lam)
    else:
        return 1.0


class PerLayerLyapunovScheduler:
    def __init__(self, optimizer, layer_states,
                 safety=0.9, cool=0.999, warm=1.01, cfg=None):
        self.opt          = optimizer
        self.layer_states = layer_states
        self.safety, self.cool, self.warm = cfg.safety, cool, warm

        self.layer2groups = {}
        for i, g in enumerate(self.opt.param_groups):
            lyr = g.get('layer', None)
            if lyr is not None:
                self.layer2groups.setdefault(lyr, []).append(i)

    def step(self, layer: str, eff_lr: float, tau: float, current_step, total_steps) -> float:
        if tau == 0.0:
            return tau 
        if layer not in self.layer2groups:
            return tau

        lr_star = tau 
        if eff_lr > 0.12 and eff_lr > self.safety * lr_star:
            factor = self.cool
        elif current_step < 0.1 * total_steps and eff_lr < 0.3 * self.safety * lr_star: 
            factor = self.warm
        else:
            return lr_star            

        for gi in self.layer2groups[layer]:
            self.opt.param_groups[gi]['lr'] *= factor
        return lr_star
        
class GradSNR:
    def __init__(self):
        self.prev = None

    @torch.no_grad()
    def update(self,
               grad_flat: torch.Tensor,
               eta_eff: float,
               batch_B: int) -> tuple[float | None, float | None]:
        if grad_flat is None:
            return None, None
        if self.prev is None or self.prev.numel() != grad_flat.numel():
            self.prev = grad_flat.detach().clone()
            return None, None

        diff = (grad_flat - self.prev)
        sigma2_hat = 0.5 * float(diff.pow(2).sum().item())

        T_t = None
        if sigma2_hat > 0.0 and eta_eff > 0.0:
            gnorm2 = float(grad_flat.pow(2).sum().item())
            T_t = (eta_eff * (sigma2_hat / max(1, batch_B))) / gnorm2

        self.prev = grad_flat.detach().clone()
        return T_t, sigma2_hat

def grad_variance_within_batch(model, loss_fn, inputs, targets):
    device = inputs.device
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    per_sample_losses = loss_fn(outputs, targets)
    if per_sample_losses.ndim > 1:
        per_sample_losses = per_sample_losses.mean(dim=tuple(range(1, per_sample_losses.ndim)))

    B = int(per_sample_losses.shape[0])

    grads = []
    for i in range(B):
        loss_i = per_sample_losses[i]
        retain = (i < B - 1)
        gi = torch.autograd.grad(loss_i, params, retain_graph=retain, allow_unused=False)
        gi_flat = torch.cat([g.contiguous().view(-1) for g in gi])
        grads.append(gi_flat.detach())

    G = torch.stack(grads, dim=0)  
    g_bar = G.mean(dim=0)
    sigma2 = ((G - g_bar)**2).sum(dim=1).mean().item()

    return sigma2

class SNRProgressPredictor:
    def __init__(self, margin: float = 0.0, window: int = 20):
        self.delta   = float(margin)
        self.window  = int(window)
        self.win     = deque(maxlen=self.window)

    def update(self,  T_mb: float | None, T_proxy: float | None):
        t = T_mb if (T_mb is not None and not math.isnan(T_mb)) else (
            T_proxy if (T_proxy is not None and not math.isnan(T_proxy)) else None
        )
        if t is None:
            return None, None, None, None, None 

        self.win.append(float(t))
        meanT = sum(self.win) / len(self.win)
        thresh = 1.0 + self.delta
        pred = 1 if meanT >= thresh else 0
        pred_real = 1 if t >= thresh else 0

        gap = abs(meanT - thresh)
        conf = min(0.99, gap / 0.5) * (len(self.win) / self.win.maxlen)

        return pred, pred_real, meanT, thresh, conf

def grad_variance_within_batch_by_layer(model, loss_fn, inputs, targets, layer_map):
    device = inputs.device
    params_all = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    per_sample_losses = loss_fn(outputs, targets)
    if per_sample_losses.ndim > 1:
        per_sample_losses = per_sample_losses.mean(dim=tuple(range(1, per_sample_losses.ndim)))

    B = int(per_sample_losses.shape[0])

    layer_param_lists = {layer: [p for p in plist if p.requires_grad]
                         for layer, plist in layer_map.items()}

    layer_grads = {layer: [] for layer in layer_param_lists}
    for i in range(B):
        loss_i = per_sample_losses[i]
        retain = (i < B - 1)
        gi = torch.autograd.grad(loss_i, params_all, retain_graph=retain, allow_unused=False)
        idx = 0
        name2grad = {}
        for p in params_all:
            g = gi[idx]
            name2grad[id(p)] = g
            idx += 1
        for layer, plist in layer_param_lists.items():
            g_l = torch.cat([name2grad[id(p)].contiguous().view(-1) for p in plist], dim=0)
            layer_grads[layer].append(g_l.detach())

    sigma2_by_layer = {}
    for layer, Glist in layer_grads.items():
        if len(Glist) == 0:
            sigma2_by_layer[layer] = 0.0
            continue
        G = torch.stack(Glist, dim=0) 
        g_bar = G.mean(dim=0)
        sigma2 = ((G - g_bar) ** 2).sum(dim=1).mean().item()
        sigma2_by_layer[layer] = float(sigma2)

    return sigma2_by_layer


# class ClampedAdam(Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False,
#                  lr_min=1e-6, lr_max=1.0):
#         super().__init__(params, lr=lr, betas=betas, eps=eps,
#                          weight_decay=weight_decay, amsgrad=amsgrad)
#         self.lr_min = lr_min
#         self.lr_max = lr_max

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad

#                 state = self.state[p]

#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = torch.zeros_like(p)
#                     state['exp_avg_sq'] = torch.zeros_like(p)
#                     if group['amsgrad']:
#                         state['max_exp_avg_sq'] = torch.zeros_like(p)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']

#                 state['step'] += 1
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#                 denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

#                 effective_lr = group['lr'] / denom
#                 effective_lr = torch.clamp(effective_lr, self.lr_min, self.lr_max)

#                 step_size = effective_lr / bias_correction1
#                 p.addcmul_(exp_avg, -step_size)

#         return loss
