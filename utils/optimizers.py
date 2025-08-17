import torch
import random
import numpy as np
from torch.utils.data import Subset
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data
from collections import deque
from typing import Deque, Dict, Tuple, List
from utils.misc import EMAState

# def power_iteration_sigma_min(W: torch.Tensor,
#                                iters: int = 3,
#                                eps: float = 1e-6) -> torch.Tensor:
#     """
#     Approximate the *smallest* singular value of W via inverse power-iteration
#     on (WᵀW).  Works for any 2-D parameter tensor.

#     Cost: one solve per iteration; for mid-sized layers (≤1 k dims) 2–3 iters
#     add <0.2 ms on GPU.
#     """
#     # (1) build symmetric positive-definite matrix
#     WT_W = W.T @ W                         # shape (in, in)

#     # (2) start with a random unit vector
#     v = torch.randn(WT_W.shape[0], device=W.device)
#     v = v / v.norm()

#     # (3) inverse power-iteration:   x_{k+1} = (WT_W + εI)^{-1} v_k
#     #     solving a linear system is faster & stabler than an explicit inverse
#     I = torch.eye(WT_W.shape[0], device=W.device)

#     for _ in range(iters):
#         # linear solve; autograd-friendly
#         v = torch.linalg.solve(WT_W + eps * I, v)
#         v = v / (v.norm() + 1e-12)

#     # Rayleigh quotient → λ_min of WT_W, so √ gives σ_min
#     sigma_min_sq = torch.dot(v, WT_W @ v)
#     return torch.sqrt(sigma_min_sq + 1e-12)
def power_iteration_sigma_min(W: torch.Tensor,
                               iters: int = 3,
                               shift_mult: float = 1e-3) -> torch.Tensor:
    """
    Robust inverse-power iteration to approximate σ_min(W).

    • Works for rank-deficient layers (adds adaptive shift).
    • Uses torch.linalg.lstsq ← never throws 'matrix is singular'.
    • Chooses the cheaper of WᵀW  (n×n) or  W Wᵀ (m×m).
    """
    m, n = W.shape
    use_WtW = (n <= m)            # pick the smaller dimension
    A = W.T @ W if use_WtW else W @ W.T          # SPD but may be singular

    # --- adaptive ridge --------------------------------------------------
    diag_mean = A.diagonal().mean()
    ridge     = shift_mult * diag_mean + 1e-6    # fallback for all-zero diag
    A_shift   = A + ridge * torch.eye(A.size(0), device=W.device, dtype=W.dtype)

    # --- initial vector --------------------------------------------------
    v = torch.randn(A_shift.shape[0], device=W.device, dtype=W.dtype)
    v.div_(v.norm() + 1e-12)

    # --- inverse power iteration ----------------------------------------
    for _ in range(iters):
        # solve A_shift x = v   (least-squares is robust to near-singularity)
        x = torch.linalg.lstsq(A_shift, v.unsqueeze(-1)).solution.squeeze(-1)
        v = x / (x.norm() + 1e-12)

    # Rayleigh quotient → λ_min(A); σ_min(W) = √λ_min
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
    """
    Estimate the top eigenvalue of P^{-1} H, where
      P = diag(sqrt(nu)) + epsilon*I.
    Uses power iteration with Hessian-vector products.

    config:
      loss   - a scalar torch.Tensor (the loss at the current point)
      params - list of model parameters (with requires_grad=True)
      nu     - list of second-moment accumulators matching params
      epsilon- small float for numerical stability
      iters  - number of power-iteration steps

    Returns:
      approx top eigenvalue (float)
    """
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
    # dimension 0 error 
    j = (cumsum / total >= thresh).nonzero()[0].item() + 1 if total !=0 else 0
    return j / float(m)


def estimate_hessian_topk(model, loss, params, k=1, iters=100):
    # First backward pass to get the gradient
    grads = torch.autograd.grad(loss, params, create_graph=True)  # Retain the graph here
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
    n = flat_grad.numel()

    # Hessian-vector product function
    def hvp(v):
        g_v = (flat_grad * v).sum()
        hv = torch.autograd.grad(g_v, params, retain_graph=True)  # Retain the graph for second backward pass
        return torch.cat([h.contiguous().view(-1) for h in hv]).detach()

    eigs = []
    vs = []
    for _ in range(k):
        # Random initialization of eigenvector
        v = torch.randn(n, device=flat_grad.device)
        v = v / (v.norm() + 1e-12)
        
        # Power iteration for finding top k eigenvalues
        for _ in range(iters):
            w = hvp(v)
            for j, u in enumerate(vs):
                w = w - eigs[j] * (u.dot(w)) * u
            v = w / (w.norm() + 1e-12)
        
        Hv = hvp(v)
        lam = v.dot(Hv).item()  # Eigenvalue (top)
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
        raise ValueError("Optimizer is not Adam, cannot get betas.")


class LyapunovScheduler:
    """
    Lightweight LR-controller that *reads* the EMA statistics you already track
    in `sharp_state` (an `EMAState` instance) instead of keeping its own deque.
    """
    def __init__(self,
                opt,
                ema_state: EMAState,
                safety: float = 0.9,
                cool:   float = 0.5,
                warm:   float = 1.05):
        self.opt      = opt
        self.state    = ema_state          # shared object, updated elsewhere
        self.safety   = safety
        self.cool     = cool
        self.warm     = warm

    def step(self, effective_lr: float, tau: float) -> Tuple[float, float]:
        """
        Adjust the optimiser’s LR **in-place** according to the Lyapunov bound.
        `effective_lr` is the value you just computed for logging.
        Returns (lr_star, effective_lr) so you can log them.
        """
        # We need the running ⟨variation⟩; if not ready, do nothing.
        if not self.state.ema_variation or self.state.ema_variation == 0.0:
            return float('inf'), effective_lr

        lr_star = tau
        if effective_lr > 0.12 and effective_lr > self.safety * lr_star:    # too aggressive
            for g in self.opt.param_groups:
                g['lr'] *= self.cool
        # elif effective_lr < 0.5 * self.safety * lr_star:    # can warm up
        #     for g in self.opt.param_groups:
        #         g['lr'] *= self.warm

        return lr_star, effective_lr

def estimate_hessian_min_eig(model, loss, params, iters=100):
    """
    Power-iteration on the **negated** Hessian to get the most-negative
    eigen-value of H in O(iters) HVPs.
    Returns lambda_min (a negative number).
    """
    # 1) ∇ℓ (retain graph) -----------------------------------------------
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    d = g_flat.numel()

    # 2) HVP for -H --------------------------------------------------------
    def hvp_minus(v):
        g_v = (g_flat * v).sum()
        Hv  = torch.autograd.grad(g_v, params, retain_graph=True)
        return -torch.cat([h.contiguous().view(-1) for h in Hv]).detach()  # -H v

    # 3) vanilla power-iteration -----------------------------------------
    v = torch.randn(d, device=g_flat.device);  v /= v.norm() + 1e-12
    for _ in range(iters):
        w = hvp_minus(v)
        v = w / (w.norm() + 1e-12)

    Hv  = hvp_minus(v)                       # last HVP
    lam = v.dot(Hv).item()                   # this is  |λ_min|
    return -lam                              # flip sign → λ_min (<0)

import math
from typing import Optional

def compute_adam_effective_lr(optimizer: torch.optim.Adam,
                              eps: float = None
                              ) -> float:
    """
    For an Adam optimizer, average over all params the
    per‐param lr/(√(exp_avg_sq)+ε) to get a single effective lr.
    """
    group = optimizer.param_groups[0]
    base_lr = group['lr']
    eps     = group.get('eps', 1e-8) if eps is None else eps

    effs = []
    for p in group['params']:
        st = optimizer.state[p]
        if 'exp_avg_sq' in st:
            # compute RMS of the second‐moment buffer
            v_avg = st['exp_avg_sq'].mean().sqrt().item()
            effs.append(base_lr / (v_avg + eps))

    return float(np.mean(effs)) if effs else base_lr


# ---- CV-of-Sharpness feedback controller ------------------------------------
import numpy as np
from collections import deque
import torch, torch.optim as _optim


class CVSharpnessController(_optim.Optimizer):
    """
    A very thin “shim” that wraps an existing optimiser and
    nudges lr / weight-decay so that the coefficient of
    variation of a sharpness proxy hovers around `target`.
    """
    def __init__(self,
                 base_opt: _optim.Optimizer,
                 target: float = 1.0,
                 k_lr: float = 0.3,
                 k_wd: float = 0.15,
                 band: float = 0.05,
                 window: int = 100):
        # We deliberately do **not** touch `base_opt`’s defaults.
        self.opt = base_opt
        self.param_groups = self.opt.param_groups   # make PyTorch happy
        self.state        = self.opt.state

        self.target = target
        self.k_lr   = k_lr
        self.k_wd   = k_wd
        self.band   = band
        self.win    = deque(maxlen=window)

    # pass-through helpers -------------------------------------------------
    def zero_grad(self, *a, **kw): return self.opt.zero_grad(*a, **kw)
    def state_dict(self):          return self.opt.state_dict()
    def load_state_dict(self, sd):       self.opt.load_state_dict(sd)

    # the only method we actually change ----------------------------------
    def step(self, lam_cv=None, closure=None):
        err  = lam_cv - self.target
        if abs(err) > self.band:
            for g in self.opt.param_groups:
                raw_lr_factor = 1.0 - self.k_lr * err
                lr_factor     = max(0.1, min(raw_lr_factor, 1.0))
                g["lr"]      *= lr_factor
                # g["weight_decay"] *= (1.0 + self.k_wd * err)

        # 3. Do the real update
        return self.opt.step(closure)


from typing import Optional

def per_layer_effective_lr(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    include_ndim1: bool = False,   # False => skip 1D params (bias / BN)
    eps: Optional[float] = None    # override group eps if desired
) -> dict[str, float]:
    """
    Per-layer 'effective LR' for Adam/AdamW:
        eff(p) = lr_group / ( sqrt(mean(v_hat)) + eps_group )

    - Bias-corrects v (uses v_hat) and respects AMSGrad.
    - Aggregates by top-level module (e.g., 'fc1' from 'fc1.weight').
    - Skips non-trainable and, by default, 1D tensors.
    - Avoids tensor equality by matching params via id(p).
    """
    # Map parameter identity → name for aggregation
    id2name = {id(p): name for name, p in model.named_parameters()}

    layer_sums: dict[str, float] = {}
    layer_counts: dict[str, int] = {}

    for group in optimizer.param_groups:
        base_lr = float(group.get("lr", 0.0))
        if base_lr == 0.0:
            continue

        group_eps   = float(group.get("eps", 1e-8) if eps is None else eps)
        beta2       = float(group.get("betas", (0.9, 0.999))[1])
        use_amsgrad = bool(group.get("amsgrad", False))

        # Identity set to avoid tensor equality comparisons
        group_param_ids = {id(q) for q in group.get("params", [])}

        for q in group.get("params", []):
            if id(q) not in group_param_ids:   # (redundant, but explicit)
                continue
            if not getattr(q, "requires_grad", False):
                continue
            if (not include_ndim1) and (q.ndim <= 1):
                continue

            st = optimizer.state.get(q, None)
            if not st:
                continue

            # Choose second-moment buffer (AMSGrad if present)
            v = st.get("max_exp_avg_sq") if (use_amsgrad and "max_exp_avg_sq" in st) else st.get("exp_avg_sq")
            if v is None or v.numel() == 0:
                continue

            step = int(st.get("step", 0))
            if step <= 0:
                denom = group_eps
            else:
                bc2 = 1.0 - (beta2 ** step)
                if bc2 <= 0.0:
                    continue
                v_hat_mean = (v.float().mean().item() / bc2)
                denom = math.sqrt(max(v_hat_mean, 0.0)) + group_eps

            if denom <= 0.0 or not math.isfinite(denom):
                continue

            eff = base_lr / denom

            # Aggregate by layer name
            name = id2name.get(id(q), None)
            if name is None:
                continue
            layer = name.split('.', 1)[0] if '.' in name else name

            layer_sums[layer]   = layer_sums.get(layer, 0.0) + eff
            layer_counts[layer] = layer_counts.get(layer, 0)   + 1

    return {k: (layer_sums[k] / layer_counts[k]) for k in layer_sums}


class PerLayerLyapunovScheduler:
    """
    Adjusts *each* param-group’s LR so that   effective_lr ≲ τ (“lr_star”)
    where τ comes from your EMAState collapse bound.
    """
    def __init__(self, optimizer, layer_states,
                 safety=0.9, cool=0.50, warm=1.05):
        self.opt          = optimizer
        self.layer_states = layer_states
        self.safety, self.cool, self.warm = safety, cool, warm

        # map "fc1" → [group_idx, …]  (usually just one group per layer)
        self.layer2groups = {}
        for i, g in enumerate(self.opt.param_groups):
            lyr = g.get('layer', None)
            if lyr is not None:
                self.layer2groups.setdefault(lyr, []).append(i)

    def step(self, layer: str, eff_lr: float, tau: float, config) -> float:
        """
        Call once per layer; mutates that layer’s LR *in place*.
        Returns lr_star so you can log it.
        """
        if tau == 0.0:
            return tau                        # not initialised yet
        if layer not in self.layer2groups:
            return tau

        lr_star = tau                         # theoretical upper-bound
        if eff_lr > 0.12 and eff_lr > self.safety * lr_star:    # too aggressive ⇒ cool
            factor = self.cool
        # elif eff_lr < 0.04 * self.safety * lr_star:   # too timid ⇒ warm
        #     factor = self.warm
        else:
            return lr_star                    # inside band → do nothing

        # apply factor to all groups that belong to this layer
        for gi in self.layer2groups[layer]:
            self.opt.param_groups[gi]['lr'] *= factor
        return lr_star
        
# ---- Gradient SNR tracker (Mark's 2nd trade-off proxy) ------------------
class GradSNR:
    """
    Maintains a one-step estimate of data-gradient variance:
        sigma^2 ≈ 0.5 * ||g_t - g_{t-1}||^2
    and returns the trade-off ratio
        T_t = ||g_t||^2 / (eta_eff * sigma^2 / B)
    No state beyond previous gradient; no scheduling.
    """
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
    """
    σ² := (1/B) * Σ_i || g_i - ḡ ||²
    where g_i = ∇_θ ℓ(f(x_i), y_i).
    Torch-only (no functorch). Safe for occasional use at log intervals.
    """
    device = inputs.device
    params = [p for p in model.parameters() if p.requires_grad]

    # Forward once, get per-sample losses (no reduction)
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    per_sample_losses = loss_fn(outputs, targets)  # ensure loss_fn has reduction='none'
    if per_sample_losses.ndim > 1:
        per_sample_losses = per_sample_losses.mean(dim=tuple(range(1, per_sample_losses.ndim)))

    B = int(per_sample_losses.shape[0])

    grads = []
    # Collect per-sample grads (reuse graph; retain_graph=True except last)
    for i in range(B):
        loss_i = per_sample_losses[i]
        retain = (i < B - 1)
        gi = torch.autograd.grad(loss_i, params, retain_graph=retain, allow_unused=False)
        gi_flat = torch.cat([g.contiguous().view(-1) for g in gi])
        grads.append(gi_flat.detach())

    G = torch.stack(grads, dim=0)          # (B, P)
    g_bar = G.mean(dim=0)
    sigma2 = ((G - g_bar)**2).sum(dim=1).mean().item()   # scalar

    return sigma2

import math

class SNRProgressPredictor:
    """
    Rolling predictor: expects progress if mean T over a short window
    stays above (1 + margin). Uses T_mb (within-batch) if present,
    else falls back to temporal proxy T_t.
    """
    def __init__(self, margin: float = 0.0, window: int = 20):
        self.delta   = float(margin)
        self.window  = int(window)
        self.win     = deque(maxlen=self.window)

    def update(self,  T_mb: float | None, T_proxy: float | None):
        # pick signal: prefer within-batch T_mb
        t = T_mb if (T_mb is not None and not math.isnan(T_mb)) else (
            T_proxy if (T_proxy is not None and not math.isnan(T_proxy)) else None
        )
        if t is None:
            return None, None, None, None  # pred, meanT, thresh, conf

        self.win.append(float(t))
        meanT = sum(self.win) / len(self.win)
        thresh = 1.0 + self.delta
        pred = 1 if meanT >= thresh else 0
        pred_real = 1 if t >= thresh else 0

        # very light confidence: gap from threshold × fill ratio (clipped)
        gap = abs(meanT - thresh)
        conf = min(0.99, gap / 0.5) * (len(self.win) / self.win.maxlen)

        return pred, pred_real, meanT, thresh, conf

def grad_variance_within_batch_by_layer(model, loss_fn, inputs, targets, layer_map):
    """
    Per-layer within-minibatch gradient variance.
      σ²_layer := (1/B) * Σ_i || g_i^(layer) - ḡ^(layer) ||²

    Returns:
      dict: layer -> sigma2 (float)
    """
    device = inputs.device
    params_all = [p for p in model.parameters() if p.requires_grad]

    # Forward once, get per-sample losses (no reduction)
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    per_sample_losses = loss_fn(outputs, targets)
    if per_sample_losses.ndim > 1:
        per_sample_losses = per_sample_losses.mean(dim=tuple(range(1, per_sample_losses.ndim)))

    B = int(per_sample_losses.shape[0])

    # Build a stable order per layer
    layer_param_lists = {layer: [p for p in plist if p.requires_grad]
                         for layer, plist in layer_map.items()}

    # Collect per-sample grads, split by layer
    layer_grads = {layer: [] for layer in layer_param_lists}
    for i in range(B):
        loss_i = per_sample_losses[i]
        retain = (i < B - 1)
        gi = torch.autograd.grad(loss_i, params_all, retain_graph=retain, allow_unused=False)
        # map back to layers
        idx = 0
        # rebuild gi per parameter by iterating the same order as params_all
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
        G = torch.stack(Glist, dim=0)          # (B, P_l)
        g_bar = G.mean(dim=0)
        sigma2 = ((G - g_bar) ** 2).sum(dim=1).mean().item()
        sigma2_by_layer[layer] = float(sigma2)

    return sigma2_by_layer
