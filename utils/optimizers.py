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


def per_layer_effective_lr(model: torch.nn.Module,
                           optimizer: torch.optim.Adam) -> dict[str, float]:
    eps = optimizer.param_groups[0].get('eps', 1e-8)
    base_lr = optimizer.param_groups[0]['lr']

    layer_lrs = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:           # frozen params
            continue
        st = optimizer.state[p]
        if 'exp_avg_sq' not in st:        # first few steps
            continue
        v_rms   = st['exp_avg_sq'].mean().sqrt().item()
        eff_lr  = base_lr / (v_rms + eps)
        layer   = name.split('.')[0]      # "fc1.weight" → "fc1"
        layer_lrs.setdefault(layer, []).append(eff_lr)

    # average within each layer
    return {k: sum(v)/len(v) for k, v in layer_lrs.items()}

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

    def step(self, layer: str, eff_lr: float, tau: float) -> float:
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
        # elif eff_lr < 0.5 * self.safety * lr_star:   # too timid ⇒ warm
        #     factor = self.warm
        else:
            return lr_star                    # inside band → do nothing

        # apply factor to all groups that belong to this layer
        for gi in self.layer2groups[layer]:
            self.opt.param_groups[gi]['lr'] *= factor
        return lr_star
        
