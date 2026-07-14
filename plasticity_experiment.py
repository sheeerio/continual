import argparse
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import matplotlib.pyplot as plt
from collections import deque

# Import your utilities
from utils import misc, optimizers

# =============================================================================
# 1. OPTIMIZATION & METRIC UTILS
# =============================================================================

def empirical_rank(model, inputs):
    """
    Computes the effective rank and dead unit percentage of the representation.
    Expects 'inputs' to be the state indices (long tensor) if ValueNet handles embedding internally.
    """
    model.eval()
    with torch.no_grad():
        acts = []
        def hook(m, i, o): acts.append(o)
        
        # Hook into the first ReLU (model.net[1]) to capture representations
        handle = model.net[1].register_forward_hook(hook) 
        model(inputs) 
        handle.remove()
        
        if len(acts) == 0: return 0.0, 0.0
        
        H = acts[0] # [Batch, Hidden]
        
        # Dead Units % (neurons that never fired in this batch)
        dead = (H <= 0).float().mean().item()
        
        # Effective Rank using SVD of centered activations
        H_centered = H - H.mean(dim=0, keepdim=True)
        try:
            S = torch.linalg.svdvals(H_centered)
            S = S / (S.sum() + 1e-12)
            entropy = -(S * (S + 1e-12).log()).sum()
            eff_rank = torch.exp(entropy).item()
        except:
            eff_rank = 0.0
            
        return eff_rank, dead


def estimate_hessian_topk(loss, params, k=1, iters=10):
    """
    Estimates top-k eigenvalues of the Hessian w.r.t params.
    Args:
        loss: A scalar Tensor (graph must be retained!).
        params: List of parameters to compute Hessian for.
    """
    # 1. Compute Gradient (create_graph=True allows HVP later)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    
    # Filter out None grads (in case some params didn't contribute to this specific loss)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads if g is not None])
    n = flat_grad.numel()

    # HVP Helper
    def hvp(v):
        g_v = (flat_grad * v).sum()
        # retain_graph=True needed because we might call this multiple times (iters)
        hv = torch.autograd.grad(g_v, params, retain_graph=True, allow_unused=True)
        return torch.cat([h.contiguous().view(-1) for h in hv if h is not None]).detach()

    eigs = []
    vs = []
    
    # Run Power Iteration
    for _ in range(k):
        v = torch.randn(n, device=flat_grad.device)
        v = v / (v.norm() + 1e-12)
        
        for _ in range(iters):
            w = hvp(v)
            # Orthogonalize against found eigenvectors
            for j, u in enumerate(vs):
                w = w - eigs[j] * (u.dot(w)) * u
            v = w / (w.norm() + 1e-12)
        
        Hv = hvp(v)
        lam = v.dot(Hv).item()
        eigs.append(lam)
        vs.append(v)
    
    return eigs

def get_norm_sharpness(optimizer, sharpness):
    """
    Calculates Normalized Sharpness specifically for Adam.
    Norm Sharpness = Sharpness * (LearningRate / sqrt(SecondMoment))
    """
    v_squares = []
    
    # 1. Collect second moment (v) from Adam state
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            # 'exp_avg_sq' is the v term in Adam
            if "exp_avg_sq" in state:
                v_sq = state["exp_avg_sq"].detach()
                v_squares.append(v_sq.view(-1))
    
    # 2. Calculate aggregation
    if len(v_squares) > 0:
        v_cat = torch.cat(v_squares)
        # Average second moment across all parameters
        rms = torch.sqrt(v_cat.mean() + 1e-16)
        
        lr0 = optimizer.param_groups[0]["lr"]
        eps = optimizer.param_groups[0]["eps"]
        
        # Effective Step Size Scale ≈ lr / (rms + eps)
        alpha_agg = lr0 / (rms + eps)
        
        # Return normalized sharpness
        return sharpness * alpha_agg.item()
    
    # Fallback if state is empty (start of training)
    return sharpness


class PerLayerLyapunovScheduler:
    """
    Adjusts *each* param-group’s LR so that effective_lr ≲ τ (“lr_star”)
    where τ comes from your EMAState collapse bound.
    """
    def __init__(self, optimizer, layer_states,
                 safety=0.9, cool=0.999, warm=1.01, cfg=None):
        self.opt          = optimizer
        self.layer_states = layer_states
        self.safety, self.cool, self.warm = cfg.ly_safety, cfg.ly_cool, cfg.ly_warm
        self.cfg = cfg

        # map "fc1" → [group_idx, …]  (usually just one group per layer)
        self.layer2groups = {}
        for i, g in enumerate(self.opt.param_groups):
            lyr = g.get('layer', None)
            if lyr is not None:
                self.layer2groups.setdefault(lyr, []).append(i)

    def step(self, layer: str, eff_lr: float, tau: float, current_step, total_steps) -> float:
        """
        Call once per layer; mutates that layer’s LR *in place*.
        Returns lr_star so you can log it.
        """
        if tau == 0.0:
            return tau                        # not initialised yet
        if layer not in self.layer2groups:
            return tau

        lr_star = tau                         # theoretical upper-bound
        
        # Scheduling Logic
        if eff_lr > 0.12 and eff_lr > self.safety * lr_star:    # too aggressive ⇒ cool
            factor = self.cool
        else:
            return lr_star                    # inside band → do nothing

        # apply factor to all groups that belong to this layer
        for gi in self.layer2groups[layer]:
            self.opt.param_groups[gi]['lr'] *= factor
        return lr_star

def make_layer_groups(model, base_lr):
    """Creates param groups split by layer for per-layer control."""
    layer_map = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            # simple logic: "net.0.weight" -> "net.0"
            layer = ".".join(name.split('.')[:2]) 
            layer_map.setdefault(layer, []).append(p)
    
    layer_groups = [{'params': params, 'lr': base_lr, 'layer': layer}
                    for layer, params in layer_map.items()]
    return layer_map, layer_groups

# =============================================================================
# 2. REGULARIZERS
# =============================================================================

class ParsevalReg:
    def __init__(self, coef): self.coef = coef
    def __call__(self, model):
        if self.coef <= 0.0: return 0.0
        reg = 0.0
        for name, p in model.named_parameters():
            if "weight" in name and p.ndim >= 2:
                W = p.view(p.shape[0], -1)
                I = torch.eye(W.shape[1], device=p.device)
                reg += torch.norm(W.t() @ W - I, p='fro')**2
        return self.coef * reg

class WassersteinReg:
    def __init__(self, coef, model):
        self.coef = coef
        self.init = {n: torch.sort(p.detach().view(-1))[0] for n, p in model.named_parameters() if p.requires_grad}
    def __call__(self, model):
        if self.coef <= 0.0: return 0.0
        reg = 0.0
        for n, p in model.named_parameters():
            if n in self.init: 
                w_sorted = torch.sort(p.view(-1))[0]
                reg += torch.norm(w_sorted - self.init[n].to(p.device))**2
        return self.coef * reg

class SpectralReg:
    def __init__(self, coef, k=1): 
        self.coef = coef
        self.k = k
    def __call__(self, model):
        if self.coef <= 0.0: return 0.0
        reg = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and p.ndim >= 2:
                # Spectral Loss: (sigma^k - 1)^2
                sigma = optimizers.power_iteration(p, iters=1)
                reg += (sigma.pow(self.k) - 1.0).pow(2)
        return self.coef * reg

# =============================================================================
# 3. ENVIRONMENT (PAPER EXACT)
# =============================================================================

class ParsevalGridWorld:
    def __init__(self, size=15, seed=0, permute=True):
        self.size = size
        self.num_states = size * size
        self.rng = np.random.default_rng(seed)
        self.goal = (self.rng.integers(0, size), self.rng.integers(0, size))
        self.perm = self.rng.permutation(self.num_states) if permute else np.arange(self.num_states)

    def get_dist(self, r, c):
        return abs(r - self.goal[0]) + abs(c - self.goal[1])

    def get_state_idx(self, r, c):
        return self.perm[r * self.size + c]

    def step(self, r, c, a):
        nr, nc = r, c
        if a == 0: nr = max(0, r-1)
        elif a == 1: nc = min(self.size-1, c+1)
        elif a == 2: nr = min(self.size-1, r+1)
        elif a == 3: nc = max(0, c-1)
        # Reward: length of shortest path / 10 (Negative for minimization)
        return nr, nc, -self.get_dist(nr, nc) / 10.0

def evaluate_success_rate(model, env, device, n_episodes=10, max_steps=100, is_random=False):
    model.eval()
    successes = 0
    for _ in range(n_episodes):
        r, c = env.rng.integers(0, env.size), env.rng.integers(0, env.size)
        while (r, c) == env.goal:
             r, c = env.rng.integers(0, env.size), env.rng.integers(0, env.size)     
        for _ in range(max_steps):
            if (r, c) == env.goal:
                successes += 1
                break
            # Greedy Policy
            if is_random:
                best_act = env.rng.integers(0, 4)
            best_val, best_act = -float('inf'), 0
            for a in range(4):
                tr, tc = r, c
                if a == 0: tr = max(0, r-1)
                elif a == 1: tc = min(env.size-1, c+1)
                elif a == 2: tr = min(env.size-1, r+1)
                elif a == 3: tc = max(0, c-1)
                s_idx = torch.tensor([env.get_state_idx(tr, tc)]).to(device)
                with torch.no_grad(): val = model(s_idx).item()
                if val > best_val: best_val, best_act = val, a
            # Step
            if best_act == 0: r = max(0, r-1)
            elif best_act == 1: c = min(env.size-1, c+1)
            elif best_act == 2: r = min(env.size-1, r+1)
            elif best_act == 3: c = max(0, c-1)
    return successes / n_episodes

class RLDataset(Dataset):
    def __init__(self, mode, env, episodes=500, gamma=0.9, steps=100):
        self.data, self.mode = [], mode
        for _ in range(episodes):
            r, c = env.rng.integers(0, env.size), env.rng.integers(0, env.size)
            traj = []
            for _ in range(steps):
                s_idx, action = env.get_state_idx(r, c), env.rng.integers(0, 4)
                nr, nc, rew = env.step(r, c, action)
                ns_idx = env.get_state_idx(nr, nc)
                if mode == 'td': self.data.append((s_idx, rew, ns_idx, 0.0))
                else: traj.append((s_idx, rew))
                r, c = nr, nc
            if mode == 'mc':
                G = 0
                for s_idx, rew in reversed(traj):
                    G = rew + gamma * G
                    self.data.append((s_idx, G))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): 
        d = self.data[i]
        if self.mode == 'mc': return torch.tensor(d[0]).long(), torch.tensor(d[1]).float()
        return torch.tensor(d[0]).long(), torch.tensor(d[1]).float(), torch.tensor(d[2]).long(), torch.tensor(d[3]).float()

# =============================================================================
# 4. MODEL & MAIN
# =============================================================================

class ValueNet(nn.Module):
    def __init__(self, num_states=225, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden),
            # nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.num_states = num_states
    def forward(self, idx):
        x = torch.nn.functional.one_hot(idx.long(), num_classes=self.num_states).float()
        return self.net(x).squeeze(-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['mc', 'td'], default='mc')
    parser.add_argument("--algo", type=str, choices=['vanilla', 'reset', 'parseval', 'wasserstein', 'l2', 'spectral', 'random'], default='vanilla')
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--steps_per_task", type=int, default=40000) 
    parser.add_argument("--eval_freq", type=int, default=2000)      
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    
    # Scheduler Params
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "pl_lyapunov"])
    parser.add_argument("--sched_param", type=str, default="tau", choices=["tau", "svar", "t", "svar1", "sqm1"], help="Which statistic to schedule on")
    parser.add_argument("--ly_safety", type=float, default=0.9)
    parser.add_argument("--ly_cool", type=float, default=0.999)
    parser.add_argument("--ly_warm", type=float, default=1.00001)

    parser.add_argument("--adaptive_reg", action="store_true", help="Enable local adaptive regularization")
    parser.add_argument("--adaptive_type", type=str, default="l2", choices=["l2", "spectral", "wasserstein", "parseval"])
    parser.add_argument("--adaptive_scope", type=str, default="local", choices=["local", "global"])
    parser.add_argument("--adaptive_target", type=float, default=1.0)
    parser.add_argument("--reg_sensitivity", type=float, default=0.001)
    parser.add_argument("--spectral_k", type=int, default=1)

    args = parser.parse_args()

    misc.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_name = f"{args.mode}_lr{args.lr}_{args.algo}"
    if args.lr_schedule == "pl_lyapunov":
        run_name += f"_sched-{args.sched_param}"
    if args.adaptive_reg:
        run_name += f"_adapt-{args.adaptive_type}_sens{args.reg_sensitivity}_target{args.adaptive_target}"
    else:
        run_name += f"_reg{args.reg_coef}"

    wandb.init(
        project="rl-plasticity-comprehensive", 
        group="yeah", 
        config=vars(args), 
        name=run_name
    )
    model = ValueNet().to(device)
    
    # [ADDED] Initialize reference weights for Adaptive Wasserstein support
    model.init = {n: torch.sort(p.detach().view(-1))[0] for n, p in model.named_parameters()}
    
    # Create Layer Groups
    layer_map, layer_groups = make_layer_groups(model, args.lr)
    optimizer = optim.Adam(layer_groups, lr=args.lr)

    # 1. Init Per-Layer States
    layer_states = {
        layer: misc.EMAState(alphas=(0.01, 0.05, 0.5))
        for layer in layer_map
    }
    
    # 2. Init GLOBAL Tracker (Restored)
    global_tau_tracker = misc.EMAState(alphas=(0.01, 0.05, 0.5))

    # Initialize Scheduler
    pl_scheduler = None
    if args.lr_schedule == "pl_lyapunov":
        pl_scheduler = PerLayerLyapunovScheduler(
            optimizer=optimizer,
            layer_states=layer_states,
            cfg=args
        )

    # Static Regularizers
    if args.algo == 'parseval':
        reg_fn = ParsevalReg(args.reg_coef)
    elif args.algo == 'wasserstein':
        reg_fn = WassersteinReg(args.reg_coef, model)
    elif args.algo == 'l2':
        reg_fn = lambda m: args.reg_coef * sum(p.pow(2).sum() for p in m.parameters())
    elif args.algo == 'spectral':
        reg_fn = SpectralReg(args.reg_coef, k=args.spectral_k)
    else:
        reg_fn = lambda m: 0.0

    all_success_rates = []
    utility_per_task = []
    total_steps = 0
    total_updates = 0

    for t in range(args.tasks):
        print(f"=== Task {t+1}/{args.tasks} ===")
        env = ParsevalGridWorld(size=15, seed=args.seed + t, permute=True)
        
        if args.algo == 'reset':
            model = ValueNet().to(device)
            # Re-init model.init for Reset algo
            model.init = {n: torch.sort(p.detach().view(-1))[0] for n, p in model.named_parameters()}
            
            layer_map, layer_groups = make_layer_groups(model, args.lr)
            optimizer = optim.Adam(layer_groups, lr=args.lr)
            
            # Reset Both Global and Layer States
            layer_states = {layer: misc.EMAState(alphas=(0.01, 0.05, 0.5)) for layer in layer_map}
            global_tau_tracker = misc.EMAState(alphas=(0.01, 0.05, 0.5))
            
            if pl_scheduler: pl_scheduler = PerLayerLyapunovScheduler(optimizer, layer_states, cfg=args)

        # Dataset Regeneration
        dataset = RLDataset(args.mode, env, episodes=600, steps=100)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        loader_iter = iter(loader)
        
        task_successes = []
        batches_total = args.steps_per_task // 64
        log_at = args.log_interval // 64
        eval_at = args.eval_freq // 64

        for b in range(batches_total):
            total_steps += 64
            total_updates += 1
            model.train()
            try: batch = next(loader_iter)
            except StopIteration: 
                loader_iter = iter(loader)
                batch = next(loader_iter)

            if args.algo == 'random': 
                loss = torch.tensor(0.0)
            else:
                optimizer.zero_grad()
                if args.mode == 'mc':
                    s, target = [x.to(device) for x in batch]
                    criterion = nn.MSELoss(reduction='none') 
                    raw_loss = criterion(model(s), target)
                    loss = raw_loss.mean()
                else:
                    s, r, ns, d = [x.to(device) for x in batch]
                    target = r + 0.9 * (1-d) * model(ns).detach()
                    criterion = nn.MSELoss(reduction='none')
                    raw_loss = criterion(model(s), target.float())
                    loss = raw_loss.mean()
                
                should_log = (b % log_at == 0)

                # Container to pass scalars from Adaptive Block to Logging Block
                step_layer_scalars = {}
                # Initialize global_scalars to None to ensure it is defined
                global_scalars = None 
                raw_reg_values = {}

                # =================================================================
                # ADAPTIVE REGULARIZATION
                # =================================================================
                adaptive_reg_loss = torch.tensor(0.0, device=device)
                if args.adaptive_reg:
                    # 1. Global Pre-calculation (if global scope)
                    global_factor = None
                    if args.adaptive_scope == "global":
                        params_all = [p for p in model.parameters() if p.requires_grad]
                        lam_global = estimate_hessian_topk(loss, params_all, k=1, iters=1)[0]
                        norm_lam_global = get_norm_sharpness(optimizer, lam_global)
                        
                        if optimizer.__class__ == optim.Adam:
                            tmp_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                            g_eff_lr = np.mean(list(tmp_eff_lrs.values())) if tmp_eff_lrs else args.lr
                        else:
                            g_eff_lr = args.lr

                        _, global_scalars = misc.update_stat(norm_lam_global, global_tau_tracker, g_eff_lr)
                        global_factor = args.reg_sensitivity * math.log(1.0 + (1.0 / (global_scalars["tau"] + 1e-12)))

                    # 2. Iterate Layers
                    if optimizer.__class__ == optim.Adam and 'tmp_eff_lrs' not in locals():
                        tmp_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                    elif 'tmp_eff_lrs' not in locals():
                        tmp_eff_lrs = {}
                    
                    for layer, l_params in layer_map.items():
                        # A. Calculate Local Stats
                        lam = estimate_hessian_topk(loss, l_params, k=1, iters=1)[0]
                        norm_lam = get_norm_sharpness(optimizer, lam)
                        eff_lr_layer = tmp_eff_lrs.get(layer, args.lr)
                        
                        # Update Local State & Capture Scalars
                        _, l_scalars = misc.update_stat(norm_lam, layer_states[layer], eff_lr_layer)
                        step_layer_scalars[layer] = l_scalars
                        
                        # B. Determine Adaptive Factor
                        if args.adaptive_scope == "global":
                            adapt_factor = global_factor
                        else:
                            adapt_factor = args.reg_sensitivity * math.log(1.0 + (args.adaptive_target / (l_scalars["tau"] + 1e-12)))

                        # C. Compute Regularization Value
                        layer_val = torch.tensor(0.0, device=device)

                        if args.adaptive_type == "l2":
                            layer_val = sum(p.pow(2).sum() for p in l_params)
                        elif args.adaptive_type == "spectral":
                            for p in l_params:
                                if p.ndim >= 2:
                                    sigma = optimizers.power_iteration(p, iters=1)
                                    layer_val += (sigma.pow(args.spectral_k) - 1.0).pow(2)
                        elif args.adaptive_type == "wasserstein":
                            for n, p in model.named_parameters():
                                if any(p is lp for lp in l_params) and n in model.init:
                                    w_sorted = torch.sort(p.view(-1))[0]
                                    w_init = model.init[n].to(device)
                                    layer_val += torch.norm(w_sorted - w_init)**2
                        elif args.adaptive_type == "parseval":
                            for p in l_params:
                                if p.ndim >= 2:
                                    W = p.view(p.shape[0], -1)
                                    dim = W.shape[1]
                                    I = torch.eye(dim, device=device)
                                    layer_val += torch.norm(W.t() @ W - I, p='fro')**2

                        raw_reg_values[layer] = layer_val.item()
                        adaptive_reg_loss += adapt_factor * layer_val
                        
                        if should_log:
                            wandb.log({f"{layer}/adapt_factor": adapt_factor,
                                        f"{layer}/raw_reg": layer_val.item()}, commit=False)

                # Backprop
                (loss + reg_fn(model) + adaptive_reg_loss).backward(retain_graph=should_log)
                
                # --- LOGGING & SCHEDULING BLOCK ---
                if should_log:
                    params_all = [p for p in model.parameters() if p.requires_grad]
                    
                    # A. Global Stats
                    p_norm = sum(p.norm(2).item()**2 for p in params_all)**0.5
                    g_norm = sum(p.grad.norm(2).item()**2 for p in params_all if p.grad is not None)**0.5
                    eff_rank, dead_units = empirical_rank(model, s)

                    lam_global = estimate_hessian_topk(loss, params_all, k=1, iters=1)[0]
                    norm_lam_global = get_norm_sharpness(optimizer, lam_global)
                    
                    layer_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                    global_eff_lr = np.mean(list(layer_eff_lrs.values())) if layer_eff_lrs else args.lr
                    
                    # FIXED: Logic to ensure global_scalars is defined even if adaptive_scope != global
                    if global_scalars is None:
                        _, global_scalars = misc.update_stat(norm_lam_global, global_tau_tracker, global_eff_lr)

                    layer_sigma2 = {}
                    if args.sched_param == "t":
                        # Use reduction='none' for variance calculation
                        if args.mode == 'mc':
                            var_criterion = nn.MSELoss(reduction='none')
                            var_targets = target
                        else:
                            var_criterion = nn.MSELoss(reduction='none')
                            var_targets = target.float()
                        layer_sigma2 = optimizers.grad_variance_by_layer(
                            model, var_criterion, s, var_targets, layer_map
                        )

                    # D. Iterate Layers
                    for layer, l_params in layer_map.items():
                        eff_lr_layer = layer_eff_lrs.get(layer, args.lr)
                        
                        # RETRIEVE OR CALCULATE SCALARS
                        if layer in step_layer_scalars:
                            l_scalars = step_layer_scalars[layer]
                        else:
                            # Adaptive Reg was OFF, so we must calculate now
                            lam = estimate_hessian_topk(loss, l_params, k=1, iters=1)[0]
                            norm_lam = get_norm_sharpness(optimizer, lam)
                            _, l_scalars = misc.update_stat(norm_lam, layer_states[layer], eff_lr_layer)
                        
                        # Scheduler Step
                        if pl_scheduler:
                            lr_star = 0.0
                            g_layer_sq = sum(p.grad.norm(2).item()**2 for p in l_params if p.grad is not None)
                            sigma2_l = layer_sigma2.get(layer, 0.0)
                            B = s.shape[0]
                            alpha_crit_t = (B * g_layer_sq) / max(sigma2_l, 1e-12)
                            if args.sched_param == "svar":
                                svar_sigma2_l = sigma2_l + 20.0 * g_layer_sq * l_scalars["lam_cv"]
                                alpha_crit_svar_layer10 = min((B * g_layer_sq) / max(svar_sigma2_l, 1e-12), alpha_crit_t)
                                lr_star = pl_scheduler.step(layer, eff_lr_layer, alpha_crit_svar_layer10, total_updates, args.tasks * args.steps_per_task)
                            elif args.sched_param == "tau":
                                svar_sigma2_l = sigma2_l + 1.0 * g_layer_sq * (1 / l_scalars["tau"] + 1e-12)
                                alpha_crit_svar_layer10 = min((B * g_layer_sq) / max(svar_sigma2_l, 1e-12), alpha_crit_t)
                                lr_star = pl_scheduler.step(layer, eff_lr_layer, alpha_crit_svar_layer10, total_updates, args.tasks * args.steps_per_task)
                            elif args.sched_param == "t":
                                lr_star = pl_scheduler.step(layer, eff_lr_layer, alpha_crit_t, total_updates, args.tasks * args.steps_per_task)
                                
                            wandb.log({
                                f"{layer}/grad_norm_sq": g_layer_sq,
                                f"{layer}/grad_variance": sigma2_l,
                                f"{layer}/alpha_crit": alpha_crit_t if args.sched_param == "t" else alpha_crit_svar_layer10,
                                f"{layer}/log_grad_norm_sq": math.log10(g_layer_sq + 1e-15),
                                f"{layer}/lr_star": lr_star,
                                f"{layer}/sharpness": lam,
                                f"{layer}/norm_sharpness": norm_lam,
                            }, commit=False)

                        # Log Per-Layer Data
                        wandb.log({
                            f"{layer}/lam_mean": l_scalars["lam_mean"],
                            f"{layer}/lam_var": l_scalars["lam_var"],
                            f"{layer}/tau": l_scalars["tau"],
                            f"{layer}/eff_lr": eff_lr_layer, 
                            f"{layer}/collapse_pred": l_scalars["collapse_pred"],
                            f"{layer}/lam_cv": l_scalars["lam_cv"],
                        }, commit=False)

                    # E. Commit ALL logs
                    wandb.log({
                        "metric/loss": loss.item(),
                        "metric/param_norm": p_norm,
                        "metric/grad_norm": g_norm,
                        "metric/effective_rank": eff_rank,
                        "metric/dead_units": dead_units,
                        "metric/sharpness": lam_global,
                        "metric/norm_sharpness": norm_lam_global,
                        "metric/tau": global_scalars["tau"],
                        "metric/global_eff_lr": global_eff_lr,
                        "debug/global_lam_mean": global_scalars["lam_mean"],
                        "global_step": total_steps,
                        "base_lr": optimizer.param_groups[0]['lr'],
                        "adaptive/reg_loss": adaptive_reg_loss.item() if args.adaptive_reg else 0.0,
                        "adaptive/total_raw_reg": sum(raw_reg_values.values()) if raw_reg_values else 0.0,
                    })

                optimizer.step()

            # --- EVALUATION BLOCK ---
            if b > 0 and b % eval_at == 0:
                sr = evaluate_success_rate(model, env, device, is_random=(args.algo == 'random'))
                task_successes.append(sr)
                all_success_rates.append(sr)
                wandb.log({"eval/success_rate": sr, "task_id": t})

        utility = np.mean(task_successes) if task_successes else 0.0
        utility_per_task.append(utility)
        print(f"Task {t+1} Utility: {utility:.4f}")
        wandb.log({
            "plasticity/utility_per_task": utility, 
            "plasticity/cumulative_avg_utility": np.mean(utility_per_task)
        })

    # Plot Generation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.tasks + 1), utility_per_task, marker='o', linewidth=2, label=args.algo)
    plt.xlabel("Task ID")
    plt.ylabel("Average Success Rate")
    plt.title(f"Plasticity Decay: {args.algo} ({args.mode})")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig("utility_decay.png")
    wandb.log({"plot/utility_decay": wandb.Image("utility_decay.png")})
    plt.close()

    wandb.finish()
    
if __name__ == "__main__":
    main()