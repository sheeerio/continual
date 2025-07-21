import torch
import random
import numpy as np
from torch.utils.data import Subset
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data
from config import get_parser

parser = get_parser()
config = parser.parse_args()

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

def empirical_fischer_rank(model, dataset, device, thresh=0.99, max_m=100):
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    m = 0
    for x, y in loader:
        if m >= max_m:
            break
        if config.model in ["CNN", "BatchNormCNN"]:
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

def get_norm_sharpness(optimizer, sharpness):
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

    if config.optimizer == "adam":
        norm_sharpness = sharpness * alpha_agg
    else:
        norm_sharpness = sharpness

    return norm_sharpness

def get_betas(config, epoch):
    if config.optimizer == "adam":
        end = 2 * config.epochs
        if epoch < end:
            beta1 = config.beta1 + (0.99 - config.beta1) * (epoch / (end))
            beta2 = config.beta2 + (0.75 - config.beta2) * (epoch / (end))
        else:
            beta1 = 0.99
            beta2 = 0.75
        return (beta1, beta2)
    else:
        raise ValueError("Optimizer is not Adam, cannot get betas.")