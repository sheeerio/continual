from __future__ import annotations
import random
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence, Deque, Dict, Tuple, List


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

activation_map = {
    "relu": "relu",
    "leaky_relu": "leaky_relu",
    "tanh": "tanh",
    "identity": "linear",
    "crelu": "relu",
    "adalin": "leaky_relu",
    "softplus": "relu",
    "swish": "relu",
}

@dataclass
class EMAState:
    """Holds EMA means/vars plus windowed history for one statistic."""
    alphas: Sequence[float]            # e.g. (0.01, 0.05, 0.5)
    mu:  List[float] = field(default_factory=list)
    var2: List[float] = field(default_factory=list)
    queue: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    lam_var_queue: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    ema_variation: float | None = None

    def __post_init__(self):
        if not self.mu:   # lazy-init so we can reuse the same class for many stats
            self.mu   = [0.0] * len(self.alphas)
            self.var2 = [0.0] * len(self.alphas)


def update_stat(x: float,
                state: EMAState,
                effective_lr: float,
                smoothing: float = 0.97
               ) -> Tuple[EMAState, Dict[str, float]]:
    """
    Update EMAs, window stats, and collapse-bound predictions for a single value.
    Returns the *mutated* state (for convenience) and a dict of scalars you can log.
    """

    # --- 1. exponential moving means & variances --------------------------------
    vols = []
    for i, alpha in enumerate(state.alphas):
        prev_mu        = state.mu[i]
        state.mu[i]    = (1 - 0.5*alpha) * state.mu[i] + 0.5*alpha * x
        state.var2[i]  = (1 - alpha)     * state.var2[i] + alpha * (x - prev_mu)**2
        vols.append(state.var2[i])                # plain variance, not 1/(2·var)

    # --- 2. windowed mean / coeff of var ----------------------------------------
    state.queue.append(x)
    lam_mean = sum(state.queue) / len(state.queue)
    lam_var  = (x - lam_mean)**2                     # per-step squared deviation
    lam_cv   = lam_var / (lam_mean + 1e-12)

    state.lam_var_queue.append(float(lam_var))
    lam_var_variance = np.var(state.lam_var_queue) if len(state.lam_var_queue) > 1 else 0.0
    lam_var_mean = sum(state.lam_var_queue) / len(state.lam_var_queue)
    lam_var_cv = lam_var_variance / (lam_var_mean + 1e-12)

    # --- 3. variation metric & its EMA ------------------------------------------
    variation = x**2 / lam_mean if lam_mean > 0 else 0.0
    if state.ema_variation is None:
        state.ema_variation = variation
    else:
        state.ema_variation = smoothing*state.ema_variation + (1-smoothing)*variation

    # --- 4. collapse-bound heuristics -------------------------------------------
    # Lyapunov-style test: effective_lr > 2 / ⟨variation⟩ₑₘₐ
    collapse_pred = float(effective_lr > (2.0 / (state.ema_variation + 1e-12)))
    # Same inequality but rearranged the way you had it:
    collapse_pred2 = float((2 / effective_lr) < (lam_mean + lam_cv))
    tau_t = 2.0*lam_mean / (lam_var + lam_mean**2 + 1e-12)

    # --- 5. package what your wandb logging needs --------------------------------
    scalars = {
        "norm":                x,
        "lam_mean":            lam_mean,
        "lam_var":             lam_var,
        "lam_cv":              lam_cv,
        "ema_variation_inv":   state.ema_variation,
        "lam_var_cv":          lam_var_cv,
        "lam_var_mean":        lam_var_mean,
        "sq_mean":             x**2 / (lam_mean + 1e-12) if lam_mean > 0 else 0.0,
        # variances with your original 1/(2·var) transform:
        "vol_1":               vols[0],
        "vol_2":               vols[1],
        "vol_3":               vols[2],
        "tau":                 tau_t,
        "collapse_pred":       collapse_pred,
        "collapse_pred2":      collapse_pred2,
    }
    return state, scalars