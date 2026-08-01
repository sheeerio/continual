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
from torchvision import transforms
from collections import deque
from config import get_parser, validate_reg_config, validate_cbp_config
from utils import misc, schedulers, optimizers
from models import mlp, cnn
from datasets import data_loader
from utils.optimizers import PerLayerLyapunovScheduler

parser = get_parser()
config = parser.parse_args()
validate_reg_config(config)
validate_cbp_config(config)

# Provenance: every results row records the exact code version that produced
# it, so a result is never attributed to a commit by run name or file mtime.
# Resolved once at startup against the script's OWN directory -- SLURM jobs
# don't reliably cwd into the repo, so `git rev-parse` on the cwd would either
# fail or (worse) resolve some unrelated repo. A modified tracked file gets a
# "-dirty" suffix so an uncommitted run is visibly flagged rather than
# silently credited to the clean commit it diverged from.
def _capture_git_sha():
    import subprocess
    repo = os.path.dirname(os.path.abspath(__file__))
    def _git(*args):
        return subprocess.check_output(
            ["git"] + list(args), cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
    try:
        sha = _git("rev-parse", "HEAD")
    except Exception:
        return "unknown"
    try:
        # --untracked-files=no: the harness leaves scratch .sh/.csv files lying
        # around in the repo, and counting those as dirty would mark every run
        # dirty forever, which destroys the signal. Tracked modifications only.
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except Exception:
        return sha + "-dirtyunknown"
    return sha + "-dirty" if dirty else sha

GIT_SHA = _capture_git_sha()

# Full argparse namespace as one JSON string, so a results row is reproducible
# standalone -- no cross-referencing the launch script to recover which flags
# were actually in effect. default=str keeps any non-JSON-able value (paths,
# enums) from killing the run at the very last write.
import json as _json
FULL_CONFIG_JSON = _json.dumps(vars(config), sort_keys=True, default=str)

# Diagnostic CSV sink: tau, sharpness, eff_lr, alpha_crit_*, coherence are
# all computed every log_interval step but were previously only reachable
# via wandb.log -- with WANDB_MODE=disabled (the default across the sbatch
# harness) every one of those values is silently discarded, nowhere on
# disk. This mirrors those same values into a local CSV, flushed after
# every write so a SLURM timeout (as happened to the ortho reruns) loses
# at most one row, not the whole run.
DIAG_CSV_FIELDS = [
    "task", "total_updates", "layer",
    "sharp", "mu", "tau", "cv", "eff_lr", "predict",
    # alpha_crit_t is the quantity --param t actually feeds the pl_lyapunov
    # controller as lr_star, and it was missing from this schema -- the one
    # alpha_crit variant that was not logged was the one the scheduler uses.
    # sigma2 is its denominator, logged alongside so the ratio is auditable.
    "alpha_crit_t", "sigma2",
    # Within-task time series for the sigma2 lead/lag question: does sigma2
    # fall BEFORE task accuracy, or do they co-decline? total_updates is
    # already the within-task step index (it resets each task), so these two
    # columns are all that was missing to resolve it at log_interval cadence.
    "g_sq", "batch_acc",
    # lam and alpha_agg separately: norm_lam is their product, so storing only
    # the product makes curvature-side volatility indistinguishable from
    # optimizer-state-side volatility after the fact.
    "lam_raw", "alpha_agg",
    # Near-zero curvature events: a layer whose top Hessian eigenvalue collapses
    # to ~0 drives tau -> 0, hence g -> 1, hence the saturating factor to its
    # MAXIMUM sensitivity*(1+kappa). A flat layer receives maximum
    # regularization. Recorded whenever it happens, not at log_interval.
    "event", "grad_norm", "weight_norm", "dead_frac",
    "alpha_crit_scv1", "alpha_crit_ss1", "alpha_crit_svar1",
    "alpha_crit_scv10", "alpha_crit_ss10", "alpha_crit_svar10",
    "alpha_crit_sqm1", "alpha_crit_sqm10",
    "alpha_crit_rcv1", "alpha_crit_rs1", "alpha_crit_rvar1", "alpha_crit_rsqm1",
    "alpha_crit_rcv10", "alpha_crit_rs10", "alpha_crit_rvar10", "alpha_crit_rsqm10",
    "reg",
    "mean_off_diag", "max_eigval", "min_eigval", "coherence_ratio",
]
_diag_csv_fh = None
_diag_csv_writer = None
if getattr(config, "diag_csv", None):
    import csv as _diag_csv_module
    _diag_exists = os.path.exists(config.diag_csv) and os.path.getsize(config.diag_csv) > 0
    _diag_csv_fh = open(config.diag_csv, "a", newline="")
    _diag_csv_writer = _diag_csv_module.DictWriter(_diag_csv_fh, fieldnames=DIAG_CSV_FIELDS)
    if not _diag_exists:
        _diag_csv_writer.writeheader()
        _diag_csv_fh.flush()

def diag_log(row):
    # Per-step series are a --diagnostics full feature; light keeps only the
    # per-task snapshots written by taskdiag_log.
    if _diag_csv_writer is None or not DIAG_FULL:
        return
    _diag_csv_writer.writerow({k: row.get(k, "") for k in DIAG_CSV_FIELDS})
    _diag_csv_fh.flush()

# Per-task diagnostic sink. The per-log-step diag_csv above is the firehose;
# this is the one-row-per-(task, layer) end-of-task summary, which is the
# granularity the trajectory analysis actually consumes. Long format because
# layer count varies by model -- wide layer-prefixed columns would give a
# different header per --model and break appends into a shared CSV.
TASKDIAG_CSV_FIELDS = [
    "task", "layer", "git_sha", "seed", "model", "dataset", "reg",
    "adaptive_reg", "adaptive_type", "adaptive_scale", "reg_sensitivity",
    "lr", "lr_schedule", "sched_param", "task_acc", "n_log_steps",
    # per-layer
    "tau_mean", "tau_final", "adaptive_factor_mean", "adaptive_factor_final",
    # Within-task std of this layer's adaptive_factor over time. This is what
    # --sat_kappa is supposed to control under --adaptive_scale centered
    # (spread, not level), so it has to be recorded to check that it does.
    # The across-LAYER std is recoverable from the per-layer means at analysis
    # time; this is the over-TIME component that isn't.
    "adaptive_factor_std",
    "sigma2_final",
    # Per-layer effective LR (per_layer_effective_lr). This is the Lyapunov
    # controllers' control variable -- it was recorded NOWHERE under
    # --diagnostics light, which is why the scheduler's inactivity could not
    # be diagnosed from the 93-cell grid. Costs nothing extra: both eff_lr
    # quantities are already computed in the logging block, so this only
    # accumulates values that were being discarded.
    "eff_lr_layer_mean", "eff_lr_layer_min", "eff_lr_layer_max",
    # run-level, repeated on each layer row so no join is needed to plot them
    "sharp_norm_mean", "sharp_norm_final",
    "eff_lr_agg_mean", "eff_lr_agg_min", "eff_lr_agg_max",
    "mean_off_diag_mean", "mean_off_diag_final",
]
_taskdiag_csv_fh = None
_taskdiag_csv_writer = None
if getattr(config, "taskdiag_csv", None):
    import csv as _taskdiag_csv_module
    _taskdiag_exists = (os.path.exists(config.taskdiag_csv)
                        and os.path.getsize(config.taskdiag_csv) > 0)
    _taskdiag_csv_fh = open(config.taskdiag_csv, "a", newline="")
    _taskdiag_csv_writer = _taskdiag_csv_module.DictWriter(
        _taskdiag_csv_fh, fieldnames=TASKDIAG_CSV_FIELDS)
    if not _taskdiag_exists:
        _taskdiag_csv_writer.writeheader()
        _taskdiag_csv_fh.flush()

def event_log(row):
    """Near-zero-curvature events bypass the --diagnostics full gate: they are
    rare, cheap, and are the plasticity-loss signal itself."""
    if _diag_csv_writer is None:
        return
    _diag_csv_writer.writerow({k: row.get(k, "") for k in DIAG_CSV_FIELDS})
    _diag_csv_fh.flush()


def taskdiag_log(row):
    if _taskdiag_csv_writer is None:
        return
    _taskdiag_csv_writer.writerow({k: row.get(k, "") for k in TASKDIAG_CSV_FIELDS})
    _taskdiag_csv_fh.flush()

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def _fmt_traj(xs):
    # Same ;-joined shape as task_acc_traj so all trajectory columns parse
    # identically downstream. nan renders as "nan", which float() round-trips.
    return ";".join(f"{x:.6g}" for x in xs)

# --diagnostics cost lever. The subtlety: the "diagnostic" block is not
# purely observational. pl_lyapunov's controller input alpha_crit_t is
# (B * g_layer_sq) / sigma2_l, and sigma2_l comes from
# grad_variance_within_batch_by_layer -- so for --lr_schedule pl_lyapunov
# those quantities are METHOD, not diagnostics, and skipping them would
# silently change what the scheduler does rather than just stop recording.
# DIAG_NEED_SCHED keeps that path alive under `off`.
DIAG_MODE = getattr(config, "diagnostics", "full")
DIAG_FULL = (DIAG_MODE == "full")          # per-step series written
DIAG_ANY = DIAG_MODE in ("light", "full")  # expensive estimates run at all
DIAG_NEED_SCHED = (getattr(config, "lr_schedule", "constant") == "pl_lyapunov")

# --adaptive_scale centered clamp accounting. Reported on the results row so a
# run where the floor engaged often is visible rather than silently truncated.
centered_clamp_hits = 0
centered_factor_evals = 0

# Per-layer top-eigenvector cache for --hessian_warm_start. Empty dict means
# every call cold-starts, which is the default.
hessian_v_cache = {}
# tau_ref state: frozen per-layer reference for --tau_ref_mode fixed, its
# calibration accumulator, and the running mean of g used by `centered`.
tau_ref_frozen = {}
tau_ref_calib = {}
g_mean_state = {}
# --tau_update_interval: last computed lam per layer, held between updates.
tau_lam_cache = {}

# Cost profiler: COST_PROFILE=1 wraps the expensive call sites with
# cuda-synchronized timers. Off by default -- the synchronize alone would
# distort a normal run. Sections are reported at exit against the wall total,
# so "everything else" is a residual rather than another estimate.
_PROF = os.environ.get("COST_PROFILE") == "1"
_prof_t, _prof_n = {}, {}
import contextlib as _ctx, time as _time


@_ctx.contextmanager
def prof(name):
    if not _PROF:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _t0 = _time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _prof_t[name] = _prof_t.get(name, 0.0) + (_time.time() - _t0)
        _prof_n[name] = _prof_n.get(name, 0) + 1


_PROF_START = _time.time()


def _timed_iter(ld):
    """Times dataloader wait separately from compute (Task 3b)."""
    if not _PROF:
        for _b in ld:
            yield _b
        return
    _it = iter(ld)
    while True:
        _t0 = _time.time()
        try:
            _b = next(_it)
        except StopIteration:
            _prof_t["dataloader_wait"] = _prof_t.get("dataloader_wait", 0.0) + (_time.time() - _t0)
            return
        _prof_t["dataloader_wait"] = _prof_t.get("dataloader_wait", 0.0) + (_time.time() - _t0)
        _prof_n["dataloader_wait"] = _prof_n.get("dataloader_wait", 0) + 1
        yield _b

RUN_DIAG = os.environ.get('RUN_DIAG','0') == '1'
from collections import deque as _dq
tau_ref_hist = {}
task_acc_history = []
# Per-task trajectories of the layer-collapsed diagnostics -- the ;-joined
# columns on results_csv, one summary statistic per metric per task.
taskdiag_history = {k: [] for k in (
    "tau_mean", "tau_final", "adaptive_factor_mean", "adaptive_factor_final",
    "sigma2_final", "sharp_norm_mean", "sharp_norm_final",
    "mean_off_diag_mean", "mean_off_diag_final",
)}
if not hasattr(config, "snr_margin"):       config.snr_margin = 0.0
if not hasattr(config, "snr_pred_window"):  config.snr_pred_window = 20
if config.diag_interval is None:            config.diag_interval = config.log_interval
LENGTH_CHOICES = [100, 300, 50, 150]

# Caches for the expensive diagnostics (hessian topk/min_eig at iters=100/20,
# per-sample grad variance at batch_size backprops, Fisher rank at
# diag_task_interval-gated per-task backprops). These previously ran on
# every config.log_interval firing (or every task, for Fisher rank)
# unconditionally, cumulatively costing more backward passes than the
# actual training loop over a full run. Now gated behind the coarser
# diag_interval/diag_task_interval, reusing the last computed value in
# between -- same cache-and-reuse pattern already used for step_stats reuse
# and the ortho fix's cached eigenvector.
sharpness = 0.0
lambda_min = 0.0
layer_sigma2_mb = {}
# Assigned inside the gated diagnostic block but read outside it (eta_eff,
# and the adaptive block's global scope). Needs a defined value for
# --diagnostics off, where that block never runs.
effective_lr = config.lr
hessian_rank = None

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
hidden = config.hidden
if config.model == "MLP":
    model = mlp.MLP(input_size, hidden, 10).to(device)
elif config.model == "BatchNormMLP":
    model = mlp.BatchNormMLP(input_size, hidden, 10).to(device)
elif config.model == "LayerNormMLP":
    model = mlp.LayerNormMLP(input_size, hidden, 10).to(device)
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
            layer = name.split('.')[0]
            layer_map.setdefault(layer, []).append(p)
    layer_groups = [{'params': params, 'lr': base_lr, 'layer': layer}
                    for layer, params in layer_map.items()]
    return layer_map, layer_groups
layer_map, layer_groups = make_layer_groups(model, config.lr)
if config.track_coherence:
    coherence_tracker = optimizers.CrossLayerCoherence(
        layer_names=list(layer_map.keys()),
        window=config.coherence_window
    )
else:
    coherence_tracker = None
coh = None
# ── one EMAState per layer ─────────────────────────────────────────────
layer_states = {
    layer : misc.EMAState(alphas=(0.01, 0.05, 0.5))   # same alphas you used globally
    for layer in layer_map
}
act_layer_states = {
    layer : misc.EMAState(alphas=(0.01, 0.05, 0.5))   # same alphas you used globally
    for layer in layer_map
}

criterion = nn.CrossEntropyLoss(reduction="mean")
criterion_nored = nn.CrossEntropyLoss(reduction="none")  # for within-batch σ² only
wd = config.l2_lambda if config.reg == "l2" else 0.0
if config.optimizer == "adam":
    optimizer = optim.Adam(layer_groups, lr=config.lr, weight_decay=wd, betas=(0.9, config.beta2))
elif config.optimizer == "sgd": 
    optimizer = torch.optim.SGD(layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9)
elif config.optimizer == "clamped_adam":
    optimizer = optimizers.ClampedAdam(layer_groups, lr=config.lr, lr_min=1e-3, lr_max=1.1)

# Continual Backprop: architectural intervention (per-unit utility tracking +
# low-utility reinit), NOT a --reg branch -- no loss term, not part of the
# coefficient-axis sweep. Scope enforced by validate_cbp_config: MLP-family
# models only, non-concatenating activations only (see ContinualBackprop
# docstring in utils/optimizers.py).
cbp_tracker = None
if config.use_cbp:
    cbp_layer_specs = [("fc1", model.fc1, model.fc2), ("fc2", model.fc2, model.fc4)]
    cbp_tracker = optimizers.ContinualBackprop(cbp_layer_specs, config, optimizer=optimizer)

def cbp_post_activation(pre_act, config):
    """Reconstruct the post-activation value from the pre-activation hook
    output (activations["l1"]/["l2"] capture the raw Linear output), using
    the SAME nonlinearity models/mlp.py's forward() applies -- scope is
    already restricted to non-concatenating activations by validate_cbp_config."""
    if config.activation == "relu":
        return F.relu(pre_act)
    elif config.activation == "leaky_relu":
        return F.leaky_relu(pre_act)
    elif config.activation == "tanh":
        return torch.tanh(pre_act)
    elif config.activation == "adalin":
        return F.leaky_relu(pre_act, negative_slope=config.alpha)
    elif config.activation == "softplus":
        return F.softplus(pre_act)
    elif config.activation == "swish":
        return pre_act * torch.sigmoid(pre_act)
    else:
        return pre_act
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
    project=f"sweep",
    entity="sheerio",
    group=config.exp_name,
    name=f"{config.name}",
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
# SNR metrics
wandb.define_metric("snr_T", hidden=False)
wandb.define_metric("snr_sigma2_hat", hidden=False)
wandb.define_metric("mb_sigma2_hat", hidden=False)
wandb.define_metric("mb_snr_T",      hidden=False)

# tracker and series for plotting later
snr_tracker = optimizers.GradSNR()
snr_T_series = []          # list of (update_idx, T_t)
snr_predictor = optimizers.SNRProgressPredictor(
    margin=config.snr_margin, window=config.snr_pred_window
)
wandb.define_metric("snr_pred",        hidden=False)
wandb.define_metric("snr_pred_conf",   hidden=True)
wandb.define_metric("snr_T_mean",      hidden=True)
wandb.define_metric("snr_T_thresh",    hidden=True)

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
r_sharp_state = misc.EMAState(alphas=(0.01,0.05,0.5))
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

task_lengths = [300, 50, 300, 300, 100] * 4
ns = [1] + 9 * [0]
for task in range(config.runs):
    # Per-task diagnostic accumulators. Reset every task -- these are
    # end-of-task snapshots, not running-since-step-0 values. Sampled at
    # log_interval (not every minibatch) so they line up with the steps where
    # the diagnostics are actually computed rather than reusing stale caches.
    td_tau_sum, td_tau_n, td_tau_last = {}, {}, {}
    td_af_sum, td_af_n, td_af_last = {}, {}, {}
    td_af_sq = {}          # sum of squares, for the within-task std
    td_sharp_sum, td_sharp_n, td_sharp_last = 0.0, 0, float("nan")
    td_mod_sum, td_mod_n, td_mod_last = 0.0, 0, float("nan")
    td_log_steps = 0
    # eff_lr: mean/min/max over the task, aggregate and per layer.
    td_elr_sum, td_elr_n = {}, {}
    td_elr_min, td_elr_max = {}, {}
    td_eagg_sum, td_eagg_n = 0.0, 0
    td_eagg_min, td_eagg_max = float("inf"), float("-inf")
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
            optimizer = optim.Adam(layer_groups, lr=config.lr, weight_decay=wd, betas=(config.beta1, config.beta2))
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9)
    if config.reset_optimizer:
        layer_map, layer_groups = make_layer_groups(model, config.lr)
        # Recreate optimizer using layer_groups (not model.parameters())
        if config.optimizer == "adam":
            optimizer = optim.Adam(layer_groups, lr=config.lr, weight_decay=wd, betas=(config.beta1, config.beta2))
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(layer_groups, lr=config.lr, weight_decay=wd, momentum=0.9)
        elif config.optimizer == "clamped_adam":
            optimizer = optimizers.ClampedAdam(layer_groups, lr=config.lr, lr_min=1e-2, lr_max=0.8)

    # shrink perturb: applied ONCE per task boundary (Ash & Adams 2020),
    # not per step. Task 0 is a fresh init, not a warm start, so it's exempt.
    if task >= 1 and config.reg == "shrink_perturb":
        for p in model.parameters():
            if p.requires_grad:
                p.data.mul_(1.0 - config.sp_weight_decay)
                p.data.add_(config.sp_noise_std * torch.randn_like(p.data))

    if config.dataset == "PermutedMNIST":
        perm = train_dataset.perms[task]
        perm_tf = transforms.Lambda(lambda x, perm=perm: x.view(-1)[perm].view(1, 28, 28))
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
        with prof("randomize_targets"):
            train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
        # if task == 0:
        #     train_dataset = optimizers.randomize_targets(train_dataset, 0.0)
        # -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
        loader = data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )

    total_updates = 0
    # Fisher rank costs max_m (default 100) single-sample backprops; this
    # condition previously looked gated (total_updates % log_interval == 0)
    # but total_updates was just reset to 0, so it always evaluated True --
    # unconditionally once per task. Gate by task instead, so it can
    # actually be throttled; reuse the cached value on skipped tasks.
    if DIAG_ANY and (task % config.diag_task_interval == 0 or hessian_rank is None):
        with prof("fisher_rank"):
            hessian_rank = optimizers.empirical_fischer_rank(model, train_dataset, device, cfg=config)

    model.train()
    ly_sched = None
    scheduler = None
    total_steps = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
    # Under --diagnostics light the expensive estimates (iters=100 hessian
    # topk, min_eig, per-sample grad variance) run on a coarse ~10-per-task
    # grid instead of every diag_interval: enough samples that the per-task
    # mean/final snapshot means something, without paying the full per-step
    # series cost. full/off keep the explicit --diag_interval.
    if DIAG_MODE == "light":
        _eff_diag_interval = max(config.diag_interval, total_steps // 10, 1)
    else:
        _eff_diag_interval = config.diag_interval
    # sigma2 is a diagnostic for most arms but a CONTROLLER INPUT under
    # pl_lyapunov (alpha_crit_t = B*|g|^2 / sigma2). If its refresh cadence
    # followed the coarse `light` grid, the scheduler would see a staler
    # sigma2 under light than under full and the two modes would train
    # differently -- making --diagnostics a silent method knob. Pin it to the
    # explicit --diag_interval whenever the scheduler consumes it.
    _sigma2_interval = config.diag_interval if DIAG_NEED_SCHED else _eff_diag_interval
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
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.gamma
        )
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
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.wsd_lambda)
    elif config.lr_schedule == "power":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.power_lambda)
    elif config.lr_schedule == "skew":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers.skew_lambda)
    elif config.lr_schedule == "lyapunov":
        # hyper-params exposed to the CLI for convenience
        ly_safety = config.safety
        ly_cool   = config.cool
        ly_warm   = config.warm
        ly_sched  = optimizers.LyapunovScheduler(optimizer,
                                    ema_state = sharp_state,   # <<– share!
                                    safety    = ly_safety,
                                    cool      = ly_cool,
                                    warm      = ly_warm,
                                    cfg = config)
        scheduler = None
    elif config.lr_schedule == "pl_lyapunov":
        pl_scheduler = PerLayerLyapunovScheduler(
            optimizer    = optimizer,
            layer_states = {layer: layer_states.get(layer, misc.EMAState(alphas=(0.01,0.05,0.5)))
                            for layer in layer_map},
            safety       = config.safety,
            cool         = config.cool,
            warm         = config.warm,
            cfg = config
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
        for x, y in _timed_iter(loader):
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
            elif config.reg == "l2_loss":
                # Honest loss-based L2 toward zero, differentiated through the
                # loss like every other regularizer here. Distinct from
                # --reg l2, which instead routes through the optimizer's
                # decoupled weight_decay (Adam: NOT equivalent to this).
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += p.pow(2).sum()
                reg *= config.l2_lambda
            elif config.reg == "wass":
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += (torch.sort(p.view(-1))[0] - init_params[n]).pow(2).sum()
                reg *= config.wass_lambda
            elif config.reg == "spectral":
                # EVERY step, EVERY weight matrix -- this is in the loss, not the
                # diagnostics, so it was never wrapped.
                with prof("spectral_power_iteration"):
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.ndim >= 2:
                            reg += (optimizers.power_iteration(p, 1).pow(config.spectral_k) - 1.0).pow(2)
                    reg *= config.spectral_lambda
            elif config.reg == "ortho":
                frac = config.ortho_frac
                for name, p in model.named_parameters():
                    if p.ndim < 2 or not p.requires_grad:
                        continue
                    # Expensive numerics (the eigenvector direction) are
                    # amortized over ortho_interval steps and always
                    # detached. The cheap quadratic form that actually
                    # carries gradient is recomputed fresh EVERY step from
                    # the current p -- caching a scalar here (as before)
                    # makes it a constant and kills the gradient again.
                    if total_updates % config.ortho_interval == 0:
                        v, use_WtW = optimizers.power_iteration_min_sv_vector(p, iters=1)
                        cached_sigma_min[name] = (v, use_WtW)
                    else:
                        v, use_WtW = cached_sigma_min.get(name, (None, None))
                        if v is None:
                            v, use_WtW = optimizers.power_iteration_min_sv_vector(p, iters=1)
                            cached_sigma_min[name] = (v, use_WtW)
                    sigma_now = optimizers.sigma_min_from_vector(p, v, use_WtW)
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
            elif config.reg == "parseval":
                # Parseval Networks (Cisse et al. 2017): push each weight
                # matrix toward a Parseval tight frame, ||W W^T - I|| (or
                # ||W^T W - I|| if that's the smaller/achievable identity).
                # d_out <= d_in is the common case here (e.g. fc1: 256x784,
                # fc4: 10x256), where only W @ W.T = I_{d_out} is reachable;
                # W.T @ W = I_{d_in} never is once rank(W) < d_in.
                for name, p in model.named_parameters():
                    if p.ndim >= 2 and p.requires_grad:
                        W = p.view(p.shape[0], -1)
                        d_out, d_in = W.shape
                        if d_out <= d_in:
                            I = torch.eye(d_out, device=W.device, dtype=W.dtype)
                            reg += (W @ W.t() - I).pow(2).sum()
                        else:
                            I = torch.eye(d_in, device=W.device, dtype=W.dtype)
                            reg += (W.t() @ W - I).pow(2).sum()
                reg *= config.parseval_lambda

            step_stats = {}

            if getattr(config, "adaptive_reg", False):
                # Configuration Defaults
                reg_scope = getattr(config, "adaptive_scope", "local") # "global" or "local"
                reg_type  = getattr(config, "adaptive_type", "l2")     # "l2" or "spectral"
                sensitivity = getattr(config, "reg_sensitivity", 0.001)

                # --- A. CALCULATE GLOBAL FACTOR (IF GLOBAL MODE) ---
                global_factor = None
                if reg_scope == "global":
                    # Estimate global sharpness on Task Loss (base)
                    # We use the existing global 'sharp_state' defined in your setup
                    global_lam = optimizers.estimate_hessian_topk(model, base, params, k=1, iters=1)[0]
                    global_norm_lam = optimizers.get_norm_sharpness(optimizer, global_lam, config)
                    global_eff_lr = effective_lr # Calculated earlier in loop or re-calc here
                    
                    # Update global EMA state
                    sharp_state, global_scalars = misc.update_stat(global_norm_lam, sharp_state, global_eff_lr)
                    
                    # Calculate Global Penalty Factor
                    g_tau = global_scalars["tau"]
                    g_inv_tau = 1.0 / (g_tau + 1e-12)
                    global_factor = sensitivity * g_inv_tau
                    
                    if total_updates % config.log_interval == 0:
                        wandb.log({"global/inv_tau_penalty": global_factor}, commit=False)

                # --- B. ITERATE LAYERS ---
                if config.optimizer == "adam":
                    tmp_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                else: 
                    tmp_eff_lrs = optimizers.per_layer_sgd_lr(model, optimizer, step=total_updates)

                for layer, l_params in layer_map.items():
                    # 1. Calculate Stats (Always needed for logging/local reg)
                    # Same estimator as the diagnostic path: residual-criterion
                    # stopping, warm-started from the previous step's
                    # eigenvector. The old iters=1 cold call biased lam low by
                    # 1.5-2.8x with a layer-dependent factor, and tau is
                    # computed from lam.
                    # --tau_update_interval: recompute every N steps, hold the
                    # cached lam (and hence adaptive_factor) in between. tau is
                    # an EMA over ~30 steps and tau_ref a rolling median, so
                    # both already smooth; a fresh estimate every step may be
                    # redundant. --curvature_proxy selects lam1 vs top-3 mean.
                    _N = max(1, getattr(config, "tau_update_interval", 1))
                    if total_updates % _N == 0 or layer not in tau_lam_cache:
                        _k = 3 if getattr(config, "curvature_proxy", "lam1") == "top3" else 1
                        with prof("est_adaptive_path"):
                            _eigs, _vs = optimizers.estimate_hessian_topk(
                                model, base, l_params, k=_k,
                                iters=config.hessian_max_iters,
                                tol=config.hessian_tol,
                                v_init=hessian_v_cache.get(layer), return_v=True)
                        hessian_v_cache[layer] = _vs
                        lam = sum(_eigs) / len(_eigs)
                        tau_lam_cache[layer] = lam
                    else:
                        lam = tau_lam_cache[layer]

                    # Near-zero curvature event. Not an exception: tau -> 0 makes
                    # g -> 1 and drives the saturating factor to its maximum, so
                    # a flat layer gets maximum regularization. Recorded every
                    # time it fires, independent of --diagnostics.
                    if abs(float(lam)) < 1e-10:
                        try:
                            _g = torch.autograd.grad(base, l_params, retain_graph=True,
                                                     allow_unused=True)
                            _gn = float(torch.cat([q.contiguous().view(-1)
                                                   for q in _g if q is not None]).norm())
                        except Exception:
                            _gn = float("nan")
                        _wn = float(sum(float(q.data.norm()) ** 2
                                        for q in l_params) ** 0.5)
                        _act = {"fc1": "l1", "fc2": "l2"}.get(layer)
                        if _act is not None and _act in activations:
                            _a = activations[_act]
                            _df = float((_a <= 0).float().mean())
                        else:
                            _df = float("nan")
                        event_log({"task": task, "total_updates": total_updates,
                                   "layer": layer, "event": "near_zero_lam",
                                   "lam_raw": float(lam), "grad_norm": _gn,
                                   "weight_norm": _wn, "dead_frac": _df})
                    norm_lam = optimizers.get_norm_sharpness(optimizer, lam, config)
                    l_eff_lr = tmp_eff_lrs.get(layer, optimizer.param_groups[0]["lr"])
                    
                    # Update Local State
                    state, scalars = misc.update_stat(norm_lam, layer_states[layer], l_eff_lr)
                    act_state, act_scalars = misc.update_stat(lam, act_layer_states[layer], l_eff_lr)

                    # Cache for logging loop
                    step_stats[layer] = {
                        "scalars": scalars, "act_scalars": act_scalars,
                        "lam": lam, "norm_lam": norm_lam, "eff_lr": l_eff_lr
                    }

                    # 2. Determine Adaptive Factor
                    if reg_scope == "global":
                        adaptive_factor = global_factor
                    else:
                        # Local Mode: Use this layer's specific volatility
                        tau = scalars["tau"]
                        _scale = getattr(config, "adaptive_scale", "inv")
                        if _scale in ("saturating", "centered"):
                            # tau_ref reference mode. Under `median`, tau_ref is
                            # the rolling median of this layer's OWN tau, so any
                            # uniform rescale of tau cancels exactly in
                            # g = tau_ref/(tau+tau_ref) -- which is why the
                            # corrected estimator halved tau while
                            # adaptive_factor moved 0.2%. `fixed` freezes the
                            # reference so g can see drift slower than the
                            # window (a task is ~2100 steps vs a 100-step median).
                            _ref_mode = getattr(config, "tau_ref_mode", "median")
                            if _ref_mode == "fixed" and layer in tau_ref_frozen:
                                tau_ref = tau_ref_frozen[layer]
                            else:
                                h = tau_ref_hist.get(layer)
                                if h is None:
                                    h = _dq(maxlen=getattr(config, "tau_ref_window", 100))
                                    tau_ref_hist[layer] = h
                                h.append(float(tau))
                                _s = sorted(h); _n = len(_s)
                                tau_ref = _s[_n//2] if _n%2 else 0.5*(_s[_n//2-1]+_s[_n//2])
                            tau_ref = max(tau_ref, 1e-12)
                            # Calibration accumulator for `fixed`: mean tau over
                            # the last 20% of task 0, frozen at that boundary.
                            if _ref_mode == "fixed" and layer not in tau_ref_frozen \
                                    and task == 0 and total_updates >= 0.8 * total_steps:
                                tau_ref_calib.setdefault(layer, []).append(float(tau))
                            g = tau_ref / (tau + tau_ref + 1e-12)
                            # Running mean of g, for the centered form.
                            _gm = g_mean_state.get(layer)
                            g_mean_state[layer] = (g if _gm is None
                                                   else 0.99 * _gm + 0.01 * g)
                            if _scale == "centered":
                                # g is centered at ~0.5 by construction (tau_ref
                                # is the rolling median of this layer's own tau),
                                # so `saturating` multiplies sensitivity by
                                # ~(1 + kappa/2) -- a near-constant rescale that
                                # moves the basin's LOCATION with kappa. Centering
                                # on (g - 0.5) makes the mean factor ~sensitivity
                                # for any kappa, so kappa controls only the SPREAD
                                # of the coefficient across layers and time.
                                # Subtract the RUNNING MEAN of g, not a hardcoded
                                # 0.5: measured g_mean is 0.42-0.45, so (g - 0.5)
                                # has mean 1 - 0.07*kappa and clamps negative near
                                # kappa ~ 14, reintroducing exactly the level shift
                                # centering exists to remove.
                                _gc = g_mean_state.get(layer, 0.5)
                                adaptive_factor = sensitivity * (1.0 + config.sat_kappa * (g - _gc))
                                # Large kappa can drive the bracket negative; floor
                                # it well below any coefficient on the axis rather
                                # than flipping the penalty's sign.
                                _floor = 1e-8 * sensitivity
                                if adaptive_factor < _floor:
                                    adaptive_factor = _floor
                                    centered_clamp_hits += 1
                                centered_factor_evals += 1
                            else:
                                adaptive_factor = sensitivity * (1.0 + config.sat_kappa * g)
                        else:
                            adaptive_factor = sensitivity * (1.0 / (tau + 1e-12))

                    # Hand the factor to the logging loop, which is what
                    # samples it into the per-task accumulators at
                    # log_interval. Computed every step here, but recording it
                    # every step would weight the mean by minibatch count
                    # rather than by the diagnostic sampling grid tau uses.
                    step_stats[layer]["adaptive_factor"] = float(adaptive_factor)

                    # 3. Apply Penalty (L2 or Spectral)
                    layer_reg_val = torch.tensor(0.0, device=device)
                    
                    if reg_type == "spectral":
                        # Spectral Loss: (sigma^k - 1)^2
                        for p in l_params:
                            if p.ndim >= 2 and p.requires_grad:
                                sigma = optimizers.power_iteration(p, iters=1)
                                layer_reg_val += (sigma.pow(config.spectral_k) - 1.0).pow(2)
                    elif reg_type == "wass":
                        # Adaptive Wasserstein: (sort(p) - sort(p0))^2
                        # Matches your paper's static implementation but scaled dynamically
                        for p in l_params:
                            if p.requires_grad:
                                name = id_to_name.get(id(p))
                                if name in init_params:
                                    p_sorted = torch.sort(p.view(-1))[0]
                                    p0 = init_params[name]
                                    if config.reg != "wass":
                                        p0 = torch.sort(p0.view(-1))[0]
                                        
                                    layer_reg_val += (p_sorted - p0).pow(2).sum()
                    else: 
                        # Default L2 Loss
                        layer_reg_val = sum(p.pow(2).sum() for p in l_params if p.requires_grad)
                    reg += adaptive_factor * layer_reg_val

                    # Log factor occasionally
                    if total_updates % config.log_interval == 0:
                        wandb.log({f"{layer}/reg_factor": adaptive_factor}, commit=False)

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

            if (DIAG_ANY or DIAG_NEED_SCHED) and total_updates % config.log_interval == 0:
                if config.optimizer == "adam":
                    layer_eff_lrs = optimizers.per_layer_effective_lr(model, optimizer)
                else:  # SGD (+ momentum)
                    layer_eff_lrs = optimizers.per_layer_sgd_lr(
                        model, optimizer, step=total_updates, sgd_mode="time"
                    )


                # Costs one backward pass per sample in the batch (256 by
                # default). Gate separately from the cheap per-log-interval
                # stuff above; reuse the last computed dict in between.
                # Still required under --diagnostics off when pl_lyapunov is
                # driving, since alpha_crit_t divides by this sigma2.
                if total_updates % _sigma2_interval == 0:
                    with prof("sigma2_grad_variance"):
                        layer_sigma2_mb = optimizers.grad_variance_within_batch_by_layer(
                            model, criterion_nored, inputs, labels, layer_map,
                            subsample=getattr(config, "sigma2_subsample", 0)
                        )

                # union across layers for collapse_pred (NO pred2)
                union_pred_step = 0
                union_eff_gt_acrit_step_scv1 = 0
                union_eff_gt_acrit_step_ss1 = 0
                union_eff_gt_acrit_step_svar1 = 0
                union_eff_gt_acrit_step_scv10 = 0
                union_eff_gt_acrit_step_ss10 = 0
                union_eff_gt_acrit_step_svar10 = 0
                union_eff_gt_acrit_step_sqm1 = 0
                union_eff_gt_acrit_step_sqm10 = 0
                union_eff_gt_acrit_step_rcv1 = 0
                union_eff_gt_acrit_step_rs1 = 0
                union_eff_gt_acrit_step_rvar1 = 0
                union_eff_gt_acrit_step_rsqm1 = 0
                union_eff_gt_acrit_step_rcv10 = 0
                union_eff_gt_acrit_step_rs10 = 0
                union_eff_gt_acrit_step_rvar10 = 0
                union_eff_gt_acrit_step_rsqm10 = 0

                # l_params, NOT params: `params` is bound above to the FULL
                # model parameter list and is consumed after this loop by
                # estimate_hessian_topk / estimate_hessian_min_eig (the global
                # `sharpness` and `lambda_min`). Rebinding it here silently
                # reduced those two to the last layer's tensors only.
                for layer, l_params in layer_map.items():
                    if layer in step_stats:
                        # REUSE: Don't update state again
                        cached = step_stats[layer]
                        scalars = cached["scalars"]
                        act_scalars = cached["act_scalars"]
                        lam = cached["lam"]
                        norm_lam = cached["norm_lam"]
                        eff_lr = cached["eff_lr"]
                    else:
                        # CALCULATE: Adaptive Reg was OFF, so we must calculate now for logging
                        # Same estimator as the adaptive path. This branch was
                        # missed in the first pass and still ran iters=1, which
                        # is why the static arm's tau was unchanged (0.96-1.02x)
                        # while the adaptive arm's halved.
                        with prof("est_diag_perlayer"):
                            _e, _v = optimizers.estimate_hessian_topk(
                                model, base, l_params, k=1,
                                iters=config.hessian_max_iters, tol=config.hessian_tol,
                                v_init=hessian_v_cache.get(layer), return_v=True)
                        hessian_v_cache[layer] = _v
                        lam = _e[0]
                        norm_lam = optimizers.get_norm_sharpness(optimizer, lam, config)
                        eff_lr = layer_eff_lrs.get(layer, optimizer.param_groups[0]["lr"])
                        state, scalars = misc.update_stat(norm_lam, layer_states[layer], eff_lr)
                        act_state, act_scalars = misc.update_stat(lam, act_layer_states[layer], eff_lr)
                        step_stats[layer] = {
                            "scalars": scalars, "act_scalars": act_scalars,
                            "lam": lam, "norm_lam": norm_lam, "eff_lr": eff_lr
                        }

                    # --- per-task diagnostic accumulation (one sample per
                    # log_interval firing, per layer) ---
                    _td_tau = float(scalars["tau"])
                    td_tau_sum[layer] = td_tau_sum.get(layer, 0.0) + _td_tau
                    td_tau_n[layer] = td_tau_n.get(layer, 0) + 1
                    td_tau_last[layer] = _td_tau
                    _td_elr = float(eff_lr)
                    td_elr_sum[layer] = td_elr_sum.get(layer, 0.0) + _td_elr
                    td_elr_n[layer] = td_elr_n.get(layer, 0) + 1
                    td_elr_min[layer] = min(td_elr_min.get(layer, float("inf")), _td_elr)
                    td_elr_max[layer] = max(td_elr_max.get(layer, float("-inf")), _td_elr)
                    _td_af = step_stats[layer].get("adaptive_factor")
                    if _td_af is not None:
                        td_af_sum[layer] = td_af_sum.get(layer, 0.0) + _td_af
                        td_af_sq[layer] = td_af_sq.get(layer, 0.0) + _td_af * _td_af
                        td_af_n[layer] = td_af_n.get(layer, 0) + 1
                        td_af_last[layer] = _td_af

                    # strictly use collapse_pred for the union
                    layer_pred = int(scalars["collapse_pred2"])
                    union_pred_step = max(union_pred_step, layer_pred)

                    # ---- per-layer alpha_crit_s (unchanged) ----
                    with prof("perlayer_autograd_grad"):
                        gi = torch.autograd.grad(loss, l_params, retain_graph=True, allow_unused=False)
                    g_layer_sq = float(torch.cat([g.contiguous().view(-1) for g in gi]).pow(2).sum().item())

                    sigma2_l = float(layer_sigma2_mb.get(layer, 0.0))
                    s_scv_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * scalars["lam_cv"]
                    s_ss_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * norm_lam
                    s_svar_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * scalars["lam_var"]
                    s_sqm_sigma2_l1 = sigma2_l + 75.0 * g_layer_sq * scalars["sq_mean"]
                    s_scv_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["lam_cv"]
                    s_ss_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * norm_lam
                    s_svar_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["lam_var"]
                    s_sqm_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * scalars["sq_mean"]
                    # actual sharpness metric
                    s_rcv_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * act_scalars["lam_cv"]
                    s_rs_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * lam
                    s_rvar_sigma2_l1 = sigma2_l + 10.0 * g_layer_sq * act_scalars["lam_var"]
                    s_rsqm_sigma2_l1 = sigma2_l + 75.0 * g_layer_sq * act_scalars["sq_mean"]
                    s_rcv_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["lam_cv"]
                    s_rs_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * lam
                    s_rvar_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["lam_var"]
                    s_rsqm_sigma2_l10 = sigma2_l + 20.0 * g_layer_sq * act_scalars["sq_mean"]

                    r = (sigma2_l / max(1, inputs.size(0))) / (g_layer_sq + 1e-12) + 1e-12

                    B = int(inputs.size(0))
                    alpha_crit_t = (B * g_layer_sq) / max(sigma2_l, 1e-12)
                    alpha_crit_scv_layer1 = (B * g_layer_sq) / max(s_scv_sigma2_l1, 1e-12)
                    alpha_crit_ss_layer1 = (B * g_layer_sq) / max(s_ss_sigma2_l1, 1e-12)
                    alpha_crit_svar_layer1 = (B * g_layer_sq) / max(s_svar_sigma2_l1, 1e-12)
                    alpha_crit_sqm_layer1 = (B * g_layer_sq) / max(s_sqm_sigma2_l1, 1e-12)
                    alpha_crit_scv_layer10 = (B * g_layer_sq) / max(s_scv_sigma2_l10, 1e-12)
                    alpha_crit_ss_layer10 = (B * g_layer_sq) / max(s_ss_sigma2_l10, 1e-12)
                    alpha_crit_svar_layer10 = (B * g_layer_sq) / max(s_svar_sigma2_l10, 1e-12)
                    alpha_crit_sqm_layer10 = (B * g_layer_sq) / max(s_sqm_sigma2_l10, 1e-12)
                    alpha_crit_rcv_layer1 = (B * g_layer_sq) / max(s_rcv_sigma2_l1, 1e-12)
                    alpha_crit_rs_layer1 = (B * g_layer_sq) / max(s_rs_sigma2_l1, 1e-12)
                    alpha_crit_rvar_layer1 = (B * g_layer_sq) / max(s_rvar_sigma2_l1, 1e-12)
                    alpha_crit_rsqm_layer1 = (B * g_layer_sq) / max(s_rsqm_sigma2_l1, 1e-12)
                    alpha_crit_rcv_layer10 = (B * g_layer_sq) / max(s_rcv_sigma2_l10, 1e-12)
                    alpha_crit_rs_layer10 = (B * g_layer_sq) / max(s_rs_sigma2_l10, 1e-12)
                    alpha_crit_rvar_layer10 = (B * g_layer_sq) / max(s_rvar_sigma2_l10, 1e-12)
                    alpha_crit_rsqm_layer10 = (B * g_layer_sq) / max(s_rsqm_sigma2_l10, 1e-12)
                    eff_gt_acrit_scv1 = int(eff_lr > alpha_crit_scv_layer1)
                    eff_gt_acrit_ss1 = int(eff_lr > alpha_crit_ss_layer1)
                    eff_gt_acrit_svar1 = int(eff_lr > alpha_crit_svar_layer1)
                    eff_gt_acrit_sqm1 = int(eff_lr > alpha_crit_sqm_layer1)
                    eff_gt_acrit_scv10 = int(eff_lr > alpha_crit_scv_layer10)
                    eff_gt_acrit_ss10 = int(eff_lr > alpha_crit_ss_layer10)
                    eff_gt_acrit_svar10 = int(eff_lr > alpha_crit_svar_layer10)
                    eff_gt_acrit_sqm10 = int(eff_lr > alpha_crit_sqm_layer10)
                    eff_gt_acrit_rcv1 = int(eff_lr > alpha_crit_rcv_layer1)
                    eff_gt_acrit_rs1 = int(eff_lr > alpha_crit_rs_layer1)
                    eff_gt_acrit_rvar1 = int(eff_lr > alpha_crit_rvar_layer1)
                    eff_gt_acrit_rsqm1 = int(eff_lr > alpha_crit_rsqm_layer1)
                    eff_gt_acrit_rcv10 = int(eff_lr > alpha_crit_rcv_layer10)
                    eff_gt_acrit_rs10 = int(eff_lr > alpha_crit_rs_layer10)
                    eff_gt_acrit_rvar10 = int(eff_lr > alpha_crit_rvar_layer10)
                    eff_gt_acrit_rsqm10 = int(eff_lr > alpha_crit_rsqm_layer10)

                    union_eff_gt_acrit_step_scv1 = max(union_eff_gt_acrit_step_scv1, eff_gt_acrit_scv1)
                    union_eff_gt_acrit_step_ss1 = max(union_eff_gt_acrit_step_ss1, eff_gt_acrit_ss1)
                    union_eff_gt_acrit_step_svar1 = max(union_eff_gt_acrit_step_svar1, eff_gt_acrit_svar1)
                    union_eff_gt_acrit_step_sqm1 = max(union_eff_gt_acrit_step_sqm1, eff_gt_acrit_sqm1)
                    union_eff_gt_acrit_step_scv10 = max(union_eff_gt_acrit_step_scv10, eff_gt_acrit_scv10)
                    union_eff_gt_acrit_step_ss10 = max(union_eff_gt_acrit_step_ss10, eff_gt_acrit_ss10)
                    union_eff_gt_acrit_step_svar10 = max(union_eff_gt_acrit_step_svar10, eff_gt_acrit_svar10)
                    union_eff_gt_acrit_step_sqm10 = max(union_eff_gt_acrit_step_sqm10, eff_gt_acrit_sqm10)
                    union_eff_gt_acrit_step_rcv1 = max(union_eff_gt_acrit_step_rcv1, eff_gt_acrit_rcv1)
                    union_eff_gt_acrit_step_rs1 = max(union_eff_gt_acrit_step_rs1, eff_gt_acrit_rs1)
                    union_eff_gt_acrit_step_rvar1 = max(union_eff_gt_acrit_step_rvar1, eff_gt_acrit_rvar1)
                    union_eff_gt_acrit_step_rsqm1 = max(union_eff_gt_acrit_step_rsqm1, eff_gt_acrit_rsqm1)
                    union_eff_gt_acrit_step_rcv10 = max(union_eff_gt_acrit_step_rcv10, eff_gt_acrit_rcv10)
                    union_eff_gt_acrit_step_rs10 = max(union_eff_gt_acrit_step_rs10, eff_gt_acrit_rs10)
                    union_eff_gt_acrit_step_rvar10 = max(union_eff_gt_acrit_step_rvar10, eff_gt_acrit_rvar10)
                    union_eff_gt_acrit_step_rsqm10 = max(union_eff_gt_acrit_step_rsqm10, eff_gt_acrit_rsqm10)


                    wandb.log({
                        f"{layer}/sharp"       : norm_lam,
                        f"{layer}/mu"          : scalars["lam_mean"],
                        f"{layer}/tau"         : scalars["tau"],
                        f"{layer}/cv"          : scalars["lam_cv"],
                        f"{layer}/eff_lr"      : eff_lr,
                        f"{layer}/predict"     : layer_pred,
                        # f"{layer}/alpha_crit_s": float(alpha_crit_s_layer),
                        f"{layer}/alpha_crit_scv1": float(alpha_crit_scv_layer1),
                        f"{layer}/alpha_crit_ss1": float(alpha_crit_ss_layer1),
                        f"{layer}/alpha_crit_svar1": float(alpha_crit_svar_layer1),
                        f"{layer}/alpha_crit_scv10": float(alpha_crit_scv_layer10),
                        f"{layer}/alpha_crit_ss10": float(alpha_crit_ss_layer10),
                        f"{layer}/alpha_crit_svar10": float(alpha_crit_svar_layer10),
                        f"{layer}/alpha_crit_sqm1": float(alpha_crit_sqm_layer1),
                        f"{layer}/alpha_crit_sqm10": float(alpha_crit_sqm_layer10),
                        f"{layer}/alpha_crit_rcv1": float(alpha_crit_rcv_layer1),
                        f"{layer}/alpha_crit_rs1": float(alpha_crit_rs_layer1),
                        f"{layer}/alpha_crit_rvar1": float(alpha_crit_rvar_layer1),
                        f"{layer}/alpha_crit_rsqm1": float(alpha_crit_rsqm_layer1),
                        f"{layer}/alpha_crit_rcv10": float(alpha_crit_rcv_layer10),
                        f"{layer}/alpha_crit_rs10": float(alpha_crit_rs_layer10),
                        f"{layer}/alpha_crit_rvar10": float(alpha_crit_rvar_layer10),
                        f"{layer}/alpha_crit_rsqm10": float(alpha_crit_rsqm_layer10),
                        "reg"                  : reg
                    })
                    diag_log({
                        "task": task, "total_updates": total_updates, "layer": layer,
                        "sharp": norm_lam, "mu": scalars["lam_mean"], "tau": scalars["tau"],
                        "cv": scalars["lam_cv"], "eff_lr": eff_lr, "predict": layer_pred,
                        "alpha_crit_t": float(alpha_crit_t), "sigma2": float(sigma2_l),
                        "g_sq": float(g_layer_sq), "batch_acc": float(acc),
                        "lam_raw": float(lam),
                        "alpha_agg": optimizers.get_alpha_agg(optimizer),
                        "alpha_crit_scv1": float(alpha_crit_scv_layer1),
                        "alpha_crit_ss1": float(alpha_crit_ss_layer1),
                        "alpha_crit_svar1": float(alpha_crit_svar_layer1),
                        "alpha_crit_scv10": float(alpha_crit_scv_layer10),
                        "alpha_crit_ss10": float(alpha_crit_ss_layer10),
                        "alpha_crit_svar10": float(alpha_crit_svar_layer10),
                        "alpha_crit_sqm1": float(alpha_crit_sqm_layer1),
                        "alpha_crit_sqm10": float(alpha_crit_sqm_layer10),
                        "alpha_crit_rcv1": float(alpha_crit_rcv_layer1),
                        "alpha_crit_rs1": float(alpha_crit_rs_layer1),
                        "alpha_crit_rvar1": float(alpha_crit_rvar_layer1),
                        "alpha_crit_rsqm1": float(alpha_crit_rsqm_layer1),
                        "alpha_crit_rcv10": float(alpha_crit_rcv_layer10),
                        "alpha_crit_rs10": float(alpha_crit_rs_layer10),
                        "alpha_crit_rvar10": float(alpha_crit_rvar_layer10),
                        "alpha_crit_rsqm10": float(alpha_crit_rsqm_layer10),
                        "reg": float(reg.detach()) if torch.is_tensor(reg) else reg,
                    })

                    if config.lr_schedule == "pl_lyapunov":
                        # if task <= 2:
                        if config.param == "sqm10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_sqm_layer10, total_updates, total_steps)
                        elif config.param == "t":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_t, total_updates, total_steps)
                        elif config.param == "sqm1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_sqm_layer1, total_updates, total_steps)
                        elif config.param == "scv10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_scv_layer10, total_updates, total_steps)
                        elif config.param == "scv1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_scv_layer1, total_updates, total_steps)
                        elif config.param == "svar10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_svar_layer10, total_updates, total_steps)
                        elif config.param == "svar1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_svar_layer1, total_updates, total_steps)
                        elif config.param == "ss10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_ss_layer10, total_updates, total_steps)
                        elif config.param == "ss1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_ss_layer1, total_updates, total_steps)
                        elif config.param == "rcv1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rcv_layer1, total_updates, total_steps)
                        elif config.param == "rs1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rs_layer1, total_updates, total_steps)
                        elif config.param == "rvar1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rvar_layer1, total_updates, total_steps)
                        elif config.param == "rqm1":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rsqm_layer1, total_updates, total_steps)
                        elif config.param == "rcv10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rcv_layer10, total_updates, total_steps)
                        elif config.param == "rs10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rs_layer10, total_updates, total_steps)
                        elif config.param == "rvar10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rvar_layer10, total_updates, total_steps)
                        elif config.param == "rqm10":
                            lr_star = pl_scheduler.step(layer, eff_lr, alpha_crit_rsqm_layer10, total_updates, total_steps)
                        elif config.param == "s_tau":
                            lr_star = pl_scheduler.step(layer, eff_lr, scalars["tau"], total_updates, total_steps)
                        elif config.param == "r_tau":
                            lr_star = pl_scheduler.step(layer, eff_lr, act_scalars["tau"], total_updates, total_steps)
                        # lr_star = pl_scheduler.step(layer, eff_lr, scalars["tau"], config)

                # accumulate union across layers for this log step
                ly_union_sum += union_pred_step
                eff_acrit_scv1_union_sum += union_eff_gt_acrit_step_scv1
                eff_acrit_ss1_union_sum += union_eff_gt_acrit_step_ss1
                eff_acrit_svar1_union_sum += union_eff_gt_acrit_step_svar1
                eff_acrit_ssqm1_union_sum += union_eff_gt_acrit_step_sqm1
                eff_acrit_scv10_union_sum += union_eff_gt_acrit_step_scv10
                eff_acrit_ss10_union_sum += union_eff_gt_acrit_step_ss10
                eff_acrit_svar10_union_sum += union_eff_gt_acrit_step_svar10
                eff_acrit_ssqm10_union_sum += union_eff_gt_acrit_step_sqm10
                eff_acrit_rcv1_union_sum += union_eff_gt_acrit_step_rcv1
                eff_acrit_rs1_union_sum += union_eff_gt_acrit_step_rs1
                eff_acrit_rvar1_union_sum += union_eff_gt_acrit_step_rvar1
                eff_acrit_rsqm1_union_sum += union_eff_gt_acrit_step_rsqm1
                eff_acrit_rcv10_union_sum += union_eff_gt_acrit_step_rcv10
                eff_acrit_rs10_union_sum += union_eff_gt_acrit_step_rs10
                eff_acrit_rvar10_union_sum += union_eff_gt_acrit_step_rvar10
                eff_acrit_rsqm10_union_sum += union_eff_gt_acrit_step_rsqm10

                if config.track_coherence:
                    tau_dict = {layer: step_stats[layer]["scalars"]["tau"]
                                for layer in layer_map if layer in step_stats}
                    coh = coherence_tracker.update(tau_dict)
                    if coh["corr_matrix"] is not None:
                        wandb.log({
                            "coherence/mean_off_diag": coh["mean_off_diag"],
                            "coherence/max_eigval": coh["max_eigval"],
                            "coherence/min_eigval": coh["min_eigval"],
                            "coherence/ratio": coh["max_eigval"] / max(coh["min_eigval"], 1e-8),
                        }, commit=False)
                        diag_log({
                            "task": task, "total_updates": total_updates, "layer": "__coherence__",
                            "mean_off_diag": coh["mean_off_diag"], "max_eigval": coh["max_eigval"],
                            "min_eigval": coh["min_eigval"],
                            "coherence_ratio": coh["max_eigval"] / max(coh["min_eigval"], 1e-8),
                        })
                        td_mod_sum += float(coh["mean_off_diag"]); td_mod_n += 1
                        td_mod_last = float(coh["mean_off_diag"])

                # estimate_hessian_topk defaults to iters=100 (100 power-
                # iteration steps, each a Hessian-vector product); min_eig
                # here is iters=20. Gate separately from the cheap
                # per-log-interval stuff above; reuse the last computed
                # sharpness/lambda_min in between.
                # One-shot check that the layer loop above no longer clobbers
                # `params` (it used to, reducing the two estimates below to the
                # last layer's tensors). Prints once per run.
                if os.environ.get("SHADOW_CHECK") == "1" and not globals().get("_shadow_done"):
                    print(f"SHADOWCHECK len(params)={len(params)} "
                          f"len(old)={len(old)} n_model_params="
                          f"{len([p for p in model.parameters() if p.requires_grad])} "
                          f"layer_map_sizes={{k: len(v) for k, v in layer_map.items()}}",
                          flush=True)
                    _shadow_done = True

                # Purely observational -- nothing downstream of these feeds
                # the optimizer, so `off` drops them outright.
                if DIAG_ANY and total_updates % _eff_diag_interval == 0:
                    # Residual-criterion stopping instead of a fixed iters=100:
                    # same estimator on the diagnostic and adaptive paths.
                    with prof("est_diag_global"):
                        eigs = optimizers.estimate_hessian_topk(
                            model, loss, params, k=1,
                            iters=config.hessian_max_iters, tol=config.hessian_tol)
                    sharpness = eigs[0]
                    # estimate_hessian_min_eig removed here: 20 HVPs per
                    # diagnostic step producing `lambda_min`, which was never
                    # logged, never plotted, and never used in any computation
                    # (the lambda_min_norm/lambda_state/lambda_log chain is
                    # commented out). Pure dead cost.

                norm_sharpness = optimizers.get_norm_sharpness(optimizer, sharpness, config)
                td_sharp_sum += float(norm_sharpness); td_sharp_n += 1
                td_sharp_last = float(norm_sharpness)
                td_log_steps += 1
                # lambda_min_norm = optimizers.get_norm_sharpness(optimizer, lambda_min, config)
                this_normalized_sharp += norm_sharpness 

                effective_lr = optimizers.compute_effective_lr(
                    optimizer, cfg=config, step=total_updates, sgd_mode="time"
                )
                # Accumulated here, AFTER the assignment above -- accumulating
                # before it would record the previous log step's value.
                _td_eagg = float(effective_lr)
                td_eagg_sum += _td_eagg; td_eagg_n += 1
                td_eagg_min = min(td_eagg_min, _td_eagg)
                td_eagg_max = max(td_eagg_max, _td_eagg)

                sharp_state,  sharp_log  = misc.update_stat(norm_sharpness,  sharp_state,  effective_lr)
                r_sharp_state,  r_sharp_log  = misc.update_stat(sharpness, r_sharp_state,  effective_lr)
                # lambda_state, lambda_log = misc.update_stat(lambda_min_norm, lambda_state, effective_lr)
                ly_snr_sum += sharp_log["collapse_pred"]
                ly_snr_2_sum += sharp_log["collapse_pred2"]
                

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
                loss.backward()
                # Second observational block (SNR tracker, within-batch grad
                # variance, Fisher/alpha_crit series). Same cost lever as the
                # first: it carries another full per-sample backward pass via
                # grad_variance_within_batch. Nothing here feeds the optimizer
                # -- the ly_sched.step branches below are dead (ly_sched is
                # only ever None) -- so `off` drops it wholesale.
                if (DIAG_ANY or DIAG_NEED_SCHED) and total_updates % config.log_interval == 0:
                    params = [p for p in model.parameters() if p.requires_grad]
                    grad_flat = torch.cat([p.grad.view(-1) for p in params])

                    # 1) use the *current* grads to get ||g||^2 BEFORE any SNR math
                    true_grad_norm_sq = float(grad_flat.pow(2).sum().item())

                    # 2) temporal proxy SNR (unchanged)
                    eta_eff = effective_lr
                    batch_B = inputs.size(0)
                    T_t, sigma2_hat = snr_tracker.update(grad_flat.detach(), eta_eff, int(batch_B))
                    if T_t is not None:
                        snr_T_series.append((total_updates, float(T_t)))

                    # 3) within-minibatch σ² (Mark’s definition)
                    sigma2_hat_mb = optimizers.grad_variance_within_batch(model, criterion_nored, inputs, labels)
                    k_rs_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharpness
                    k_ss_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*norm_sharpness
                    k_rs_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharpness
                    k_ss_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*norm_sharpness
                    s_scv_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["lam_cv"]
                    s_svar_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["lam_var"]
                    s_rcv_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["lam_cv"]
                    s_rvar_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["lam_var"] # decent contender w/ a smaller coeff
                    s_scv_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharp_log["lam_cv"]
                    s_svar_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*sharp_log["lam_var"]
                    s_rcv_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["lam_cv"]
                    s_rvar_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["lam_var"]
                    s_ssqm_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*sharp_log["sq_mean"]
                    s_rsqm_sigma2_hat_mb10 = sigma2_hat_mb + 20.0 * true_grad_norm_sq*r_sharp_log["sq_mean"]
                    s_ssqm_sigma2_hat_mb1 = sigma2_hat_mb + 75.0 * true_grad_norm_sq*sharp_log["sq_mean"]
                    s_rsqm_sigma2_hat_mb1 = sigma2_hat_mb + 10.0 * true_grad_norm_sq*r_sharp_log["sq_mean"]

                    r = (sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq + 1e-12
                    T_t_mb = eta_eff * (sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                    # S_t_mb = eta_eff * (s_sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                    # K_t_mb = eta_eff * (k_sigma2_hat_mb / max(1, batch_B)) / true_grad_norm_sq
                    K_rs_t_mb1 = eta_eff * (k_rs_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    K_ss_t_mb1 = eta_eff * (k_ss_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    K_rs_t_mb10 = eta_eff * (k_rs_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    K_ss_t_mb10 = eta_eff * (k_ss_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq

                    S_scv_t_mb10 = eta_eff * (s_scv_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    S_svar_t_mb10 = eta_eff * (s_svar_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    S_rcv_t_mb10 = eta_eff * (s_rcv_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    S_rvar_t_mb10 = eta_eff * (s_rvar_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq

                    S_scv_t_mb1 = eta_eff * (s_scv_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    S_svar_t_mb1 = eta_eff * (s_svar_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    S_rcv_t_mb1 = eta_eff * (s_rcv_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    S_rvar_t_mb1 = eta_eff * (s_rvar_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq

                    S_ssqm_t_mb1 = eta_eff * (s_ssqm_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    S_rsqm_t_mb1 = eta_eff * (s_rsqm_sigma2_hat_mb1 / max(1, batch_B)) / true_grad_norm_sq
                    S_ssqm_t_mb10 = eta_eff * (s_ssqm_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    S_rsqm_t_mb10 = eta_eff * (s_rsqm_sigma2_hat_mb10 / max(1, batch_B)) / true_grad_norm_sq
                    # --- SNR-based progress prediction (no scheduling) -----------------------
                    # pred, pred_real, meanT, thresh, conf = snr_predictor.update(T_t_mb, T_t)
                    # K_pred, K_pred_real, K_meanT, K_thresh, K_conf = snr_predictor.update(K_t_mb, T_t)
                    # S_pred, S_pred_real, S_meanT, S_thresh, S_conf = snr_predictor.update(S_t_mb, T_t)
                    pred, pred_real, meanT, thresh, conf = snr_predictor.update(T_t_mb, T_t)
                    K_rs_pred1, K_rs_pred_real1, K_rs_meanT1, K_rs_thresh1, K_rs_conf1 = snr_predictor.update(K_rs_t_mb1, T_t)
                    K_ss_pred1, K_ss_pred_real1, K_ss_meanT1, K_ss_thresh1, K_ss_conf1 = snr_predictor.update(K_ss_t_mb1, T_t)
                    K_rs_pred10, K_rs_pred_real10, K_rs_meanT10, K_rs_thresh10, K_rs_conf10 = snr_predictor.update(K_rs_t_mb10, T_t)
                    K_ss_pred10, K_ss_pred_real10, K_ss_meanT10, K_ss_thresh10, K_ss_conf10 = snr_predictor.update(K_ss_t_mb10, T_t)
                    S_scv_pred10, S_scv_pred_real10, S_scv_meanT10, S_scv_thresh10, S_scv_conf10 = snr_predictor.update(S_scv_t_mb10, T_t)
                    S_svar_pred10, S_svar_pred_real10, S_svar_meanT10, S_svar_thresh10, S_svar_conf10 = snr_predictor.update(S_svar_t_mb10, T_t)
                    S_rcv_pred10, S_rcv_pred_real10, S_rcv_meanT10, S_rcv_thresh10, S_rcv_conf10 = snr_predictor.update(S_rcv_t_mb10, T_t)
                    S_rvar_pred10, S_rvar_pred_real10, S_rvar_meanT10, S_rvar_thresh10, S_rvar_conf10 = snr_predictor.update(S_rvar_t_mb10, T_t)
                    S_scv_pred1, S_scv_pred_real1, S_scv_meanT1, S_scv_thresh1, S_scv_conf1 = snr_predictor.update(S_scv_t_mb1, T_t)
                    S_svar_pred1, S_svar_pred_real1, S_svar_meanT1, S_svar_thresh1, S_svar_conf1 = snr_predictor.update(S_svar_t_mb1, T_t)
                    S_rcv_pred1, S_rcv_pred_real1, S_rcv_meanT1, S_rcv_thresh1, S_rcv_conf1 = snr_predictor.update(S_rcv_t_mb1, T_t)
                    S_rvar_pred1, S_rvar_pred_real1, S_rvar_meanT1, S_rvar_thresh1, S_rvar_conf1 = snr_predictor.update(S_rvar_t_mb1, T_t)
                    S_ssqm_pred1, S_ssqm_pred_real1, S_ssqm_meanT1, S_ssqm_thresh1, S_ssqm_conf1 = snr_predictor.update(S_ssqm_t_mb1, T_t)
                    S_rsqm_pred1, S_rsqm_pred_real1, S_rsqm_meanT1, S_rsqm_thresh1, S_rsqm_conf1 = snr_predictor.update(S_rsqm_t_mb1, T_t)
                    S_ssqm_pred10, S_ssqm_pred_real10, S_ssqm_meanT10, S_ssqm_thresh10, S_ssqm_conf10 = snr_predictor.update(S_ssqm_t_mb10, T_t)
                    S_rsqm_pred10, S_rsqm_pred_real10, S_rsqm_meanT10, S_rsqm_thresh10, S_rsqm_conf10 = snr_predictor.update(S_rsqm_t_mb10, T_t)

                    # Critical effective LR from Mark's 2nd trade-off
                    # alpha_crit_t = (batch_B * true_grad_norm_sq) / max(sigma2_hat_mb, 1e-12)
                    # alpha_crit_k = (batch_B * true_grad_norm_sq) / max(k_sigma2_hat_mb, 1e-12)
                    # alpha_crit_s = (batch_B * true_grad_norm_sq) / max(s_sigma2_hat_mb, 1e-12)
                    alpha_crit_t = (batch_B * true_grad_norm_sq) / max(sigma2_hat_mb, 1e-12)
                    alpha_crit_k_rs1 = (batch_B * true_grad_norm_sq) / max(k_rs_sigma2_hat_mb1, 1e-12)
                    alpha_crit_k_ss1 = (batch_B * true_grad_norm_sq) / max(k_ss_sigma2_hat_mb1, 1e-12)
                    alpha_crit_k_rs10 = (batch_B * true_grad_norm_sq) / max(k_rs_sigma2_hat_mb10, 1e-12)
                    alpha_crit_k_ss10 = (batch_B * true_grad_norm_sq) / max(k_ss_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_scv10 = (batch_B * true_grad_norm_sq) / max(s_scv_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_svar10 = (batch_B * true_grad_norm_sq) / max(s_svar_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_rcv10 = (batch_B * true_grad_norm_sq) / max(s_rcv_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_rvar10 = (batch_B * true_grad_norm_sq) / max(s_rvar_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_scv1 = (batch_B * true_grad_norm_sq) / max(s_scv_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_svar1 = (batch_B * true_grad_norm_sq) / max(s_svar_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_rcv1 = (batch_B * true_grad_norm_sq) / max(s_rcv_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_rvar1 = (batch_B * true_grad_norm_sq) / max(s_rvar_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_sqm1 = (batch_B * true_grad_norm_sq) / max(s_ssqm_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_rsqm1 = (batch_B * true_grad_norm_sq) / max(s_rsqm_sigma2_hat_mb1, 1e-12)
                    alpha_crit_s_sqm10 = (batch_B * true_grad_norm_sq) / max(s_ssqm_sigma2_hat_mb10, 1e-12)
                    alpha_crit_s_rsqm10 = (batch_B * true_grad_norm_sq) / max(s_rsqm_sigma2_hat_mb10, 1e-12)
                    alphas = {
                        # "effective_lr": effective_lr,
                        "alpha_crit_t": alpha_crit_t,
                        "alpha_crit_k_rs1": alpha_crit_k_rs1,
                        "alpha_crit_k_ss1": alpha_crit_k_ss1,
                        "alpha_crit_k_rs10": alpha_crit_k_rs10,
                        "alpha_crit_k_ss10": alpha_crit_k_ss10,
                        "alpha_crit_s_scv10": alpha_crit_s_scv10,
                        "alpha_crit_s_svar10": alpha_crit_s_svar10,
                        "alpha_crit_s_rcv10": alpha_crit_s_rcv10,
                        "alpha_crit_s_rvar10": alpha_crit_s_rvar10,
                        "alpha_crit_s_scv1": alpha_crit_s_scv1,
                        "alpha_crit_s_svar1": alpha_crit_s_svar1,
                        "alpha_crit_s_rcv1": alpha_crit_s_rcv1,
                        "alpha_crit_s_rvar1": alpha_crit_s_rvar1,
                        "alpha_crit_s_sqm1": alpha_crit_s_sqm1,
                        "alpha_crit_s_rsqm1": alpha_crit_s_rsqm1,
                        "alpha_crit_s_sqm10": alpha_crit_s_sqm10,
                        "alpha_crit_s_rsqm10": alpha_crit_s_rsqm10,
                    }

                    # Prediction: 1 if we are above the critical LR
                    # snr_sum += pred_real 
                    # k_snr_sum += K_pred_real
                    # s_snr_sum += S_pred_real
                    snr_sum += pred_real
                    k_rs_sum1 += K_rs_pred_real1
                    k_ss_sum1 += max(K_ss_pred_real1, union_eff_gt_acrit_step_ss1)
                    k_rs_sum10 += K_rs_pred_real10
                    k_ss_sum10 += max(K_ss_pred_real10, union_eff_gt_acrit_step_ss10)
                    s_scv_sum10 += max(S_scv_pred_real10, union_eff_gt_acrit_step_scv10)
                    s_svar_sum10 += max(S_svar_pred_real10, union_eff_gt_acrit_step_svar10)
                    s_rcv_sum10 += S_rcv_pred_real10
                    s_rvar_sum10 += S_rvar_pred_real10
                    s_scv_sum1 += max(S_scv_pred_real1, union_eff_gt_acrit_step_scv1)
                    s_svar_sum1 += max(S_svar_pred_real1,union_eff_gt_acrit_step_svar1)
                    s_rcv_sum1 += S_rcv_pred_real1
                    s_rvar_sum1 += S_rvar_pred_real1
                    s_sqm_sum1 += max(S_ssqm_pred_real1, union_eff_gt_acrit_step_sqm1)
                    s_rsqm_sum1 += S_rsqm_pred_real1
                    s_sqm_sum10 += max(S_ssqm_pred_real10, union_eff_gt_acrit_step_sqm10)
                    s_rsqm_sum10 += S_rsqm_pred_real10

                    n_crit = min(0.8/r, 2/(norm_sharpness * (1 + r)))

                    mark = {
                        # "snr_T":            float(T_t) if T_t is not None else float("nan"),
                        # "snr_sigma2_hat":   float(sigma2_hat) if sigma2_hat is not None else float("nan"),
                        "mb_sigma2_hat":    float(sigma2_hat_mb),
                        "mb_snr_T":         float(T_t_mb),
                        # "S_t_mb":           float(S_t_mb),
                        "sharpness":        sharpness,
                        # "rho":              effective_lr * sharpness / 2,
                        # "rho_norm":      effective_lr * norm_sharpness / 2,
                        "n_crit_nois":  0.8/r,
                        "n_crit_curv":  2/(norm_sharpness * (1 + r)),
                        "n_crit":       min(0.8/r, 2/(norm_sharpness * (1 + r))),
                        "lbo":          2/(sharpness+1e-12),
                        "lbo_norm":     2/(norm_sharpness+1e-12),
                        # "K_t_mb":           float(K_t_mb),
                        **alphas,  # critical LRs
                        # "alpha_crit_t":     float(alpha_crit_t),
                        # "alpha_crit_k":     float(alpha_crit_k),
                        # "alpha_crit_s":     float(alpha_crit_s),

                        # "snr_pred":         int(pred) if pred is not None else -1,
                        # "snr_pred_real":    int(pred_real) if pred_real is not None else -1,
                        # "snr_pred_conf":    float(conf) if conf is not None else float("nan"),
                        # "snr_T_mean":       float(meanT) if meanT is not None else float("nan"),
                        # "snr_T_thresh":     float(thresh) if thresh is not None else float("nan"),
                        # "snr_K_real":       int(K_pred_real) if K_pred_real is not None else -1,
                        # "snr_S_real":       int(S_pred_real) if S_pred_real is not None else -1,
                    }
                    # ------------------------------------------------------------------------
                    if ly_sched is not None:
                        if config.param == "sqm10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_sqm10, total_updates, total_steps)
                        elif config.param == "t":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_t, total_updates, total_steps)
                        elif config.param == "sqm1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_sqm1, total_updates, total_steps)
                        elif config.param == "svar10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_svar10, total_updates, total_steps)
                        elif config.param == "svar1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_svar1, total_updates, total_steps)
                        elif config.param == "ss10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_ss10, total_updates, total_steps)
                        elif config.param == "ss1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_ss1, total_updates, total_steps)
                        elif config.param == "scv10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_scv10, total_updates, total_steps)
                        elif config.param == "scv1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_scv1, total_updates, total_steps)
                        elif config.param == "rqm10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rsqm10, total_updates, total_steps)
                        elif config.param == "rqm1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rsqm1, total_updates, total_steps)
                        elif config.param == "rcv10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rcv10, total_updates, total_steps)
                        elif config.param == "rvar10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rvar10, total_updates, total_steps)
                        elif config.param == "rcv1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rcv1, total_updates, total_steps)
                        elif config.param == "rvar1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_s_rvar1, total_updates, total_steps)
                        elif config.param == "rs10":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_rs10, total_updates, total_steps)
                        elif config.param == "rs1":
                            lr_star, _ = ly_sched.step(effective_lr, alpha_crit_k_rs1, total_updates, total_steps)
                        elif config.param == "s_tau":
                            lr_star, _ = ly_sched.step(effective_lr, sharp_log["tau"], total_updates, total_steps)
                        elif config.param == "r_tau":
                            lr_star, _ = ly_sched.step(effective_lr, r_sharp_log["tau"], total_updates, total_steps)
                        log_extra  = {"ly_lr_star": lr_star}
                    else:
                        log_extra  = {}
                optimizer.step()

                if config.use_cbp:
                    post_l1 = cbp_post_activation(activations["l1"], config)
                    post_l2 = cbp_post_activation(activations["l2"], config)
                    cbp_tracker.step("fc1", post_l1)
                    cbp_tracker.step("fc2", post_l2)
                    if total_updates % config.log_interval == 0:
                        wandb.log({
                            "cbp/fc1_replaced_this_step": cbp_tracker.last_replaced_count["fc1"],
                            "cbp/fc2_replaced_this_step": cbp_tracker.last_replaced_count["fc2"],
                            "cbp/fc1_replaced_total": cbp_tracker.total_replaced_count["fc1"],
                            "cbp/fc2_replaced_total": cbp_tracker.total_replaced_count["fc2"],
                        }, commit=False)

                # betas scheduling -- OFF unless --beta_schedule is passed.
                #
                # This block used to fire on every Adam run while the
                # --beta_schedule flag that appears to control it was never
                # read anywhere in the codebase. Because `epoch` resets at
                # each task, it drove a per-task beta2 sawtooth (0.999 ->
                # ~0.876 within a task, then snapping back). beta2 sets the
                # exp_avg_sq accumulation rate, and get_norm_sharpness
                # normalizes by exactly that -- so the sawtooth propagated
                # into normalized sharpness, compute_effective_lr, and every
                # alpha_crit_*, i.e. straight into the quantities the
                # plasticity claims rest on. Now opt-in, so the schedule
                # stays available for deliberate study but is not a silent
                # confound on every run.
                if config.optimizer == "adam" and getattr(config, "beta_schedule", False):
                    # Apply to every layer group, not just param_groups[0].
                    # Only updating group 0 previously meant one layer (fc1)
                    # got a per-task beta1/beta2 ramp that reset every task,
                    # while every other layer group kept the original static
                    # betas -- a purely artifactual per-layer discrepancy
                    # that fed straight into compute_effective_lr (which
                    # reads param_groups[0] only) and per_layer_effective_lr
                    # (which reads each group's own betas), and from there
                    # into alpha_crit and the lr-schedule's own step inputs.
                    new_betas = optimizers.get_betas(config, epoch)
                    for group in optimizer.param_groups:
                        group['betas'] = new_betas
                # lr schedule step
                if config.lr_schedule != "constant" and scheduler is not None:
                    scheduler.step()    
                # if ly_sched is not None:
                #     lr_star, _ = ly_sched.step(effective_lr, lambda_log["tau"])
                #     log_extra  = {"ly_lr_star": lr_star}
                # else:
                #     log_extra  = {}

            delta = torch.cat(
                [(p.data - o).view(-1).abs() for p, o in zip(params, old)]
            )
            update_norm = delta.mean().item()
            sum_up += update_norm
            total_updates += 1
            
            # Third observational block: the big per-step run.log dict. It
            # reads names (sharp_log, true_grad_norm_sq, log_extra, mark) that
            # only exist once the two blocks above have run, so it has to be
            # gated on the same condition rather than left to NameError under
            # --diagnostics off.
            if (DIAG_ANY or DIAG_NEED_SCHED) and total_updates % config.log_interval == 0:
                # use_vals = { f"use_{name}": optimizers.compute_use_for_activation(h) for name, h in activations.items() }
                # avg_use_val = sum(use_vals.values()) / len(use_vals)
                with prof("max_singular_logblock"):
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
                    "true_grad_norm_sq": true_grad_norm_sq,
                    "effective_lr": effective_lr,
                    # **init_sigma_min,
                    "max_singular": gg,
                    **sharp_log,                                   # sharpness stats
                    # **{f"lam_{k}": v for k, v in lambda_log.items()},
                    **log_extra,               # only when ly_sched is active
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
    steps = (config.epochs * len(loader) / config.log_interval)
    run.log(
        {
            "J": J,
            "param_norm": pn,
            "avg_norm_sharp": this_normalized_sharp / (config.epochs * len(loader)),
            "average_update_norm": aun,
            "effective_rank": effective_rank,
            "task_acc": this_task_acc / (config.epochs * len(loader)),
            "snr_pct": 1 - (snr_sum / steps),
            "k_rs_pct1": 1 - (k_rs_sum1 / steps),
            "k_ss_pct1": 1 - (k_ss_sum1 / steps),
            "k_rs_pct10": 1 - (k_rs_sum10 / steps),
            "k_ss_pct10": 1 - (k_ss_sum10 / steps),
            "s_scv_pct10": 1 - (s_scv_sum10 / steps),
            "s_svar_pct10": 1 - (s_svar_sum10 / steps),
            "s_rcv_pct10": 1 - (s_rcv_sum10 / steps),
            "s_rvar_pct10": 1 - (s_rvar_sum10 / steps),
            "s_scv_pct1": 1 - (s_scv_sum1 / steps),
            "s_svar_pct1": 1 - (s_svar_sum1 / steps),
            "s_rcv_pct1": 1 - (s_rcv_sum1 / steps),
            "s_rvar_pct1": 1 - (s_rvar_sum1 / steps),
            "s_sqm_pct1": 1 - (s_sqm_sum1 / steps),
            "s_rsqm_pct1": 1 - (s_rsqm_sum1 / steps),
            "s_sqm_pct10": 1 - (s_sqm_sum10 / steps),
            "s_rsqm_pct10": 1 - (s_rsqm_sum10 / steps),
            # "k_snr_pct": 1 - (k_snr_sum / steps),
            # "s_snr_pct": 1 - (s_snr_sum / steps),
            "ly_snr_pct": 1 - (ly_snr_sum / steps),
            "ly_snr_pct2": 1 - (ly_snr_2_sum / steps),
            "ly_snr_pct_union": 1 - (ly_union_sum / steps),
            "eff_acrit_scv1_pct_union": 1 - (eff_acrit_scv1_union_sum / steps),
            "eff_acrit_ss1_pct_union": 1 - (eff_acrit_ss1_union_sum / steps),
            "eff_acrit_svar1_pct_union": 1 - (eff_acrit_svar1_union_sum / steps),
            "eff_acrit_scv10_pct_union": 1 - (eff_acrit_scv10_union_sum / steps),
            "eff_acrit_ss10_pct_union": 1 - (eff_acrit_ss10_union_sum / steps),
            "eff_acrit_svar10_pct_union": 1 - (eff_acrit_svar10_union_sum / steps),
            "eff_acrit_ssqm1_pct_union": 1 - (eff_acrit_ssqm1_union_sum / steps),
            "eff_acrit_ssqm10_pct_union": 1 - (eff_acrit_ssqm10_union_sum / steps),
            "eff_acrit_rcv1_pct_union": 1 - (eff_acrit_rcv1_union_sum / steps),
            "eff_acrit_rs1_pct_union": 1 - (eff_acrit_rs1_union_sum / steps),
            "eff_acrit_rvar1_pct_union": 1 - (eff_acrit_rvar1_union_sum / steps),
            "eff_acrit_rsqm1_pct_union": 1 - (eff_acrit_rsqm1_union_sum / steps),
            "eff_acrit_rcv10_pct_union": 1 - (eff_acrit_rcv10_union_sum / steps),
            "eff_acrit_rs10_pct_union": 1 - (eff_acrit_rs10_union_sum / steps),
            "eff_acrit_rvar10_pct_union": 1 - (eff_acrit_rvar10_union_sum / steps),
            "eff_acrit_rsqm10_pct_union": 1 - (eff_acrit_rsqm10_union_sum / steps),
        }
    )
    # Freeze the fixed tau_ref at the end of task 0 from the calibration
    # window (last 20% of task 0), unless an explicit value was given.
    if task == 0 and getattr(config, "tau_ref_mode", "median") == "fixed":
        _explicit = getattr(config, "tau_ref_fixed", None)
        for _l, _vals in tau_ref_calib.items():
            tau_ref_frozen[_l] = (_explicit if _explicit is not None
                                  else (sum(_vals) / len(_vals) if _vals else 1e-3))
        print(f"[tau_ref] frozen: {tau_ref_frozen}", flush=True)

    task_acc_history.append(this_task_acc / (config.epochs * len(loader)))

    # ---- per-task diagnostic snapshot ----
    # Long-format detail: one row per (task, layer). The layer-collapsed
    # version of the same numbers goes onto the per-run results_csv row as
    # ;-joined trajectories, so the summary CSV alone is enough to plot a
    # metric against task index without opening this file.
    _td_task_acc = task_acc_history[-1]
    _td_sharp_mean = (td_sharp_sum / td_sharp_n) if td_sharp_n else float("nan")
    _td_mod_mean = (td_mod_sum / td_mod_n) if td_mod_n else float("nan")
    _td_layers = sorted(td_tau_last.keys())
    _td_sigma2_final = {l: float(layer_sigma2_mb.get(l, float("nan"))) for l in _td_layers}

    for _l in _td_layers:
        taskdiag_log({
            "task": task, "layer": _l, "git_sha": GIT_SHA, "seed": config.seed,
            "model": config.model, "dataset": config.dataset, "reg": config.reg,
            "adaptive_reg": bool(getattr(config, "adaptive_reg", False)),
            "adaptive_type": getattr(config, "adaptive_type", ""),
            "adaptive_scale": getattr(config, "adaptive_scale", ""),
            "reg_sensitivity": getattr(config, "reg_sensitivity", ""),
            "lr": config.lr, "lr_schedule": getattr(config, "lr_schedule", ""),
            "sched_param": getattr(config, "param", ""),
            "task_acc": _td_task_acc, "n_log_steps": td_log_steps,
            "tau_mean": td_tau_sum[_l] / max(td_tau_n[_l], 1),
            "tau_final": td_tau_last[_l],
            "adaptive_factor_mean": (td_af_sum[_l] / td_af_n[_l]) if td_af_n.get(_l) else "",
            "adaptive_factor_final": td_af_last.get(_l, ""),
            "adaptive_factor_std": (
                max(td_af_sq[_l] / td_af_n[_l] - (td_af_sum[_l] / td_af_n[_l]) ** 2, 0.0) ** 0.5
                if td_af_n.get(_l) else ""),
            "sigma2_final": _td_sigma2_final[_l],
            "eff_lr_layer_mean": (td_elr_sum[_l] / td_elr_n[_l]) if td_elr_n.get(_l) else "",
            "eff_lr_layer_min": td_elr_min.get(_l, ""),
            "eff_lr_layer_max": td_elr_max.get(_l, ""),
            "eff_lr_agg_mean": (td_eagg_sum / td_eagg_n) if td_eagg_n else "",
            "eff_lr_agg_min": td_eagg_min if td_eagg_n else "",
            "eff_lr_agg_max": td_eagg_max if td_eagg_n else "",
            "sharp_norm_mean": _td_sharp_mean, "sharp_norm_final": td_sharp_last,
            "mean_off_diag_mean": _td_mod_mean, "mean_off_diag_final": td_mod_last,
        })

    # Layer-collapsed summaries -> one value per metric per task.
    taskdiag_history["tau_mean"].append(
        _mean([td_tau_sum[l] / max(td_tau_n[l], 1) for l in _td_layers]))
    taskdiag_history["tau_final"].append(_mean([td_tau_last[l] for l in _td_layers]))
    taskdiag_history["adaptive_factor_mean"].append(
        _mean([td_af_sum[l] / td_af_n[l] for l in _td_layers if td_af_n.get(l)]))
    taskdiag_history["adaptive_factor_final"].append(
        _mean([td_af_last[l] for l in _td_layers if l in td_af_last]))
    taskdiag_history["sigma2_final"].append(_mean([_td_sigma2_final[l] for l in _td_layers]))
    taskdiag_history["sharp_norm_mean"].append(_td_sharp_mean)
    taskdiag_history["sharp_norm_final"].append(td_sharp_last)
    taskdiag_history["mean_off_diag_mean"].append(_td_mod_mean)
    taskdiag_history["mean_off_diag_final"].append(td_mod_last)

    res = results[config.activation]
    res["batch_error"].append(J)
    res["param_norm"].append(pn)
    res["update_norm"].append(aun)

    if config.track_coherence and coh is not None and coh["corr_matrix"] is not None:
        fig, ax = plt.subplots()
        im = ax.imshow(coh["corr_matrix"], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(coh["layer_names"])))
        ax.set_yticks(range(len(coh["layer_names"])))
        ax.set_xticklabels(coh["layer_names"], rotation=45)
        ax.set_yticklabels(coh["layer_names"])
        plt.colorbar(im)
        wandb.log({f"coherence/matrix_task_{task}": wandb.Image(fig)})
        plt.close(fig)

if _PROF:
    _tot = _time.time() - _PROF_START
    _acc = sum(_prof_t.values())
    print("=" * 78, flush=True)
    print(f"COST PROFILE  total wall {_tot:.1f} s", flush=True)
    print(f"{'section':<28}{'seconds':>12}{'pct':>9}{'calls':>10}{'ms/call':>11}", flush=True)
    for _k in sorted(_prof_t, key=lambda x: -_prof_t[x]):
        _v = _prof_t[_k]; _c = _prof_n[_k]
        print(f"{_k:<28}{_v:>12.1f}{100.0*_v/_tot:>8.1f}%{_c:>10}{1000.0*_v/max(_c,1):>11.2f}",
              flush=True)
    print(f"{'everything else (residual)':<28}{_tot-_acc:>12.1f}{100.0*(_tot-_acc)/_tot:>8.1f}%",
          flush=True)
    print("=" * 78, flush=True)

wandb.finish()
if _diag_csv_fh is not None:
    _diag_csv_fh.close()
if _taskdiag_csv_fh is not None:
    _taskdiag_csv_fh.close()

tasks = np.arange(1, len(results[config.activation]["batch_error"]) + 1)
for m in ["batch_error", "param_norm", "update_norm"]:
    plt.figure()
    plt.plot(tasks, results[config.activation][m], label=config.activation)
    plt.xlabel("Task")
    plt.ylabel(m)
    plt.legend()
    plt.show()


# ---- per-run summary CSV ----
import csv as _csv, os as _os
if getattr(config, "results_csv", None):
    _traj = task_acc_history
    _auc = sum(_traj)/len(_traj) if _traj else 0.0
    _n = len(_traj)
    if _n >= 2:
        _x = list(range(_n)); _xm=sum(_x)/_n; _ym=sum(_traj)/_n
        _num=sum((a-_xm)*(b-_ym) for a,b in zip(_x,_traj)); _den=sum((a-_xm)**2 for a in _x) or 1e-12
        _slope=_num/_den
    else:
        _slope=0.0
    _row={"adaptive_type":getattr(config,"adaptive_type",""),
          "reg_sensitivity":getattr(config,"reg_sensitivity",""),
          "adaptive_reg":bool(getattr(config,"adaptive_reg",False)),
          "adaptive_scale":getattr(config,"adaptive_scale","inv"),
          "reg":config.reg,"model":config.model,"dataset":config.dataset,
          "lr":config.lr,"ns":config.ns,"hidden":getattr(config,"hidden",256),
          "lr_schedule":getattr(config,"lr_schedule","constant"),
          "sched_param":getattr(config,"param",""),
          "seed":config.seed,"runs":config.runs,
          "auc":_auc,"slope":_slope,
          "final_acc":_traj[-1] if _traj else 0.0,
          "task_acc_traj":";".join(f"{a:.5f}" for a in _traj)}
    # One summary statistic per metric per task, same ;-joined shape as
    # task_acc_traj. Layer-collapsed (mean over layers); per-layer detail
    # lives in --taskdiag_csv.
    for _k, _v in taskdiag_history.items():
        _row[_k + "_traj"] = _fmt_traj(_v)
    _row["centered_clamp_hits"] = centered_clamp_hits
    _row["centered_factor_evals"] = centered_factor_evals
    _row["centered_clamp_frac"] = (centered_clamp_hits / centered_factor_evals
                                   if centered_factor_evals else 0.0)
    _row["git_sha"] = GIT_SHA
    _row["full_config"] = FULL_CONFIG_JSON
    _exists=_os.path.exists(config.results_csv)
    with open(config.results_csv,"a",newline="") as _fh:
        _w=_csv.DictWriter(_fh,fieldnames=list(_row.keys()))
        if not _exists or _fh.tell()==0: _w.writeheader()
        _w.writerow(_row)
