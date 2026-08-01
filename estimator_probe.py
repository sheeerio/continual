# TASKS 1-3: warm-start validation, adequately-sampled step-to-step CV, and
# the historical iters=1 bias factor. One run on the CALIBRATED config
# (50 epochs/task, ~2100 steps/task) so the sampling windows fit inside a task.
#
# Arm is static spectral c=1e-3 throughout, which is what Task 3 specifies and
# lets one trajectory serve all three measurements.
#
# Warm-start variance protocol: warm-start is deterministic once seeded, so
# "std over 30 repeats" means 30 independent CHAINS, each seeded with its own
# random vector and advanced along the SAME weight trajectory. Spread across
# chains at a checkpoint is the residual init-dependence; distance from cold
# iters=100 is the bias.
import os, sys, time, json
import statistics as st
import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_parser
from utils import misc, optimizers
from models import mlp
from datasets import data_loader

parser = get_parser()
config = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np, random
torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

train_dataset, test_dataset, in_ch, input_size, MEAN, STD = data_loader.get_dataset(config)
model = mlp.MLP(input_size, config.hidden, 10).to(device)
criterion = nn.CrossEntropyLoss()

layer_map = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        layer_map.setdefault(name.split('.')[0], []).append(p)
layer_groups = [{'params': ps, 'lr': config.lr, 'layer': l} for l, ps in layer_map.items()]
optimizer = torch.optim.Adam(layer_groups, lr=config.lr, betas=(config.beta1, config.beta2))
LAYERS = list(layer_map.keys())

N_CHAINS = 30
WARM_ITERS = [1, 2, 5]
WARM_WINDOW = 60          # steps of chain advancement before each checkpoint
STEP_WINDOW = 50          # >= 50 consecutive LOGGED steps for Task 2
SPECTRAL_LAMBDA = config.spectral_lambda

R = {"warm": [], "fixed": [], "batch": [], "steps": [], "bias": [], "timing": {}}


def cv(xs):
    if len(xs) < 2:
        return float("nan")
    m = st.mean(xs)
    return st.stdev(xs) / abs(m) if m else float("nan")


def spectral_reg():
    r = torch.tensor(0.0, device=device)
    for n_, p in model.named_parameters():
        if p.requires_grad and p.ndim >= 2:
            r = r + (optimizers.power_iteration(p, 1).pow(config.spectral_k) - 1.0).pow(2)
    return r * SPECTRAL_LAMBDA


def cold(inputs, labels, layer, iters):
    base = criterion(model(inputs), labels)
    return optimizers.estimate_hessian_topk(model, base, layer_map[layer], k=1, iters=iters)[0]


def fixed_point(tag, inputs, labels):
    """Cold estimator variance at this checkpoint (reference for Task 2)."""
    out = []
    for iters in (1, 100):
        for layer in LAYERS:
            vals = [cold(inputs, labels, layer, iters) for _ in range(N_CHAINS)]
            out.append({"checkpoint": tag, "iters": iters, "layer": layer,
                        "mean": st.mean(vals), "std": st.stdev(vals), "cv": cv(vals)})
    return out


def batch_spread(tag, loader):
    """Real batch-to-batch curvature variation, iters=100."""
    out = []
    per = {l: [] for l in LAYERS}
    it = iter(loader)
    for _ in range(N_CHAINS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x = x.view(x.size(0), -1).to(device); y = y.to(device)
        base = criterion(model(x), y)
        for layer in LAYERS:
            per[layer].append(optimizers.estimate_hessian_topk(
                model, base, layer_map[layer], k=1, iters=100)[0])
    for layer, v in per.items():
        out.append({"checkpoint": tag, "layer": layer, "mean": st.mean(v),
                    "std": st.stdev(v), "cv": cv(v)})
    return out


def time_setting(inputs, labels, iters, warm, cache):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time(); n = 0
    for _ in range(10):
        for layer in LAYERS:
            base = criterion(model(inputs), labels)
            if warm:
                _e, _v = optimizers.estimate_hessian_topk(
                    model, base, layer_map[layer], k=1, iters=iters,
                    v_init=cache.get(layer), return_v=True)
                cache[layer] = _v
            else:
                optimizers.estimate_hessian_topk(model, base, layer_map[layer], k=1, iters=iters)
            n += 1
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) / n


# chains[iters][layer] -> list of N_CHAINS vectors
chains = {it: {l: [None] * N_CHAINS for l in LAYERS} for it in WARM_ITERS}
# Task 1c: one warm chain vs cold-100, tracked over a stretch of steps
wc_series = {l: [] for l in LAYERS}
cold_series = {l: [] for l in LAYERS}
wc_cache = {}

CHECK = {}   # tag -> (task, step)
for task in range(3):
    if task > 0:
        train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
    loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    n_steps = config.epochs * len(loader)
    mid = n_steps // 2
    if task == 0:
        cp_step = min(200, n_steps // 4); tag_here = "early_task0"
    elif task == 1:
        cp_step = mid; tag_here = "mid_task1"
    else:
        cp_step = mid; tag_here = "mid_task2"
    warm_begin = cp_step - WARM_WINDOW
    step_begin = cp_step
    step_end = cp_step + STEP_WINDOW * config.log_interval

    step = 0
    for epoch in range(config.epochs):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device); y = y.to(device)

            # ---- advance warm chains in the window before the checkpoint ----
            if warm_begin <= step < cp_step:
                base = criterion(model(x), y)
                for iters in WARM_ITERS:
                    for layer in LAYERS:
                        for ci in range(N_CHAINS):
                            v0 = chains[iters][layer][ci]
                            _e, _v = optimizers.estimate_hessian_topk(
                                model, base, layer_map[layer], k=1, iters=iters,
                                v_init=[v0] if v0 is not None else None, return_v=True)
                            chains[iters][layer][ci] = _v[0]

            # ---- checkpoint measurements ----
            if step == cp_step:
                print(f"[probe] {tag_here} task={task} step={step}", flush=True)
                base = criterion(model(x), y)
                truth = {l: optimizers.estimate_hessian_topk(
                    model, base, layer_map[l], k=1, iters=100)[0] for l in LAYERS}
                for iters in WARM_ITERS:
                    for layer in LAYERS:
                        vals = []
                        for ci in range(N_CHAINS):
                            v0 = chains[iters][layer][ci]
                            e, v = optimizers.estimate_hessian_topk(
                                model, base, layer_map[layer], k=1, iters=iters,
                                v_init=[v0] if v0 is not None else None, return_v=True)
                            chains[iters][layer][ci] = v[0]
                            vals.append(e[0])
                        R["warm"].append({
                            "checkpoint": tag_here, "iters": iters, "layer": layer,
                            "mean": st.mean(vals), "std": st.stdev(vals), "cv": cv(vals),
                            "truth": truth[layer],
                            "rel_bias": st.mean(vals) / truth[layer] - 1.0})
                # Task 3: cold iters=1 vs truth at this checkpoint
                for layer in LAYERS:
                    c1 = [cold(x, y, layer, 1) for _ in range(N_CHAINS)]
                    R["bias"].append({"checkpoint": tag_here, "layer": layer,
                                      "cold1_mean": st.mean(c1), "truth": truth[layer],
                                      "bias_factor": truth[layer] / st.mean(c1)})
                R["fixed"] += fixed_point(tag_here, x, y)
                R["batch"] += batch_spread(tag_here, loader)
                if tag_here == "mid_task1":
                    cache = {}
                    for iters in WARM_ITERS:
                        R["timing"][f"warm_{iters}"] = time_setting(x, y, iters, True, cache)
                    for iters in (1, 20):
                        R["timing"][f"cold_{iters}"] = time_setting(x, y, iters, False, {})

            # ---- Task 2 + 1c: step-to-step series at logged cadence ----
            if step_begin <= step < step_end and step % config.log_interval == 0:
                base = criterion(model(x), y)
                row = {"checkpoint": tag_here, "task": task, "step": step,
                       "alpha_agg": optimizers.get_alpha_agg(optimizer)}
                for layer in LAYERS:
                    row[f"lam1_{layer}"] = optimizers.estimate_hessian_topk(
                        model, base, layer_map[layer], k=1, iters=1)[0]
                    e, v = optimizers.estimate_hessian_topk(
                        model, base, layer_map[layer], k=1, iters=1,
                        v_init=wc_cache.get(layer), return_v=True)
                    wc_cache[layer] = v
                    row[f"lamw_{layer}"] = e[0]
                    row[f"lam100_{layer}"] = optimizers.estimate_hessian_topk(
                        model, base, layer_map[layer], k=1, iters=100)[0]
                R["steps"].append(row)

            optimizer.zero_grad()
            loss = criterion(model(x), y) + spectral_reg()
            loss.backward()
            optimizer.step()
            step += 1

OUT = os.path.expanduser("~/scratch/estimator_probe/results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
