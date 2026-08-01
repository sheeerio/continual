# TASKS 1-3: sign-indefiniteness, spectral gap, and a corrected warm-start
# evaluation with convergence-based stopping.
#
# Task 3a note: the previous probe did NOT share caches across chains or
# settings (chains[iters][layer][ci] is a distinct slot, seeded independently).
# But after a 60-step window every chain converges to the SAME vector, so its
# CV ~1e-7 measured reproducibility, not accuracy. Here bias is measured
# against cold iters=100 over ~50 CONSECUTIVE training steps instead, which is
# the quantity that actually matters.
import os, sys, time, json
import statistics as st
import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_parser
from utils import optimizers
from models import mlp
from datasets import data_loader

parser = get_parser()
config = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np, random
torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

train_dataset, _, _, input_size, _, _ = data_loader.get_dataset(config)
model = mlp.MLP(input_size, config.hidden, 10).to(device)
criterion = nn.CrossEntropyLoss()

layer_map = {}
for name, p in model.named_parameters():
    if p.requires_grad:
        layer_map.setdefault(name.split('.')[0], []).append(p)
layer_groups = [{'params': ps, 'lr': config.lr, 'layer': l} for l, ps in layer_map.items()]
optimizer = torch.optim.Adam(layer_groups, lr=config.lr, betas=(config.beta1, config.beta2))
LAYERS = list(layer_map.keys())

WARM_ITERS = [1, 2, 5]
STEP_WINDOW = 50
R = {"sign": [], "gap": [], "bias": [], "conv": [], "timing": {}, "signwin": []}


def cv(xs):
    m = st.mean(xs)
    return st.stdev(xs) / abs(m) if len(xs) > 1 and m else float("nan")


def spectral_reg():
    r = torch.tensor(0.0, device=device)
    for _n, p in model.named_parameters():
        if p.requires_grad and p.ndim >= 2:
            r = r + (optimizers.power_iteration(p, 1).pow(config.spectral_k) - 1.0).pow(2)
    return r * config.spectral_lambda


# independent caches, one per (setting, layer)
warm_cache = {it: {l: None for l in LAYERS} for it in WARM_ITERS}
conv_cache = {l: None for l in LAYERS}
lmax_cache = {l: None for l in LAYERS}

for task in range(3):
    if task > 0:
        train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
    loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    n_steps = config.epochs * len(loader)
    mid = n_steps // 2
    if task == 0:
        cp, tag = min(200, n_steps // 4), "early_task0"
    else:
        cp, tag = mid, f"mid_task{task}"

    step = 0
    for epoch in range(config.epochs):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device); y = y.to(device)

            if step == cp:
                print(f"[probe] {tag} task={task} step={step}", flush=True)
                base = criterion(model(x), y)
                for layer in LAYERS:
                    lp = layer_map[layer]
                    signed = optimizers.estimate_hessian_topk(model, base, lp, k=1, iters=100)[0]
                    lmax, _it = optimizers.estimate_hessian_lambda_max(model, base, lp, iters=100)
                    lmin = optimizers.estimate_hessian_min_eig(model, base, lp, iters=20)
                    R["sign"].append({"checkpoint": tag, "layer": layer,
                                      "signed_topk": signed, "lambda_max": lmax,
                                      "lambda_min": float(lmin),
                                      "abs_min_gt_max": abs(float(lmin)) > lmax})
                    # TASK 2: top-3 by deflation
                    e3 = optimizers.estimate_hessian_topk(model, base, lp, k=3, iters=100)
                    R["gap"].append({"checkpoint": tag, "layer": layer,
                                     "l1": e3[0], "l2": e3[1], "l3": e3[2]})
                # TASK 3d timing
                if tag == "mid_task1":
                    for it in WARM_ITERS:
                        c = {l: None for l in LAYERS}
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t0 = time.time(); n = 0
                        for _ in range(10):
                            for layer in LAYERS:
                                b2 = criterion(model(x), y)
                                e, v = optimizers.estimate_hessian_topk(
                                    model, b2, layer_map[layer], k=1, iters=it,
                                    v_init=c[layer], return_v=True)
                                c[layer] = v; n += 1
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        R["timing"][f"warm_{it}"] = (time.time() - t0) / n
                    for nm, fn in (("conv_stop", "conv"), ("lambda_max", "lmax")):
                        c = {l: None for l in LAYERS}
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t0 = time.time(); n = 0
                        for _ in range(10):
                            for layer in LAYERS:
                                b2 = criterion(model(x), y)
                                if fn == "conv":
                                    e, v, iu = optimizers.estimate_hessian_topk(
                                        model, b2, layer_map[layer], k=1,
                                        iters=config.hessian_max_iters,
                                        v_init=c[layer], return_v=True,
                                        tol=config.hessian_tol, return_iters=True)
                                    c[layer] = v
                                else:
                                    lm, iu, v = optimizers.estimate_hessian_lambda_max(
                                        model, b2, layer_map[layer],
                                        iters=config.hessian_max_iters,
                                        v_init=c[layer], return_v=True,
                                        tol=config.hessian_tol,
                                        max_iters=config.hessian_max_iters)
                                    c[layer] = v
                                n += 1
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        R["timing"][nm] = (time.time() - t0) / n

            # TASK 3a/3b + TASK 1b: consecutive-step window
            if cp <= step < cp + STEP_WINDOW * config.log_interval and step % config.log_interval == 0:
                base = criterion(model(x), y)
                row = {"checkpoint": tag, "step": step}
                for layer in LAYERS:
                    lp = layer_map[layer]
                    truth = optimizers.estimate_hessian_topk(model, base, lp, k=1, iters=100)[0]
                    row[f"truth_{layer}"] = truth
                    for it in WARM_ITERS:
                        e, v = optimizers.estimate_hessian_topk(
                            model, base, lp, k=1, iters=it,
                            v_init=warm_cache[it][layer], return_v=True)
                        warm_cache[it][layer] = v
                        row[f"warm{it}_{layer}"] = e[0]
                    e, v, iu = optimizers.estimate_hessian_topk(
                        model, base, lp, k=1, iters=config.hessian_max_iters,
                        v_init=conv_cache[layer], return_v=True,
                        tol=config.hessian_tol, return_iters=True)
                    conv_cache[layer] = v
                    row[f"conv_{layer}"] = e[0]
                    row[f"convit_{layer}"] = iu[0]
                    lm, itu, vlm = optimizers.estimate_hessian_lambda_max(
                        model, base, lp, iters=config.hessian_max_iters,
                        v_init=lmax_cache[layer], return_v=True,
                        tol=config.hessian_tol, max_iters=config.hessian_max_iters)
                    lmax_cache[layer] = vlm
                    row[f"lmax_{layer}"] = lm
                    row[f"lmaxit_{layer}"] = itu
                R["bias"].append(row)

            optimizer.zero_grad()
            (criterion(model(x), y) + spectral_reg()).backward()
            optimizer.step()
            step += 1

OUT = os.path.expanduser("~/scratch/spectrum_probe/results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
