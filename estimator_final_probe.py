# TASKS 1b, 2c, 3: residual-criterion cap sweep, cost accounting, and the
# gap-free curvature proxies.
#
# Ground truth is cold iters=200: at lambda2/lambda1 = 0.98 a cold iters=100
# run is itself only ~1 - 0.98^100 = 87% converged, so 100 is not a safe
# reference for measuring few-percent biases.
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
optimizer = torch.optim.Adam(
    [{'params': ps, 'lr': config.lr, 'layer': l} for l, ps in layer_map.items()],
    lr=config.lr, betas=(config.beta1, config.beta2))
LAYERS = list(layer_map.keys())

TRUTH_ITERS = 200
TOLS = [3e-2, 1e-2, 3e-3]
CAPS = [20, 50, 100]
STEP_WINDOW = 50
N_REPEAT = 15
R = {"cap": [], "proxy": [], "fixed": [], "batch": [], "timing": {}, "trainstep": None}


def cv(xs):
    m = st.mean(xs)
    return st.stdev(xs) / abs(m) if len(xs) > 1 and m else float("nan")


def spectral_reg():
    r = torch.tensor(0.0, device=device)
    for _n, p in model.named_parameters():
        if p.requires_grad and p.ndim >= 2:
            r = r + (optimizers.power_iteration(p, 1).pow(config.spectral_k) - 1.0).pow(2)
    return r * config.spectral_lambda


caches = {(t, c): {l: None for l in LAYERS} for t in TOLS for c in CAPS}
proxy_cache = {l: None for l in LAYERS}

for task in range(3):
    if task > 0:
        train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
    loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    n_steps = config.epochs * len(loader)
    mid = n_steps // 2
    cp, tag = (min(200, n_steps // 4), "early_task0") if task == 0 else (mid, f"mid_task{task}")

    step = 0
    for epoch in range(config.epochs):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device); y = y.to(device)

            if cp <= step < cp + STEP_WINDOW * config.log_interval and step % config.log_interval == 0:
                base = criterion(model(x), y)
                for layer in LAYERS:
                    lp = layer_map[layer]
                    truth = optimizers.estimate_hessian_topk(model, base, lp, k=1, iters=TRUTH_ITERS)[0]
                    row = {"checkpoint": tag, "step": step, "layer": layer, "truth": truth}
                    for t in TOLS:
                        for c in CAPS:
                            e, v, iu = optimizers.estimate_hessian_topk(
                                model, base, lp, k=1, iters=c, tol=t,
                                v_init=caches[(t, c)][layer], return_v=True, return_iters=True)
                            caches[(t, c)][layer] = v
                            row[f"lam_t{t}_c{c}"] = e[0]
                            row[f"it_t{t}_c{c}"] = iu[0]
                    # ---- TASK 3 proxies ----
                    e, v, iu = optimizers.estimate_hessian_topk(
                        model, base, lp, k=1, iters=50, tol=1e-2,
                        v_init=proxy_cache[layer], return_v=True, return_iters=True)
                    proxy_cache[layer] = v
                    row["proxy_lam1"] = e[0]
                    row["proxy_trace"] = optimizers.hessian_trace(base, lp, n_samples=10)
                    e3 = optimizers.estimate_hessian_topk(model, base, lp, k=3, iters=50, tol=1e-2)
                    row["proxy_top3mean"] = sum(e3) / 3.0
                    R["cap"].append(row)

            if step == cp:
                print(f"[probe] {tag} step={step}", flush=True)
                base = criterion(model(x), y)
                # fixed-point estimator CV for each proxy (same weights/batch)
                for layer in LAYERS:
                    lp = layer_map[layer]
                    l1 = [optimizers.estimate_hessian_topk(model, base, lp, k=1, iters=50, tol=1e-2)[0]
                          for _ in range(N_REPEAT)]
                    tr = [optimizers.hessian_trace(base, lp, n_samples=10) for _ in range(N_REPEAT)]
                    t3 = [sum(optimizers.estimate_hessian_topk(model, base, lp, k=3, iters=50, tol=1e-2)) / 3.0
                          for _ in range(N_REPEAT)]
                    R["fixed"].append({"checkpoint": tag, "layer": layer,
                                       "cv_lam1": cv(l1), "cv_trace": cv(tr), "cv_top3": cv(t3)})
                # batch-to-batch
                per = {l: {"lam1": [], "trace": [], "top3": []} for l in LAYERS}
                it_ = iter(loader)
                for _ in range(N_REPEAT):
                    try:
                        bx, by = next(it_)
                    except StopIteration:
                        it_ = iter(loader); bx, by = next(it_)
                    bx = bx.view(bx.size(0), -1).to(device); by = by.to(device)
                    b2 = criterion(model(bx), by)
                    for layer in LAYERS:
                        lp = layer_map[layer]
                        per[layer]["lam1"].append(
                            optimizers.estimate_hessian_topk(model, b2, lp, k=1, iters=50, tol=1e-2)[0])
                        per[layer]["trace"].append(optimizers.hessian_trace(b2, lp, n_samples=10))
                        per[layer]["top3"].append(
                            sum(optimizers.estimate_hessian_topk(model, b2, lp, k=3, iters=50, tol=1e-2)) / 3.0)
                for layer, d in per.items():
                    R["batch"].append({"checkpoint": tag, "layer": layer,
                                       "cv_lam1": cv(d["lam1"]), "cv_trace": cv(d["trace"]),
                                       "cv_top3": cv(d["top3"])})
                if tag == "mid_task1":
                    def timeit(fn):
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        t0 = time.time(); n = 0
                        for _ in range(10):
                            for layer in LAYERS:
                                fn(criterion(model(x), y), layer_map[layer]); n += 1
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        return (time.time() - t0) / n
                    R["timing"]["lam1_tol1e-2_cap50"] = timeit(
                        lambda b, lp: optimizers.estimate_hessian_topk(model, b, lp, k=1, iters=50, tol=1e-2))
                    R["timing"]["trace_h10"] = timeit(
                        lambda b, lp: optimizers.hessian_trace(b, lp, n_samples=10))
                    R["timing"]["top3_tol1e-2"] = timeit(
                        lambda b, lp: optimizers.estimate_hessian_topk(model, b, lp, k=3, iters=50, tol=1e-2))
                    R["timing"]["min_eig_iters20"] = timeit(
                        lambda b, lp: optimizers.estimate_hessian_min_eig(model, b, lp, iters=20))
                    # bare training step cost
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    t0 = time.time()
                    for _ in range(20):
                        optimizer.zero_grad()
                        (criterion(model(x), y) + spectral_reg()).backward()
                        optimizer.step()
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    R["trainstep"] = (time.time() - t0) / 20

            optimizer.zero_grad()
            (criterion(model(x), y) + spectral_reg()).backward()
            optimizer.step()
            step += 1

OUT = os.path.expanduser("~/scratch/estfinal_probe/results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(R, open(OUT, "w"), indent=1)
print("wrote", OUT, flush=True)
