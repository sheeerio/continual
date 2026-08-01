# TASK 1: is tau measuring sharpness volatility, or estimator noise?
#
# tau = 2*lam_mean/(lam_var + lam_mean^2 + eps) is dominated by lam_var, and
# the adaptive method computes lam with estimate_hessian_topk(..., iters=1) --
# ONE power-iteration step from a random init. This script separates the three
# sources of spread in that number:
#   (a) estimator noise      : same weights, same batch, different random init
#   (c) batch-to-batch       : same weights, iters=100, different minibatches
#   (b) observed step-to-step: consecutive logged steps during real training
# and prices the fix by timing iters in {1, 5, 20}.
#
# Standalone rather than bolted into implicit_regularization.py: it needs to
# freeze weights and re-probe the SAME state many times, which the training
# loop cannot do without perturbing the run it is measuring.
import os, sys, time, math, json
import statistics as st
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_parser
from utils import misc, optimizers
from models import mlp
from datasets import data_loader
import torch.utils.data as data

parser = get_parser()
config = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(config.seed)
import numpy as np, random
np.random.seed(config.seed); random.seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)

train_dataset, test_dataset, in_ch, input_size, MEAN, STD = data_loader.get_dataset(config)
model = mlp.MLP(input_size, config.hidden, 10).to(device)
criterion = nn.CrossEntropyLoss()


def make_layer_groups(m, base_lr):
    lmap = {}
    for name, p in m.named_parameters():
        if p.requires_grad:
            lmap.setdefault(name.split('.')[0], []).append(p)
    groups = [{'params': ps, 'lr': base_lr, 'layer': l} for l, ps in lmap.items()]
    return lmap, groups


layer_map, layer_groups = make_layer_groups(model, config.lr)
optimizer = torch.optim.Adam(layer_groups, lr=config.lr,
                             betas=(config.beta1, config.beta2))

ITERS_GRID = [1, 5, 20, 100]
N_REPEAT = 30
results = {"a": [], "b": [], "c": [], "d": {}}


def cv(xs):
    if len(xs) < 2:
        return float("nan")
    m = st.mean(xs)
    return st.stdev(xs) / abs(m) if m else float("nan")


def fixed_point_variance(tag, inputs, labels):
    """(a) same weights, same batch, N_REPEAT random inits, per iters setting."""
    out = []
    for iters in ITERS_GRID:
        for layer, l_params in layer_map.items():
            vals = []
            for _ in range(N_REPEAT):
                base = criterion(model(inputs), labels)
                lam = optimizers.estimate_hessian_topk(model, base, l_params,
                                                       k=1, iters=iters)[0]
                vals.append(float(lam))
            out.append({"checkpoint": tag, "iters": iters, "layer": layer,
                        "mean": st.mean(vals), "std": st.stdev(vals),
                        "min": min(vals), "max": max(vals), "cv": cv(vals)})
    return out


def batch_variance(tag, loader):
    """(c) same weights, iters=100, N_REPEAT DIFFERENT minibatches."""
    out = []
    per_layer = {l: [] for l in layer_map}
    it = iter(loader)
    for _ in range(N_REPEAT):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x = x.view(x.size(0), -1).to(device); y = y.to(device)
        base = criterion(model(x), y)
        for layer, l_params in layer_map.items():
            lam = optimizers.estimate_hessian_topk(model, base, l_params,
                                                   k=1, iters=100)[0]
            per_layer[layer].append(float(lam))
    for layer, vals in per_layer.items():
        out.append({"checkpoint": tag, "layer": layer, "mean": st.mean(vals),
                    "std": st.stdev(vals), "min": min(vals), "max": max(vals),
                    "cv": cv(vals)})
    return out


def time_iters(inputs, labels):
    """(d) wall-clock per estimate_hessian_topk call at each iters setting."""
    out = {}
    for iters in [1, 5, 20]:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        n = 0
        for _ in range(10):
            for layer, l_params in layer_map.items():
                base = criterion(model(inputs), labels)
                optimizers.estimate_hessian_topk(model, base, l_params, k=1, iters=iters)
                n += 1
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        out[iters] = (time.time() - t0) / n
    return out


# ---------------- training, with probes at the requested checkpoints -------
CHECKPOINTS = {}          # tag -> (task, step_within_task)
CHECKPOINTS["start_task0"] = (0, 0)
CHECKPOINTS["mid_task1"] = (1, None)     # None -> half way through the task
CHECKPOINTS["mid_task2"] = (2, None)

fixed_batch = None
for task in range(3):
    if task > 0:
        # Use the SAME task-boundary transform as implicit_regularization.py.
        # A hand-rolled label permutation is wrong here: train_dataset is a
        # Subset (the 10,600-image MNIST slice), so it has no .targets --
        # randomize_targets handles the Subset indirection and applies the
        # ns-fraction random relabelling the testbed actually uses.
        train_dataset = optimizers.randomize_targets(train_dataset, config.ns)
    loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    n_steps = config.epochs * len(loader)
    mid = n_steps // 2
    step = 0
    for epoch in range(config.epochs):
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device); y = y.to(device)

            tag = None
            if task == 0 and step == 0:
                tag = "start_task0"
            elif task == 1 and step == mid:
                tag = "mid_task1"
            elif task == 2 and step == mid:
                tag = "mid_task2"

            if tag:
                print(f"[probe] {tag}: task={task} step={step}", flush=True)
                results["a"] += fixed_point_variance(tag, x, y)
                results["c"] += batch_variance(tag, loader)
                if tag == "mid_task1":
                    results["d"]["timing_s_per_call"] = time_iters(x, y)

            # (b) observed step-to-step spread of lam and alpha_agg, 50
            # consecutive LOGGED steps in task 1, at the method's own iters=1.
            if task == 1 and mid <= step < mid + 50 * config.log_interval \
                    and step % config.log_interval == 0:
                base = criterion(model(x), y)
                row = {"task": task, "step": step,
                       "alpha_agg": optimizers.get_alpha_agg(optimizer)}
                for layer, l_params in layer_map.items():
                    row[f"lam_{layer}"] = float(
                        optimizers.estimate_hessian_topk(model, base, l_params,
                                                         k=1, iters=1)[0])
                results["b"].append(row)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

OUT = os.path.expanduser("~/scratch/variance_probe/results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print("wrote", OUT, flush=True)
