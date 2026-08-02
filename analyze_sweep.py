# Sweep readout. Order of output is deliberate: the width-in-decades table
# first (that is the robustness metric), then the summary table, then plots.
#
# AUC and slope only. final_acc is carried in the CSV but is never a conclusion.
import csv, os, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.environ["HOME"]
CD = f"{H}/scratch/sweep/cells"
OUT = f"{H}/scratch/sweep/plots"
os.makedirs(OUT, exist_ok=True)

COEFS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1"]
CV = [float(c) for c in COEFS]
LOGC = np.log10(CV)
SEEDS = (0, 1, 2)
ARMS = [("stat", "static spectral", "#2a78d6"),
        ("admed", "adaptive sat, tau_ref=median", "#eb6834"),
        ("adfix", "adaptive sat, tau_ref=fixed", "#1baf7a")]
THRESH = [0.40, 0.45, 0.50]


def load(name):
    p = f"{CD}/{name}.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return rows[0] if rows else None


def collect(arm):
    """-> auc[coef, seed], slope[coef, seed], with nan for missing cells."""
    a = np.full((len(COEFS), len(SEEDS)), np.nan)
    s = np.full((len(COEFS), len(SEEDS)), np.nan)
    for i, c in enumerate(COEFS):
        for j, sd in enumerate(SEEDS):
            r = load(f"{arm}_{c}_s{sd}")
            if r:
                a[i, j] = float(r["auc"])
                s[i, j] = float(r["slope"])
    return a, s


van = np.array([float(load(f"van_s{sd}")["auc"]) for sd in SEEDS
                if load(f"van_s{sd}")])
van_slope = np.array([float(load(f"van_s{sd}")["slope"]) for sd in SEEDS
                      if load(f"van_s{sd}")])

data = {}
for arm, label, col in ARMS:
    data[arm] = collect(arm)

missing = sum(int(np.isnan(data[a][0]).sum()) for a, _, _ in ARMS) + (3 - len(van))
print(f"cells present: {75 - missing}/75" + ("" if not missing else f"  ({missing} MISSING)"))
print()


# ---------------- 1. width in decades ----------------
def width_decades(auc_mean, thr):
    """Total extent (in decades of coefficient) where the seed-mean AUC is at or
    above thr, linearly interpolated in log10(c). Returns (width, censored),
    where censored is True if the region runs off an end of the measured axis --
    the true width is then a lower bound."""
    y = auc_mean
    ok = ~np.isnan(y)
    x, y = LOGC[ok], y[ok]
    above = y >= thr
    if not above.any():
        return 0.0, False
    w = 0.0
    for i in range(len(x) - 1):
        a, b = above[i], above[i + 1]
        if a and b:
            w += x[i + 1] - x[i]
        elif a != b:
            # crossing: interpolate where y hits thr
            t = (thr - y[i]) / (y[i + 1] - y[i])
            xc = x[i] + t * (x[i + 1] - x[i])
            w += (xc - x[i]) if a else (x[i + 1] - xc)
    censored = bool(above[0] or above[-1])
    return w, censored


print("=" * 74)
print("WIDTH IN DECADES above AUC threshold  (seed-mean, interpolated in log10 c)")
print("  '>' = region runs off the measured axis; the value is a lower bound")
print("=" * 74)
print(f"{'arm':<32}" + "".join(f"{f'AUC>={t:.2f}':>13}" for t in THRESH))
for arm, label, _ in ARMS:
    am = np.nanmean(data[arm][0], axis=1)
    cells = []
    for t in THRESH:
        w, cen = width_decades(am, t)
        cells.append(f"{'>' if cen else ' '}{w:>11.2f}")
    print(f"{label:<32}" + "".join(f"{c:>13}" for c in cells))
if len(van):
    print(f"{'(vanilla reference AUC)':<32}{np.mean(van):>13.5f}")
print()


# ---------------- 2. summary table ----------------
print("=" * 74)
print("SUMMARY  (mean +- std over seeds)")
print("=" * 74)
print(f"{'arm':<14}{'coef':>8}{'AUC':>10}{'sd':>9}{'slope':>11}{'sd':>10}{'n':>4}")
for arm, label, _ in ARMS:
    a, s = data[arm]
    for i, c in enumerate(COEFS):
        n = int((~np.isnan(a[i])).sum())
        if not n:
            continue
        print(f"{arm:<14}{c:>8}{np.nanmean(a[i]):>10.5f}{np.nanstd(a[i]):>9.5f}"
              f"{np.nanmean(s[i]):>11.6f}{np.nanstd(s[i]):>10.6f}{n:>4}")
if len(van):
    print(f"{'vanilla':<14}{'-':>8}{van.mean():>10.5f}{van.std():>9.5f}"
          f"{van_slope.mean():>11.6f}{van_slope.std():>10.6f}{len(van):>4}")
print(f"\nresolvability threshold (seed std): 0.00553")
print()


# ---------------- 3. P1 basin ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, idx, ylab in ((axes[0], 0, "AUC"), (axes[1], 1, "slope")):
    for arm, label, col in ARMS:
        m = np.nanmean(data[arm][idx], axis=1)
        sd = np.nanstd(data[arm][idx], axis=1)
        ax.plot(CV, m, "-o", color=col, label=label, ms=4, lw=1.6)
        ax.fill_between(CV, m - sd, m + sd, color=col, alpha=0.18, lw=0)
    ref = van if idx == 0 else van_slope
    if len(ref):
        ax.axhline(ref.mean(), color="#666666", ls="--", lw=1.2, label="vanilla")
    ax.set_xscale("log")
    ax.set_xlabel("coefficient (spectral_lambda / reg_sensitivity)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("P1  basin: AUC and slope vs coefficient (3 seeds, band = +-1 sd)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/P1_basin.png", dpi=160)
print(f"wrote {OUT}/P1_basin.png")


# ---------------- 4. dead units per layer per task ----------------
def deadtraj(name):
    p = f"{CD}/{name}_taskdiag.csv"
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        v = r.get("n_dead", "")
        if v in ("", None):
            continue
        try:
            out.setdefault(r["layer"], []).append((int(r["task"]), int(float(v))))
        except ValueError:
            continue
    return {k: [d for _, d in sorted(v)] for k, v in out.items()}


LAYERS = ["fc1", "fc2"]
fig, axes = plt.subplots(1, len(LAYERS), figsize=(11, 4.0), sharey=True)
PEAK = "1e-2"
series = [("van_s0", "vanilla", "#666666")] + \
         [(f"{arm}_{PEAK}_s0", label, col) for arm, label, col in ARMS]
for ax, layer in zip(axes, LAYERS):
    for nm, label, col in series:
        d = deadtraj(nm).get(layer)
        if d:
            ax.plot(range(len(d)), d, "-", color=col, label=label, lw=1.6)
    ax.set_title(f"{layer}  (256 units)", fontsize=10)
    ax.set_xlabel("task")
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel("strictly dead units (n_dead)")
axes[0].legend(fontsize=7.5, frameon=False)
fig.suptitle(f"Strictly dead units per layer per task, seed 0, coefficient {PEAK}",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/P4_dead_units.png", dpi=160)
print(f"wrote {OUT}/P4_dead_units.png")

print("\nn_dead final-task counts (seed 0, coef %s):" % PEAK)
print(f"{'arm':<32}" + "".join(f"{l:>10}" for l in LAYERS))
for nm, label, _ in series:
    t = deadtraj(nm)
    print(f"{label:<32}" + "".join(
        f"{(t[l][-1] if t.get(l) else float('nan')):>10}" for l in LAYERS))
