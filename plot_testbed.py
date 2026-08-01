# STEP 5: testbed plots. P1 basin, P2 trajectories, P3 diagnostics.
#
# Encoding choice, and why it matters for this figure set: the whole point of
# the four arms is CONTRIBUTION ATTRIBUTION, so the encoding is built to make
# the two contributions readable as separate visual operations --
#   colour     = penalty variant  (static / adaptive-inv / adaptive-sat)
#   linestyle  = pl_lyapunov scheduler off (solid) / on (dashed)
# Comparing a solid line to its own dashed twin IS the scheduler's
# contribution; comparing colours at matched linestyle IS the modulation's.
# Six arms therefore need only THREE categorical hues rather than six, which
# also clears the strict all-pairs colour gate instead of the weaker adjacent
# one (validated: worst all-pairs CVD dE 9.2 light / 9.4 dark, normal-vision
# 24.0 / 20.9).
#
# Vanilla is a NEUTRAL DASHED REFERENCE, not a categorical slot -- it is the
# baseline the arms are measured against, not a seventh peer series.
import csv, glob, os, math, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

ROOT = os.path.expanduser("~/scratch/testbed")
CELLDIR = f"{ROOT}/cells"
OUTDIR = os.path.expanduser(f"~/scratch/testbed_plots/{date.today().isoformat()}")
os.makedirs(OUTDIR, exist_ok=True)

COEFFS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
SEEDS = [0, 1, 2]

# (arm_key, label, colour_slot, linestyle)
LIGHT = {"static": "#2a78d6", "inv": "#eb6834", "sat": "#1baf7a"}
ARMS = [
    ("A_static",             "static spectral",              "static", "-"),
    ("B_static_sched",       "static + scheduler",           "static", "--"),
    ("C_adaptive_inv",       "adaptive inv",                 "inv",    "-"),
    ("D_adaptive_inv_sched", "adaptive inv + scheduler",     "inv",    "--"),
    ("C_adaptive_sat",       "adaptive sat",                 "sat",    "-"),
    ("D_adaptive_sat_sched", "adaptive sat + scheduler",     "sat",    "--"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def read_cell(name):
    f = f"{CELLDIR}/{name}.csv"
    if not (os.path.exists(f) and os.path.getsize(f) > 0):
        return None
    return list(csv.DictReader(open(f)))[0]


def agg(arm, c, field):
    """mean/std of a scalar field over seeds."""
    vals = []
    for s in SEEDS:
        r = read_cell(f"{arm}_c{c}_seed{s}")
        if r and r.get(field) not in (None, "", "nan"):
            vals.append(float(r[field]))
    if not vals:
        return float("nan"), 0.0, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def traj(arm, c):
    """mean/std task-accuracy trajectory over seeds."""
    series = []
    for s in SEEDS:
        r = read_cell(f"{arm}_c{c}_seed{s}")
        if r and r.get("task_acc_traj"):
            series.append([float(x) for x in r["task_acc_traj"].split(";")])
    if not series:
        return [], [], []
    n = min(len(x) for x in series)
    xs = list(range(1, n + 1))
    mean = [st.mean([x[i] for x in series]) for i in range(n)]
    sd = [st.stdev([x[i] for x in series]) if len(series) > 1 else 0.0 for i in range(n)]
    return xs, mean, sd


def vanilla(field):
    """Returns (mean, std, n) -- n reported, never assumed, so a partially
    complete vanilla reference is visible rather than printed as if it were 3."""
    vals = []
    for s in SEEDS:
        r = read_cell(f"vanilla_seed{s}")
        if r and r.get(field) not in (None, "", "nan"):
            vals.append(float(r[field]))
    if not vals:
        return float("nan"), 0.0, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def taskdiag_mean(arm, c, field):
    """Mean over tasks and layers of a per-task diagnostic column, then over seeds."""
    per_seed = []
    for s in SEEDS:
        f = f"{CELLDIR}/{arm}_c{c}_seed{s}_taskdiag.csv"
        if not (os.path.exists(f) and os.path.getsize(f) > 0):
            continue
        vals = []
        for row in csv.DictReader(open(f)):
            v = row.get(field, "")
            if v not in ("", None, "nan"):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not math.isnan(fv):
                    vals.append(fv)
        if vals:
            per_seed.append(st.mean(vals))
    if not per_seed:
        return float("nan"), 0.0
    return st.mean(per_seed), (st.stdev(per_seed) if len(per_seed) > 1 else 0.0)


X = list(range(len(COEFFS)))


def style_x(ax):
    ax.set_xticks(X)
    ax.set_xticklabels(COEFFS)
    ax.set_xlabel("coefficient (log-spaced)")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)


# ---------------- P1: basin ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, field, ylab in ((axes[0], "auc", "AUC"), (axes[1], "slope", "slope")):
    for arm, label, slot, ls in ARMS:
        m = [agg(arm, c, field)[0] for c in COEFFS]
        sd = [agg(arm, c, field)[1] for c in COEFFS]
        ax.plot(X, m, ls, color=LIGHT[slot], label=label, marker="o", markersize=4)
        ax.fill_between(X, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=LIGHT[slot], alpha=0.13, linewidth=0)
    vm, vs, vn = vanilla(field)
    if vm == vm:
        ax.axhline(vm, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
        ax.fill_between([X[0], X[-1]], vm - vs, vm + vs, color=MUTED, alpha=0.10, linewidth=0)
        ax.annotate("vanilla", (X[-1], vm), textcoords="offset points", xytext=(-4, 5),
                    ha="right", color=MUTED, fontsize=8)
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab} vs coefficient", loc="left")
    style_x(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, fontsize=8)
fig.suptitle("P1  Basin: spectral testbed, 20 tasks x 50 epochs, 3 seeds (shaded = seed std)",
             x=0.01, y=0.985, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.845])
fig.savefig(f"{OUTDIR}/P1_basin.png", dpi=170)
plt.close(fig)

# ---------------- P2: trajectories ----------------
fig, axes = plt.subplots(1, len(COEFFS), figsize=(4 * len(COEFFS), 3.8), sharey=True)
for ax, c in zip(axes, COEFFS):
    for arm, label, slot, ls in ARMS:
        xs, m, sd = traj(arm, c)
        if not xs:
            continue
        ax.plot(xs, m, ls, color=LIGHT[slot], label=label, linewidth=1.8)
        ax.fill_between(xs, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=LIGHT[slot], alpha=0.12, linewidth=0)
    # Vanilla has no coefficient axis, so it is read directly rather than via
    # traj(arm, c) -- same grey dashed reference repeated in every panel.
    series = []
    for s in SEEDS:
        r = read_cell(f"vanilla_seed{s}")
        if r and r.get("task_acc_traj"):
            series.append([float(x) for x in r["task_acc_traj"].split(";")])
    if series:
        n = min(len(x) for x in series)
        vxs = list(range(1, n + 1))
        vm = [st.mean([x[i] for x in series]) for i in range(n)]
        ax.plot(vxs, vm, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4, zorder=1)
    ax.set_title(f"c = {c}", loc="left")
    ax.set_xlabel("task")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
axes[0].set_ylabel("task accuracy")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=6, fontsize=8)
fig.suptitle("P2  Trajectories: task accuracy vs task, one panel per coefficient "
             "(mean +/- seed std; grey dashed = vanilla)", x=0.005, y=0.985, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.87])
fig.savefig(f"{OUTDIR}/P2_trajectories.png", dpi=170)
plt.close(fig)

# ---------------- P3: diagnostics ----------------
PANELS = [("mean_off_diag_mean", "coherence  mean_off_diag"),
          ("tau_mean", "mean tau"),
          ("adaptive_factor_mean", "mean adaptive_factor")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
for ax, (field, title) in zip(axes, PANELS):
    plotted = False
    for arm, label, slot, ls in ARMS:
        m, sd = [], []
        for c in COEFFS:
            a, b = taskdiag_mean(arm, c, field)
            m.append(a); sd.append(b)
        if all(x != x for x in m):
            continue          # arm has no data for this field (e.g. static has no adaptive_factor)
        plotted = True
        ax.plot(X, m, ls, color=LIGHT[slot], label=label, marker="o", markersize=4)
        ax.fill_between(X, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=LIGHT[slot], alpha=0.13, linewidth=0)
    if field == "adaptive_factor_mean":
        ax.set_yscale("log")
        ax.set_title(title + "  (adaptive arms only, log y)", loc="left")
    else:
        ax.set_title(title, loc="left")
    ax.set_ylabel(title)
    style_x(ax)
    if not plotted:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=6, fontsize=8)
fig.suptitle("P3  Diagnostics vs coefficient, from the per-task diagnostic CSV "
             "(mean over tasks and layers; shaded = seed std)", x=0.005, y=0.985, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.865])
fig.savefig(f"{OUTDIR}/P3_diagnostics.png", dpi=170)
plt.close(fig)

# ---------------- summary table ----------------
rows = []
for arm, label, _, _ in ARMS:
    for c in COEFFS:
        am, asd, n = agg(arm, c, "auc")
        sm, ssd, _ = agg(arm, c, "slope")
        rows.append((label, c, am, asd, sm, ssd, n))
vm_a, vs_a, vn_a = vanilla("auc"); vm_s, vs_s, _ = vanilla("slope")

with open(f"{OUTDIR}/summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm", "coefficient", "auc_mean", "auc_std", "slope_mean", "slope_std", "n_seeds"])
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])
    w.writerow(["vanilla", "-", f"{vm_a:.5f}", f"{vs_a:.5f}", f"{vm_s:.6f}", f"{vs_s:.6f}", vn_a])

print(f"{'arm':<28}{'c':>7}{'AUC mean':>11}{'+-std':>9}{'slope mean':>12}{'+-std':>10}{'n':>3}")
print("-" * 80)
for r in rows:
    print(f"{r[0]:<28}{r[1]:>7}{r[2]:>11.5f}{r[3]:>9.5f}{r[4]:>12.6f}{r[5]:>10.6f}{r[6]:>3}")
print(f"{'vanilla (reference)':<28}{'-':>7}{vm_a:>11.5f}{vs_a:>9.5f}{vm_s:>12.6f}{vs_s:>10.6f}{vn_a:>3}")
print(f"\nplots + summary.csv -> {OUTDIR}")
