# TASK 1 figure: P1 basin over the EXTENDED coefficient axis (1e-4 .. 3e-1).
# Reads 1e-4..1e-2 from ~/scratch/testbed (the 93-cell grid) and 3e-2..3e-1
# from ~/scratch/testbed_extend. Same encoding as plot_testbed.py: colour =
# penalty variant, and only the two arms Task 1 covers are drawn.
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

OLD = os.path.expanduser("~/scratch/testbed")
NEW = os.path.expanduser("~/scratch/testbed_extend")
OUTDIR = os.path.expanduser(f"~/scratch/testbed_plots/{date.today().isoformat()}_extended")
os.makedirs(OUTDIR, exist_ok=True)

AX = [("1e-4", OLD), ("3e-4", OLD), ("1e-3", OLD), ("3e-3", OLD),
      ("1e-2", OLD), ("3e-2", NEW), ("1e-1", NEW), ("3e-1", NEW)]
SEEDS = [0, 1, 2]
ARMS = [("A_static", "static spectral", "#2a78d6"),
        ("C_adaptive_sat", "adaptive sat", "#1baf7a")]
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


def agg(d, arm, c, f):
    v = []
    for s in SEEDS:
        p = f"{d}/cells/{arm}_c{c}_seed{s}.csv"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            r = list(csv.DictReader(open(p)))[0]
            if r.get(f) not in ("", "nan", None):
                v.append(float(r[f]))
    if not v:
        return float("nan"), 0.0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def vanilla(f):
    v = []
    for s in SEEDS:
        p = f"{OLD}/cells/vanilla_seed{s}.csv"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            r = list(csv.DictReader(open(p)))[0]
            if r.get(f) not in ("", "nan", None):
                v.append(float(r[f]))
    if not v:
        return float("nan"), 0.0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


X = list(range(len(AX)))
OLD_EDGE = 4          # index of 1e-2, the right edge of the original grid

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, field, ylab in ((axes[0], "auc", "AUC"), (axes[1], "slope", "slope")):
    # Mark where the original axis stopped -- everything right of this is new.
    ax.axvspan(OLD_EDGE, X[-1], color=MUTED, alpha=0.05, linewidth=0)
    ax.axvline(OLD_EDGE, color=MUTED, linewidth=1, linestyle=":", alpha=0.8)
    for arm, label, col in ARMS:
        m = [agg(d, arm, c, field)[0] for c, d in AX]
        sd = [agg(d, arm, c, field)[1] for c, d in AX]
        ax.plot(X, m, "-", color=col, label=label, marker="o", markersize=4.5)
        ax.fill_between(X, [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)],
                        color=col, alpha=0.14, linewidth=0)
        if field == "auc":
            pk = max(range(len(m)), key=lambda i: m[i])
            ax.plot([pk], [m[pk]], marker="o", markersize=10, markerfacecolor="none",
                    markeredgecolor=col, markeredgewidth=1.8)
            ax.annotate(f"peak {AX[pk][0]}", (pk, m[pk]), textcoords="offset points",
                        xytext=(0, 13), ha="center", color=col, fontsize=8)
    vm, vs = vanilla(field)
    if vm == vm:
        ax.axhline(vm, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
        ax.annotate("vanilla", (X[-1], vm), textcoords="offset points", xytext=(-4, 5),
                    ha="right", color=MUTED, fontsize=8)
    ax.set_xticks(X)
    ax.set_xticklabels([c for c, _ in AX])
    ax.set_xlabel("coefficient (log-spaced)")
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab} vs coefficient", loc="left")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)

axes[0].annotate("original grid", (OLD_EDGE / 2, 0.13), xycoords=("data", "axes fraction"),
                 ha="center", color=MUTED, fontsize=8)
axes[0].annotate("Task 1 extension", ((OLD_EDGE + X[-1]) / 2, 0.13),
                 xycoords=("data", "axes fraction"), ha="center", color=MUTED, fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9)
fig.suptitle("P1 (extended)  Basin over the full coefficient axis, 20 tasks x 50 epochs, "
             "3 seeds (shaded = seed std)", x=0.01, y=0.985, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.865])
fig.savefig(f"{OUTDIR}/P1_basin_extended.png", dpi=170)
plt.close(fig)
print(f"-> {OUTDIR}/P1_basin_extended.png")
