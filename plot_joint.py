"""
Plot Q2 results: joint LR-reg controller vs plain versions, across regs.

Compares four methods per reg: static, static_joint, adaptive_inv, adaptive_inv_joint.
The key comparison is (static vs static_joint) and (adaptive_inv vs adaptive_inv_joint).

Outputs:
  - joint_auc_{reg}.png:      AUC vs coefficient, faceted by (lr), all four methods.
  - joint_delta_{reg}.png:    delta AUC (joint minus plain) vs coefficient.
  - joint_regime_{reg}.png:   fraction of time in shock regime, per method.

Usage:
  python plot_joint.py --exp_name joint_20260713 --wandb_entity sheerio --wandb_project sweep
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--wandb_entity", type=str, default="sheerio")
parser.add_argument("--wandb_project", type=str, default="sweep")
parser.add_argument("--out_dir", type=str, default="./plots")
parser.add_argument("--regs", type=str, nargs="+",
                    default=["l2", "spectral", "wass"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

METHOD_COLORS = {
    "static":             "#c44e52",  # red
    "static_joint":       "#8c2d40",  # dark red
    "adaptive_inv":       "#4c72b0",  # blue
    "adaptive_inv_joint": "#2a4a70",  # dark blue
}
METHOD_ORDER = ["static", "static_joint", "adaptive_inv", "adaptive_inv_joint"]

METHOD_LINESTYLE = {
    "static":             "-",
    "static_joint":       "--",
    "adaptive_inv":       "-",
    "adaptive_inv_joint": "--",
}

# ---- pull runs ----
print(f"Fetching runs from {args.wandb_entity}/{args.wandb_project} group={args.exp_name}...")
api = wandb.Api()
runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}",
                filters={"group": args.exp_name})

NAME_RE = re.compile(
    r"reg-(?P<reg>[^_]+)_method-(?P<method>[a-z_]+?)_lr-(?P<lr>[^_]+)_coef-(?P<coef>[^_]+)"
)

records = []
for run in runs:
    m = NAME_RE.match(run.name)
    if m is None:
        print(f"skip: {run.name}")
        continue
    d = m.groupdict()
    d["lr"] = float(d["lr"])
    d["coef"] = float(d["coef"])
    d["run_id"] = run.id
    d["seed"] = run.config.get("random_seed", -1)

    hist = run.history(keys=["task_acc"], pandas=True)
    task_accs = hist["task_acc"].dropna().tolist()
    if len(task_accs) < 3:
        d["auc"] = np.nan
        d["slope"] = np.nan
    else:
        d["auc"] = float(np.mean(task_accs))
        x = np.arange(len(task_accs))
        d["slope"] = float(np.polyfit(x, task_accs, 1)[0])

    # regime fraction (only meaningful for _joint runs)
    if "joint" in d["method"]:
        # pick fc1 as a representative layer for the regime signal
        try:
            regime_hist = run.history(keys=["fc1/joint_regime"], pandas=True)
            regime_vals = regime_hist["fc1/joint_regime"].dropna().values
            d["regime_frac_shock"] = float(np.mean(regime_vals)) if len(regime_vals) else np.nan
        except Exception:
            d["regime_frac_shock"] = np.nan
    else:
        d["regime_frac_shock"] = np.nan

    records.append(d)

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs.")

group_cols = ["reg", "method", "lr", "coef"]
agg = df.groupby(group_cols).agg(
    auc_mean=("auc", "mean"),
    auc_std=("auc", "std"),
    slope_mean=("slope", "mean"),
    slope_std=("slope", "std"),
    regime_mean=("regime_frac_shock", "mean"),
    n=("run_id", "count"),
).reset_index()

# ---- Plot 1: AUC curves, all four methods overlaid ----
def plot_joint_auc(agg):
    for reg in args.regs:
        sub = agg[agg["reg"] == reg]
        if len(sub) == 0:
            continue
        lrs = sorted(sub["lr"].unique())
        fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 4.5), squeeze=False)

        for ci, lr in enumerate(lrs):
            ax = axes[0, ci]
            for method in METHOD_ORDER:
                cell = sub[(sub["lr"] == lr) & (sub["method"] == method)].sort_values("coef")
                if len(cell) == 0:
                    continue
                mean = cell["auc_mean"].values
                std = cell["auc_std"].fillna(0).values
                coefs = cell["coef"].values
                ax.plot(coefs, mean,
                        color=METHOD_COLORS[method],
                        linestyle=METHOD_LINESTYLE[method],
                        marker="o", label=method, linewidth=2)
                ax.fill_between(coefs, mean - std, mean + std,
                                color=METHOD_COLORS[method], alpha=0.15)

            ax.set_xscale("log")
            ax.set_xlabel("coefficient")
            ax.set_ylabel("AUC (mean task_acc)")
            ax.set_title(f"reg={reg}, lr={lr}")
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"Joint controller AUC [{reg}]", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"joint_auc_{reg}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

plot_joint_auc(agg)

# ---- Plot 2: delta AUC (joint minus plain) ----
def plot_joint_delta(agg):
    for reg in args.regs:
        sub = agg[agg["reg"] == reg]
        if len(sub) == 0:
            continue

        # merge joint and plain versions
        pivot = sub.pivot_table(index=["lr", "coef"], columns="method",
                                values="auc_mean").reset_index()

        lrs = sorted(pivot["lr"].unique())
        fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 4.5), squeeze=False)

        for ci, lr in enumerate(lrs):
            ax = axes[0, ci]
            row = pivot[pivot["lr"] == lr].sort_values("coef")

            if "static" in row.columns and "static_joint" in row.columns:
                delta_static = row["static_joint"] - row["static"]
                ax.plot(row["coef"], delta_static, "-o",
                        color=METHOD_COLORS["static_joint"],
                        label="static_joint - static", linewidth=2)

            if "adaptive_inv" in row.columns and "adaptive_inv_joint" in row.columns:
                delta_adap = row["adaptive_inv_joint"] - row["adaptive_inv"]
                ax.plot(row["coef"], delta_adap, "-o",
                        color=METHOD_COLORS["adaptive_inv_joint"],
                        label="adaptive_inv_joint - adaptive_inv", linewidth=2)

            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("coefficient")
            ax.set_ylabel("delta AUC (joint - plain)")
            ax.set_title(f"reg={reg}, lr={lr}")
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=9, loc="best")

        fig.suptitle(f"Joint controller lift [{reg}]", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"joint_delta_{reg}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

plot_joint_delta(agg)

# ---- Plot 3: shock regime fraction ----
def plot_regime(agg):
    for reg in args.regs:
        sub = agg[(agg["reg"] == reg) & (agg["method"].str.contains("joint"))]
        if len(sub) == 0:
            continue
        lrs = sorted(sub["lr"].unique())
        fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 4.5), squeeze=False)

        for ci, lr in enumerate(lrs):
            ax = axes[0, ci]
            for method in ["static_joint", "adaptive_inv_joint"]:
                cell = sub[(sub["lr"] == lr) & (sub["method"] == method)].sort_values("coef")
                if len(cell) == 0:
                    continue
                ax.plot(cell["coef"], cell["regime_mean"], "-o",
                        color=METHOD_COLORS[method],
                        label=method, linewidth=2)

            ax.set_xscale("log")
            ax.set_xlabel("coefficient")
            ax.set_ylabel("fraction of steps in shock regime (fc1)")
            ax.set_title(f"reg={reg}, lr={lr}")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(fontsize=9, loc="best")

        fig.suptitle(f"Joint controller regime distribution [{reg}]", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"joint_regime_{reg}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

plot_regime(agg)

# ---- summary CSV ----
summary_path = os.path.join(args.out_dir, "joint_summary.csv")
agg.to_csv(summary_path, index=False)
print(f"wrote {summary_path}")
