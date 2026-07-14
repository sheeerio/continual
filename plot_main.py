"""
Plot Q1 results: adaptive_saturating vs adaptive_inv vs static, across regs.

Outputs:
  - grid_auc_{reg}.png:   AUC vs coefficient, faceted by (lr), one line per method.
  - grid_slope_{reg}.png: slope vs coefficient, same faceting.
  - coherence_{reg}.png:  coherence trajectories at peak coef, one panel per method.

Usage:
  python plot_main.py --exp_name main_20260713 --wandb_entity sheerio --wandb_project sweep
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True,
                    help="wandb group name (matches --exp_name from the sweep script)")
parser.add_argument("--wandb_entity", type=str, default="sheerio")
parser.add_argument("--wandb_project", type=str, default="sweep")
parser.add_argument("--out_dir", type=str, default="./plots")
parser.add_argument("--regs", type=str, nargs="+",
                    default=["l2", "spectral", "wass"])
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

METHOD_COLORS = {
    "static":              "#c44e52",  # red
    "adaptive_inv":        "#4c72b0",  # blue
    "adaptive_saturating": "#55a868",  # green
}
METHOD_ORDER = ["static", "adaptive_inv", "adaptive_saturating"]

# ---- pull runs from wandb ----
print(f"Fetching runs from {args.wandb_entity}/{args.wandb_project} group={args.exp_name}...")
api = wandb.Api()
runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}",
                filters={"group": args.exp_name})

# ---- parse run names into a dataframe ----
NAME_RE = re.compile(
    r"reg-(?P<reg>[^_]+)_method-(?P<method>[a-z_]+?)_lr-(?P<lr>[^_]+)_coef-(?P<coef>[^_]+)"
)

records = []
for run in runs:
    m = NAME_RE.match(run.name)
    if m is None:
        print(f"skip (name parse failed): {run.name}")
        continue
    d = m.groupdict()
    d["lr"] = float(d["lr"])
    d["coef"] = float(d["coef"])
    d["run_id"] = run.id
    d["seed"] = run.config.get("random_seed", -1)

    # per-task accuracies for AUC and slope
    hist = run.history(keys=["task_acc"], pandas=True)
    task_accs = hist["task_acc"].dropna().tolist()
    if len(task_accs) < 3:
        d["auc"] = np.nan
        d["slope"] = np.nan
    else:
        d["auc"] = float(np.mean(task_accs))
        x = np.arange(len(task_accs))
        d["slope"] = float(np.polyfit(x, task_accs, 1)[0])
    d["task_accs"] = task_accs
    records.append(d)

df = pd.DataFrame(records)
print(f"Loaded {len(df)} runs.")
if len(df) == 0:
    raise SystemExit("No runs found. Check --exp_name.")

# ---- aggregate across seeds ----
group_cols = ["reg", "method", "lr", "coef"]
agg = df.groupby(group_cols).agg(
    auc_mean=("auc", "mean"),
    auc_std=("auc", "std"),
    slope_mean=("slope", "mean"),
    slope_std=("slope", "std"),
    n=("run_id", "count"),
).reset_index()
print(f"Aggregated to {len(agg)} (reg,method,lr,coef) cells.")

# ---- vanilla baseline (per lr) ----
# We treat the lowest-coef static run as an approximate vanilla anchor.
# If you ran a true vanilla (no reg), that logic would go here.
vanilla_by_lr = (
    agg[agg["method"] == "static"]
    .groupby("lr")["auc_mean"].min()
    .to_dict()
)

# ---- plot: AUC vs coef, one figure per reg, faceted by lr ----
def plot_grid(agg, metric, ylabel, title_prefix, fname_prefix):
    for reg in args.regs:
        sub = agg[agg["reg"] == reg]
        if len(sub) == 0:
            print(f"no runs for reg={reg}, skipping")
            continue

        lrs = sorted(sub["lr"].unique())
        n_lrs = len(lrs)
        fig, axes = plt.subplots(1, n_lrs, figsize=(6 * n_lrs, 4.5),
                                  sharey=False, squeeze=False)

        for ci, lr in enumerate(lrs):
            ax = axes[0, ci]
            for method in METHOD_ORDER:
                cell = sub[(sub["lr"] == lr) & (sub["method"] == method)].sort_values("coef")
                if len(cell) == 0:
                    continue
                mean = cell[f"{metric}_mean"].values
                std = cell[f"{metric}_std"].fillna(0).values
                coefs = cell["coef"].values
                ax.plot(coefs, mean, "-o",
                        color=METHOD_COLORS[method],
                        label=method, linewidth=2)
                ax.fill_between(coefs, mean - std, mean + std,
                                color=METHOD_COLORS[method], alpha=0.2)

            v = vanilla_by_lr.get(lr, None)
            if v is not None:
                ax.axhline(v, color="gray", linestyle="--", alpha=0.6, label="vanilla (approx)")

            ax.set_xscale("log")
            ax.set_xlabel("coefficient")
            ax.set_ylabel(ylabel)
            ax.set_title(f"reg={reg}, lr={lr}")
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.legend(loc="best", fontsize=9)

        fig.suptitle(f"{title_prefix} [{reg}]", fontsize=13)
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"{fname_prefix}_{reg}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

plot_grid(agg, "auc", "AUC (mean task_acc)",
          "AUC vs coefficient (higher = better)", "grid_auc")
plot_grid(agg, "slope", "slope of per-task acc",
          "Slope vs coefficient (flatter/positive = better)", "grid_slope")

# ---- coherence trajectories at peak coef ----
# We pick the peak coef PER (reg, lr, method), then plot mean_off_diag and ratio.
peak_records = []
for (reg, lr, method), grp in agg.groupby(["reg", "lr", "method"]):
    if len(grp) == 0:
        continue
    peak_row = grp.loc[grp["auc_mean"].idxmax()]
    peak_records.append(peak_row)
peak_df = pd.DataFrame(peak_records)

def plot_coherence(peak_df):
    for reg in args.regs:
        sub_peak = peak_df[peak_df["reg"] == reg]
        if len(sub_peak) == 0:
            continue
        lrs = sorted(sub_peak["lr"].unique())
        for lr in lrs:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for method in METHOD_ORDER:
                # find matching run(s) in df, average across seeds
                cell_peak = sub_peak[(sub_peak["lr"] == lr) & (sub_peak["method"] == method)]
                if len(cell_peak) == 0:
                    continue
                coef = float(cell_peak["coef"].values[0])
                matching = df[
                    (df["reg"] == reg) & (df["lr"] == lr) &
                    (df["method"] == method) & (df["coef"] == coef)
                ]
                if len(matching) == 0:
                    continue

                # pull coherence history from wandb for each seed
                mod_series, ratio_series = [], []
                for _, r in matching.iterrows():
                    try:
                        wrun = api.run(f"{args.wandb_entity}/{args.wandb_project}/{r['run_id']}")
                        h = wrun.history(keys=["coherence/mean_off_diag", "coherence/ratio"],
                                         pandas=True)
                        if "coherence/mean_off_diag" in h.columns:
                            mod_series.append(h["coherence/mean_off_diag"].dropna().values)
                        if "coherence/ratio" in h.columns:
                            ratio_series.append(h["coherence/ratio"].dropna().values)
                    except Exception as e:
                        print(f"coherence fetch failed for {r['run_id']}: {e}")

                if len(mod_series) > 0:
                    minlen = min(len(s) for s in mod_series)
                    mod_arr = np.stack([s[:minlen] for s in mod_series], axis=0)
                    ax1.plot(mod_arr.mean(axis=0), color=METHOD_COLORS[method],
                             label=method, linewidth=1.5)
                if len(ratio_series) > 0:
                    minlen = min(len(s) for s in ratio_series)
                    ratio_arr = np.stack([s[:minlen] for s in ratio_series], axis=0)
                    ax2.plot(ratio_arr.mean(axis=0), color=METHOD_COLORS[method],
                             label=method, linewidth=1.5)

            ax1.set_ylabel("mean_off_diag")
            ax1.set_title(f"Coherence trajectory at peak coef, reg={reg}, lr={lr}")
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)
            ax2.set_ylabel("ratio (max/min eig)")
            ax2.set_yscale("log")
            ax2.set_xlabel("log step")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out = os.path.join(args.out_dir, f"coherence_{reg}_lr-{lr}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

plot_coherence(peak_df)

# ---- summary CSV ----
summary_path = os.path.join(args.out_dir, "summary.csv")
agg.to_csv(summary_path, index=False)
print(f"wrote {summary_path}")
