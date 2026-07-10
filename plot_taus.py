# save as: plot_taus.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

- Accuracy: logged once per task -> aggregate across seeds per task.
- Rank: logged many times per task -> summarize within each task window
        per run (last/mean/median), then aggregate across seeds per task.

Supports TWO cohorts (e.g., baseline vs "_reset") plotted together with
separate mean curves and ribbons.

Example:
  python plot_hessian_false.py \
    --entity sheerio \
    --project random_label_MNIST \
    --group sweeps \
    --match-mode exact \
    --name  "sweeps_l2_lr1e-4_wd1e-3" \
    --name2 "sweeps_wass_lr1e-3_lam1e-3" \
    --label1 l2 --label2 wass --color2 tab:orange \
    --acc-key task_acc \
    --rank-key hessian_rank \
    --rank-stat last \
    --smooth-seeds 0.1 \
    --band std \
    --min-seeds 2 \
    --out plots/seed_agg_wandb.pdf
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb

# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)

    p.add_argument("--name", required=True, help="Primary cohort run name (exact or substring)")
    p.add_argument("--name2", default=None, help="Optional second cohort (e.g., *_reset)")

    p.add_argument("--label1", default=None, help="Legend label for cohort 1 (defaults to --name)")
    p.add_argument("--label2", default=None, help="Legend label for cohort 2 (defaults to --name2)")

    p.add_argument("--color1", default=None, help="Matplotlib color for cohort 1 (optional)")
    p.add_argument("--color2", default="tab:orange", help="Matplotlib color for cohort 2")

    p.add_argument("--match-mode", choices=["exact","contains"], default="contains")

    p.add_argument("--acc-key", default="task_acc")
    p.add_argument("--rank-key", default="hessian_rank")   # or effective_rank
    p.add_argument("--rank-stat", choices=["last","mean","median"], default="last")

    p.add_argument("--downsample-steps", type=int, default=1)

    # smoothing
    p.add_argument("--smooth-seeds", type=float, default=0.9,
                   help="EMA beta applied to EACH seed before aggregation (0 disables).")
    p.add_argument("--smooth-mean", type=float, default=0.0,
                   help="Optional EMA on aggregated mean only (0 disables).")

    # ribbon type
    p.add_argument("--band", choices=["std","iqr","q","minmax"], default="std",
                   help="Ribbon across seeds: std (±1σ), iqr (25–75), q ([q,1-q]), minmax.")
    p.add_argument("--quantile", type=float, default=0.1,
                   help="If --band q, use [q, 1-q] interval (e.g., 0.1).")

    # plotting & misc
    p.add_argument("--min-seeds", type=int, default=2,
                   help="Hide task positions where fewer than this many seeds report.")
    p.add_argument("--show-seeds", action="store_true")
    p.add_argument("--out", default="plots/seed_agg_wandb.pdf")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ------------------------- helpers -------------------------

def match_run(r, group, name, mode):
    if r.group != group: return False
    nm = r.name or r.display_name or ""
    return (nm == name) if mode == "exact" else (name in nm)

def fetch_df(run, keys):
    want = list({"_step", *keys})
    try:
        df = run.history(keys=want, pandas=True)
    except Exception:
        df = None
    if df is None or df.empty:
        rows = []
        try:
            for row in run.scan_history(): rows.append(row)
        except Exception:
            pass
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            keep = [k for k in want if k in df.columns]
            df = df[keep]
    return df if df is not None else pd.DataFrame()

def per_task_series(df, acc_key, rank_key, rank_stat="last",
                    downsample_steps=1, debug=False):
    """
    For ONE run:
      - 1 accuracy value per task (from logs)
      - summarize rank within each task window -> 1 value per task
    """
    if acc_key not in df.columns: return None
    dfa = df[["_step", acc_key]].dropna().sort_values("_step")
    if dfa.empty: return None

    steps_tasks = dfa["_step"].to_numpy().astype(float)
    acc_tasks   = dfa[acc_key].to_numpy().astype(float)
    T = len(steps_tasks)

    max_step = float(np.nanmax(df["_step"])) if "_step" in df.columns else steps_tasks[-1]
    left = steps_tasks
    right = np.concatenate([steps_tasks[1:], np.array([max_step + 1.0])])

    rank_tasks = np.full(T, np.nan)
    if rank_key in df.columns:
        dfr = df[["_step", rank_key]].dropna().sort_values("_step")
        if downsample_steps > 1: dfr = dfr.iloc[::downsample_steps, :]
        rs, rv = dfr["_step"].to_numpy().astype(float), dfr[rank_key].to_numpy().astype(float)

        j = 0
        for t in range(T):
            l, r = left[t], right[t]
            while j < len(rs) and rs[j] < l: j += 1
            k = j
            vals = []
            while k < len(rs) and rs[k] < r:
                vals.append(rv[k]); k += 1
            if vals:
                if rank_stat == "last":
                    rank_tasks[t] = vals[-1]
                elif rank_stat == "mean":
                    rank_tasks[t] = float(np.mean(vals))
                else:
                    rank_tasks[t] = float(np.median(vals))

    if debug: print(f"[per_task] T={T} rank-missing={int(np.isnan(rank_tasks).sum())}")
    return np.arange(T), acc_tasks, rank_tasks

def ema(x, beta):
    if beta is None or beta <= 0.0 or np.isnan(beta): return x.astype(float)
    y = np.empty_like(x, dtype=float)
    m = float(x[0])
    for i, v in enumerate(x):
        m = beta*m + (1.0-beta)*float(v)
        y[i] = m
    return y

def smooth_seed_list(arr_list, beta):
    if beta is None or beta <= 0.0:
        return [np.asarray(a, dtype=float) for a in arr_list]
    return [ema(np.asarray(a, dtype=float), beta) for a in arr_list]

def aggregate_per_step(arr_list, mode="std", q=0.1, min_seeds=2):
    """
    Aggregate ACROSS SEEDS for EACH TASK index t.
    Returns tasks, mean, lo, hi with positions that have < min_seeds dropped.
    """
    if not arr_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    T = min(len(a) for a in arr_list)
    tasks = np.arange(T)
    mean = np.full(T, np.nan, dtype=float)
    lo   = np.full(T, np.nan, dtype=float)
    hi   = np.full(T, np.nan, dtype=float)

    mode = mode.lower()
    for t in range(T):
        vals = np.array([a[t] for a in arr_list if t < len(a) and not np.isnan(a[t])], dtype=float)
        if vals.size < min_seeds: continue
        m = vals.mean()
        if mode == "minmax":
            l, h = vals.min(), vals.max()
        elif mode == "iqr":
            l, h = np.percentile(vals, 25), np.percentile(vals, 75)
        elif mode == "q":
            l, h = np.percentile(vals, 100*q), np.percentile(vals, 100*(1.0-q))
        else:
            sd = vals.std(ddof=0)
            l, h = m - sd, m + sd
        mean[t], lo[t], hi[t] = m, l, h

    keep = ~np.isnan(mean)
    return tasks[keep], mean[keep], lo[keep], hi[keep]

# ----- cohort pipeline -----

def load_and_aggregate_cohort(api, args, cohort_name):
    """Fetch runs matching cohort_name, then return aggregated accuracy/rank per task."""
    runs = api.runs(f"{args.entity}/{args.project}")
    sel = [r for r in runs if match_run(r, args.group, cohort_name, args.match_mode)]
    sel = sorted(sel, key=lambda r: r.created_at)

    if not sel: return None  # allow missing cohort

    acc_per_seed, rank_per_seed = [], []
    for r in sel:
        df = fetch_df(r, keys=[args.acc_key, args.rank_key])
        if df.empty:
            if args.debug: print(f"[{cohort_name}] skip empty {r.id}")
            continue
        pts = per_task_series(df, args.acc_key, args.rank_key,
                              rank_stat=args.rank_stat,
                              downsample_steps=args.downsample_steps,
                              debug=args.debug)
        if pts is None:
            if args.debug: print(f"[{cohort_name}] no '{args.acc_key}' in {r.id}")
            continue
        _, acc_t, rank_t = pts
        acc_per_seed.append(acc_t)
        rank_per_seed.append(rank_t)

    if not acc_per_seed and not rank_per_seed:
        return None

    # per-seed smoothing
    if acc_per_seed: acc_per_seed = smooth_seed_list(acc_per_seed, args.smooth_seeds)
    if rank_per_seed: rank_per_seed = smooth_seed_list(rank_per_seed, args.smooth_seeds)

    # aggregate (across seeds per task)
    acc_tasks = acc_mean = acc_lo = acc_hi = None
    rank_tasks = rank_mean = rank_lo = rank_hi = None

    if acc_per_seed:
        acc_tasks, acc_mean, acc_lo, acc_hi = aggregate_per_step(
            acc_per_seed, mode=args.band, q=args.quantile, min_seeds=args.min_seeds
        )
        if acc_mean.size: acc_mean = ema(acc_mean, args.smooth_mean)

    if rank_per_seed:
        rank_tasks, rank_mean, rank_lo, rank_hi = aggregate_per_step(
            rank_per_seed, mode=args.band, q=args.quantile, min_seeds=args.min_seeds
        )
        if rank_mean.size: rank_mean = ema(rank_mean, args.smooth_mean)

    return {
        "acc_tasks": acc_tasks, "acc_mean": acc_mean, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "rank_tasks": rank_tasks, "rank_mean": rank_mean, "rank_lo": rank_lo, "rank_hi": rank_hi,
        "acc_per_seed": acc_per_seed, "rank_per_seed": rank_per_seed,
        "n_runs": len(sel)
    }

# ------------------------- main -------------------------

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    api = wandb.Api()

    cohort1 = load_and_aggregate_cohort(api, args, args.name)
    cohort2 = load_and_aggregate_cohort(api, args, args.name2) if args.name2 else None

    if cohort1 is None and cohort2 is None:
        raise SystemExit("No matching runs for either cohort.")

    # ------------------------- plot -------------------------
    fig, axL = plt.subplots(figsize=(8, 5))
    axR = axL.twinx()

    def plot_cohort(cohort, color, label_prefix):
        if cohort is None: return
        # optional per-seed overlays
        if args.show_seeds:
            for a in cohort["acc_per_seed"]:
                axL.plot(np.arange(len(a)), a, linestyle="--", alpha=0.10, color=color)
            for r in cohort["rank_per_seed"]:
                axR.plot(np.arange(len(r)), r, linestyle="--", alpha=0.10, color=color)

        # accuracy
        if cohort["acc_mean"] is not None and cohort["acc_mean"].size:
            axL.plot(cohort["acc_tasks"], cohort["acc_mean"],
                     linewidth=3, color=color, label=f"{label_prefix}")
            axL.fill_between(cohort["acc_tasks"], cohort["acc_lo"], cohort["acc_hi"],
                             alpha=0.18, color=color)
        # rank
        if cohort["rank_mean"] is not None and cohort["rank_mean"].size:
            axR.plot(cohort["rank_tasks"], cohort["rank_mean"],
                     linewidth=3, linestyle="--", color=color)
            axR.fill_between(cohort["rank_tasks"], cohort["rank_lo"], cohort["rank_hi"],
                             alpha=0.18, color=color)

    label1 = args.label1 or args.name
    label2 = (args.label2 or args.name2) if args.name2 else None

    plot_cohort(cohort1, args.color1, label1)
    if cohort2: plot_cohort(cohort2, args.color2, label2)

    # labels and title
    axL.set_xlabel("Task id", fontsize=20)
    axL.set_ylabel("Task accuracy (solid)", fontsize=20)
    axR.set_ylabel("Hessian Rank (dashed)", fontsize=20)
    title = f""
    axL.set_title(title)
    # axL.grid(True, linewidth=0.6, alpha=0.3)

    # legend
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="lower left", fontsize=15)

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
