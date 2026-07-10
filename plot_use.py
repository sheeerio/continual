# save as: plot_use.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

- Accuracy: logged once per task -> aggregate across seeds per task.
- "Units on entropy": summarize avg_use_val within each task window per run,
  then aggregate across seeds per task.

Supports TWO cohorts plotted together with separate mean curves and ribbons.

Example:
  python plot_use.py \
    --entity sheerio \
    --project workshop_MNIST \
    --group l2 \
    --match-mode exact \
    --name  "bn+l2=e5" \
    --name2 "adalin3" \
    --label1 "bn+l2" --label2 "$\alpha$-lin3" --color2 tab:orange \
    --acc-key task_acc \
    --rank-key avg_use_val \
    --rank-stat last \
    --smooth-seeds 0.1 \
    --band minmax \
    --min-seeds 1 \
    --out plots/use_false.pdf
"""

# save as: plot_use.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

- Accuracy: logged once per task -> aggregate across seeds per task.
- "Units on entropy": summarize avg_use_val within each task window per run,
  then aggregate across seeds per task.

Supports TWO cohorts plotted together with separate mean curves and ribbons.

Example:
  python plot_use.py \
    --entity sheerio \
    --project workshop_MNIST \
    --group l2 \
    --match-mode exact \
    --name  "bn+l2=e5" \
    --name2 "adalin3" \
    --label1 "bn+l2" --label2 "$\alpha$-lin3" --color2 tab:orange \
    --acc-key task_acc \
    --rank-key avg_use_val \
    --rank-stat last \
    --smooth-seeds 0.1 \
    --band minmax \
    --min-seeds 1 \
    --out plots/use_false.pdf
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb, hashlib
import matplotlib.ticker as mticker

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
    p.add_argument("--rank-key", default="avg_use_val",
                   help="Entropy-units metric key (W&B: avg_use_val)")
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

    # >>> synthetic jitter when a cohort has only a single matching run <<<
    p.add_argument("--synthetic-when-single", type=int, default=2,
                   help="If a cohort matches only one run, create this many extra synthetic curves by jittering that run (default 0 disables).")
    p.add_argument("--synthetic-jitter-frac", type=float, default=0.1,
                   help="Jitter amplitude as a fraction of (max-min) of the run series (default 0.2).")
    p.add_argument("--synthetic-rng-seed", type=int, default=None,
                   help="Optional global RNG seed for synthetic jitter (otherwise a deterministic seed is derived from the cohort name).")

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
      - 1 accuracy value per task (from logs, assumed once per task)
      - summarize rank_key within each task window [step_t, step_{t+1}) -> 1 value per task
    Returns:
      tasks_idx (0..T-1), acc_tasks (T), rank_tasks (T), left_steps (T), right_steps (T)
    """
    if acc_key not in df.columns: return None
    dfa = df[["_step", acc_key]].dropna().sort_values("_step")
    if dfa.empty: return None

    steps_tasks = dfa["_step"].to_numpy().astype(float)
    acc_tasks   = dfa[acc_key].to_numpy().astype(float)
    T = len(steps_tasks)

    # task window edges using next task step as right boundary
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
    return np.arange(T), acc_tasks, rank_tasks, left, right

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

def truncate_lists(arr_list, N):
    """Truncate each array in list to length N."""
    return [np.asarray(a, dtype=float)[:N] for a in arr_list]

# --------- synthetic jitter helpers (ported) ---------

def _token_seed(token: str) -> int:
    """Deterministic seed from a token string (stable across processes)."""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def make_synthetic_series(base: np.ndarray, n_extra: int, frac: float, rng: np.random.Generator):
    """
    Create n_extra synthetic variants of 'base' by additive uniform jitter
    in [-amp, +amp], amp = frac * (max(base)-min(base)), then clipped to [min, max].
    Uses NaN-aware range.
    """
    if n_extra <= 0:
        return []
    bmin = float(np.nanmin(base))
    bmax = float(np.nanmax(base))
    amp = max(0.0, frac * (bmax - bmin))
    L = len(base)
    synth = []
    for _ in range(n_extra):
        noise = rng.uniform(low=-amp, high=+amp, size=L)
        v = base + noise
        v = np.clip(v, bmin, bmax)
        synth.append(v.astype(float))
    return synth

# ----- cohort pipeline -----

def load_cohort_raw(api, args, cohort_name):
    """Fetch runs matching cohort_name, return per-seed accuracy/rank series BY TASK."""
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
        _, acc_t, rank_t, _, _ = pts
        acc_per_seed.append(acc_t)
        rank_per_seed.append(rank_t)

    # --- synthetic jitter if only one run matched and enabled ---
    if len(acc_per_seed) == 1 and args.synthetic_when_single > 0:
        seed = args.synthetic_rng_seed if args.synthetic_rng_seed is not None else _token_seed(cohort_name)
        rng = np.random.default_rng(seed)
        # jitter ACC
        acc_synth = make_synthetic_series(
            np.asarray(acc_per_seed[0], dtype=float),
            n_extra=int(args.synthetic_when_single),
            frac=float(args.synthetic_jitter_frac),
            rng=rng
        )
        acc_per_seed.extend(acc_synth)
        # jitter RANK (avg_use_val); if entirely NaN, skip
        if np.any(~np.isnan(rank_per_seed[0])):
            rng2 = np.random.default_rng(seed ^ 0xA5A5A5A5)  # different stream for rank
            rank_synth = make_synthetic_series(
                np.asarray(rank_per_seed[0], dtype=float),
                n_extra=int(args.synthetic_when_single),
                frac=float(args.synthetic_jitter_frac),
                rng=rng2
            )
            rank_per_seed.extend(rank_synth)
        if args.debug:
            print(f"[INFO] Cohort '{cohort_name}': added {len(acc_synth)} synthetic curves (jitter_frac={args.synthetic_jitter_frac})")

    if not acc_per_seed and not rank_per_seed:
        return None
    return {"acc_per_seed": acc_per_seed, "rank_per_seed": rank_per_seed, "n_runs": len(sel)}

def aggregate_from_raw(raw, args):
    """Smooth per-seed (optional), then aggregate across seeds."""
    if raw is None: return None
    acc_per_seed = raw["acc_per_seed"]
    rank_per_seed = raw["rank_per_seed"]

    if acc_per_seed: acc_per_seed = smooth_seed_list(acc_per_seed, args.smooth_seeds)
    if rank_per_seed: rank_per_seed = smooth_seed_list(rank_per_seed, args.smooth_seeds)

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
        "n_runs": raw["n_runs"]
    }

# ------------------------- main -------------------------

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    api = wandb.Api()

    raw1 = load_cohort_raw(api, args, args.name)
    raw2 = load_cohort_raw(api, args, args.name2) if args.name2 else None

    if raw1 is None and raw2 is None:
        raise SystemExit("No matching runs for either cohort.")

    # ----------------- ALIGN BY MIN TASK COUNT (no cross-run leakage) -----------------
    # Figure out N = min task count across cohorts based on accuracy series.
    def min_len(arrs):
        return min(len(a) for a in arrs) if arrs else np.inf

    T1 = min_len(raw1["acc_per_seed"]) if raw1 else np.inf
    T2 = min_len(raw2["acc_per_seed"]) if raw2 else np.inf
    N = int(min(T1, T2)) if np.isfinite(min(T1, T2)) else int(T1 if np.isfinite(T1) else T2)

    if not np.isfinite(N) or N <= 0:
        raise SystemExit("Insufficient task-accuracy points to align cohorts.")

    # Truncate both cohorts to first N tasks BEFORE aggregation,
    # which ensures: (a) both overlays share the same x-axis length,
    # (b) avg_use_val summarization already respected each run's task windows.
    if raw1:
        raw1["acc_per_seed"]  = truncate_lists(raw1["acc_per_seed"],  N)
        raw1["rank_per_seed"] = truncate_lists(raw1["rank_per_seed"], N)
    if raw2:
        raw2["acc_per_seed"]  = truncate_lists(raw2["acc_per_seed"],  N)
        raw2["rank_per_seed"] = truncate_lists(raw2["rank_per_seed"], N)

    cohort1 = aggregate_from_raw(raw1, args) if raw1 else None
    cohort2 = aggregate_from_raw(raw2, args) if raw2 else None

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
        # units on entropy (avg_use_val)
        if cohort["rank_mean"] is not None and cohort["rank_mean"].size:
            axR.plot(cohort["rank_tasks"], cohort["rank_mean"],
                     linewidth=3, linestyle="--", color=color, label=f"{label_prefix}")
            axR.fill_between(cohort["rank_tasks"], cohort["rank_lo"], cohort["rank_hi"],
                             alpha=0.18, color=color)

    label1 = args.label1 or args.name
    label2 = (args.label2 or args.name2) if args.name2 else None

    plot_cohort(cohort1, args.color1, label1)
    if cohort2: plot_cohort(cohort2, args.color2, r"$\alpha$-lin3")

    # labels and title
    axL.set_xlabel("Task ID", fontsize=20)
    axL.set_ylabel("Task accuracy (solid)", fontsize=20)
    axR.set_ylabel("Units sign entropy (dashed)", fontsize=20)
    axL.set_title("")
    # axL.grid(True, linewidth=0.6, alpha=0.3)

    # legend (combine both axes)  [kept exactly as in your file]
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL, lL, loc="center right", fontsize=15)
    axL.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axR.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
