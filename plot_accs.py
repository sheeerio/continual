# save as: plot_param.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

Modes:
  - two_cohorts (default): original behavior with up to 2 cohorts on one figure.
  - multipanel: multiple panels (pages) where each panel overlays N cohorts
                (e.g., 4 learning rates) for task_acc only.

New flags (multipanel):
  --mode multipanel
  --panel "Title|name1,name2,..."  (repeatable)
  --exclude "_reset"               (repeatable; drop any run whose name contains these)
  --first-tasks 20                 (limit to first T tasks)
  --out plots/acc_panels.pdf       (multi-page PDF)  OR  --out plots/panels/ (dir)

Examples:

  # Multi-page PDF with 5 panels (each with 4 LR cohorts), excluding *_reset
  python plot_accs.py \
    --entity sheerio \
    --project camera_ready \
    --group MNIST \
    --mode multipanel \
    --match-mode exact \
    --first-tasks 40 \
    --smooth-seeds 0.1 \
    --band std \
    --acc-key task_acc \
    --min-seeds 2 \
    --panel "Wasserstein|wass_0.0_250_MNIST_ly_sqm10,wass_f_1e-3_250_reset,wass_f_1e-3_250,wass_f_1e-3_ply_sqm10rrl_250" \
    --panel "L2|l2_1e-3_ly_ss10_0.001_250_MNIST,l2_f_1e-3_250,l2_f_1e-3_250_reset,l2_f_1e-3_ply_ss10_250" \
    --panel "CReLU|crelu_1e-2_ly_svar10_0.0_250_MNIST,crelu_f_1e-2_250_reset,crelu_f_1e-2_250,crelu_f_1e-2_ply_svar10ll_250" \
    --out plots/app_ply_v_ly.pdf \
    --exclude _reset

  # Original two-cohort behavior still works:
  python plot_accs.py \
    --entity sheerio \
    --project workshop_MNIST \
    --group l2 \
    --match-mode exact \
    --name  "adalin7" \
    --name2 "adalin3" \
    --label1 "$\\alpha$-lin7" --label2 "$\\alpha$-lin3" --color2 tab:orange \
    --acc-key task_acc \
    --rank-key param_norm \
    --rank-stat last \
    --smooth-seeds 0.1 \
    --band minmax \
    --min-seeds 2 \
    --out plots/param_false.pdf
"""

# save as: plot_param.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

Modes:
  - two_cohorts (default): original behavior with up to 2 cohorts on one figure.
  - multipanel: multiple panels (pages) where each panel overlays N cohorts
                (e.g., 4 learning rates) for task_acc only.

New flags (multipanel):
  --mode multipanel
  --panel "Title|name1,name2,..."  (repeatable)
  --exclude "_reset"               (repeatable; drop any run whose name contains these)
  --first-tasks 20                 (limit to first T tasks)
  --out plots/acc_panels.pdf       (multi-page PDF)  OR  --out plots/panels/ (dir)
"""

import argparse, os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)

    # Modes
    p.add_argument("--mode", choices=["two_cohorts", "multipanel"], default="two_cohorts")

    # Original (two_cohorts) args - explicit required=False
    p.add_argument("--name", required=False, default=None, help="Primary cohort run name (exact or substring)")
    p.add_argument("--name2", required=False, default=None, help="Optional second cohort (e.g., *_reset)")

    p.add_argument("--label1", default=None, help="Legend label for cohort 1 (defaults to --name)")
    p.add_argument("--label2", default=None, help="Legend label for cohort 2 (defaults to --name2)")

    p.add_argument("--color1", default=None, help="Matplotlib color for cohort 1 (optional)")
    p.add_argument("--color2", default="tab:orange", help="Matplotlib color for cohort 2")

    # Multipanel specs (repeatable)
    p.add_argument("--panel", action="append", default=[],
                   help='Define one panel: "Title|name1,name2,...". Repeat for multiple panels.')
    p.add_argument("--panel-colors", default=None,
                   help='Optional comma-separated colors to cycle within each panel.')

    p.add_argument("--exclude", action="append", default=[],
                   help="Substring(s) to exclude from run names (repeatable).")

    p.add_argument("--match-mode", choices=["exact","contains"], default="contains")

    p.add_argument("--acc-key", default="task_acc")
    p.add_argument("--rank-key", default="hessian_rank")   # or effective_rank (two_cohorts mode)
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
    p.add_argument("--first-tasks", type=int, default=None,
                   help="If set, cap to first T tasks (e.g., 20).")

    p.add_argument("--out", default="plots/seed_agg_wandb.pdf",
                   help="For multipanel: if path ends with .pdf => multi-page; if ends with / or is a dir => individual PDFs inside.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ------------------------- helpers -------------------------

def name_contains_any(s, subs):
    s = s or ""
    for t in subs:
        if t and t in s:
            return True
    return False

def match_run(r, group, name, mode, excludes):
    if r.group != group: return False
    nm = r.name or r.display_name or ""
    if name_contains_any(nm, excludes): return False
    return (nm == name) if mode == "exact" else (name in nm)

def fetch_df(run, keys):
    want = list({"_step", *keys})
    try:
        # Attempt to fetch history efficiently
        df = run.history(keys=want, pandas=True)
    except Exception:
        df = None
    
    # Fallback scan if history fail or empty (slow path)
    if df is None or df.empty:
        rows = []
        try:
            for row in run.scan_history(keys=want): 
                rows.append(row)
        except Exception:
            pass
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
    if not df.empty:
        # Filter cols just in case
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

    # --- Step Logic: Normalize to Task Index 0..T ---
    # We return np.arange(T) as the x-axis, ignoring the actual step count.
    
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
    if cohort_name is None:
        return None

    print(f" -> Finding runs for cohort: '{cohort_name}' ...")
    runs = api.runs(f"{args.entity}/{args.project}")
    sel = [r for r in runs if match_run(r, args.group, cohort_name, args.match_mode, args.exclude)]
    sel = sorted(sel, key=lambda r: r.created_at)

    if not sel: 
        print(f"    [Warn] No runs found for '{cohort_name}'")
        return None

    print(f"    Found {len(sel)} runs. Fetching history...")

    acc_per_seed, rank_per_seed = [], []
    for i, r in enumerate(sel):
        # Simple progress indicator
        print(f"    [{i+1}/{len(sel)}] Fetching {r.name}...")
        
        df = fetch_df(r, keys=[args.acc_key, args.rank_key])
        if df.empty:
            if args.debug: print(f"[{cohort_name}] skip empty {r.id}")
            continue
        print(f"      [Debug] DF Shape: {df.shape}. Task_acc missing: {df['task_acc'].isnull().sum()}")
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

def _slice_first_T(tasks, mean, lo, hi, T):
    if tasks is None or mean is None or T is None: return tasks, mean, lo, hi
    mask = tasks < T
    return tasks[mask], mean[mask], lo[mask], hi[mask]

def _title_to_slug(title):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title.strip()).strip("_")
    return slug or "panel"

# ------------------------- plotting -------------------------

def plot_two_cohorts(args, api):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cohort1 = load_and_aggregate_cohort(api, args, args.name)
    cohort2 = load_and_aggregate_cohort(api, args, args.name2) if args.name2 else None

    if cohort1 is None and cohort2 is None:
        raise SystemExit("No matching runs for either cohort.")

    fig, axL = plt.subplots(figsize=(8, 5))
    axR = axL.twinx()

    def plot_cohort(cohort, color, label_prefix):
        if cohort is None: return
        # optional per-seed overlays
        if args.show_seeds and cohort["acc_per_seed"]:
            for a in cohort["acc_per_seed"]:
                axL.plot(np.arange(len(a)), a, linestyle="--", alpha=0.10, color=color)
            if cohort["rank_per_seed"]:
                for r in cohort["rank_per_seed"]:
                    axR.plot(np.arange(len(r)), r, linestyle="--", alpha=0.10, color=color)

        # accuracy
        if cohort["acc_mean"] is not None and cohort["acc_mean"].size:
            acc_tasks, acc_mean, acc_lo, acc_hi = cohort["acc_tasks"], cohort["acc_mean"], cohort["acc_lo"], cohort["acc_hi"]
            
            # Fixed variable name syntax error here
            if args.first_tasks is not None:
                T = args.first_tasks
            else:
                T = None
            if T is not None:
                acc_tasks, acc_mean, acc_lo, acc_hi = _slice_first_T(acc_tasks, acc_mean, acc_lo, acc_hi, T)
            axL.plot(acc_tasks, acc_mean, linewidth=3, color=color, label=f"{label_prefix}")
            axL.fill_between(acc_tasks, acc_lo, acc_hi, alpha=0.18, color=color)

        # rank
        if cohort["rank_mean"] is not None and cohort["rank_mean"].size:
            axR.plot(cohort["rank_tasks"], cohort["rank_mean"],
                     linewidth=3, linestyle="--", color=color)
            axR.fill_between(cohort["rank_tasks"], cohort["rank_lo"], cohort["rank_hi"],
                             alpha=0.18, color=color)

    label1 = args.label1 or (args.name or "cohort1")
    label2 = (args.label2 or args.name2) if args.name2 else None

    plot_cohort(cohort1, args.color1, label1)
    if cohort2: plot_cohort(cohort2, args.color2, label2)

    # labels and title
    axL.set_xlabel("Task id", fontsize=20)
    axL.set_ylabel("Task accuracy (solid)", fontsize=20)
    axR.set_ylabel("Param rank (dashed)", fontsize=20)
    axL.set_title("")
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper left", fontsize=15)

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

def _parse_panels(panel_args):
    """
    Each entry: "Title|name1,name2,name3[,name4...]"
    Returns list of dicts: {"title": str, "names": [str, ...]}
    """
    panels = []
    for spec in panel_args:
        if "|" not in spec:
            raise SystemExit(f'--panel must be "Title|name1,name2,...", got: {spec}')
        title, names_str = spec.split("|", 1)
        names = [s.strip() for s in names_str.split(",") if s.strip()]
        if not names:
            raise SystemExit(f'--panel "{spec}" has no names')
        panels.append({"title": title.strip(), "names": names})
    return panels

def _color_cycle_from_arg(arg):
    if not arg: return None
    cols = [c.strip() for c in arg.split(",") if c.strip()]
    return cols if cols else None

def _ensure_out_target(path):
    # If endswith .pdf -> treat as multi-page file
    # Else treat as directory (create it)
    if path.lower().endswith(".pdf"):
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        return ("pdf", path)
    # directory mode
    if not path.endswith("/"): path = path + "/"
    os.makedirs(path, exist_ok=True)
    return ("dir", path)

def plot_multipanel(args, api):
    panels = _parse_panels(args.panel)
    if not panels:
        raise SystemExit("multipanel mode requires at least one --panel")

    target_kind, target_path = _ensure_out_target(args.out)
    color_cycle = _color_cycle_from_arg(args.panel_colors)

    # Preload all cohorts per panel/name
    cache = {}  # (name) -> cohort dict
    def get_cohort(name):
        if name not in cache:
            cache[name] = load_and_aggregate_cohort(api, args, name)
        return cache[name]

    def plot_one_panel(panel):
        print(f"Rendering panel: {panel['title']}")
        fig, ax = plt.subplots(figsize=(8, 5))
        used_colors = []
        for i, nm in enumerate(panel["names"]):
            cohort = get_cohort(nm)
            if cohort is None or cohort["acc_mean"] is None or not cohort["acc_mean"].size:
                print(f"  [WARN] No data for '{nm}' (panel '{panel['title']}')")
                continue
            # Color pick
            if color_cycle:
                color = color_cycle[i % len(color_cycle)]
            else:
                # let mpl cycle; store to fill_between with same
                color = None
            # First-T slicing
            tasks, mean, lo, hi = cohort["acc_tasks"], cohort["acc_mean"], cohort["acc_lo"], cohort["acc_hi"]
            if args.first_tasks is not None:
                tasks, mean, lo, hi = _slice_first_T(tasks, mean, lo, hi, args.first_tasks)

            line, = ax.plot(tasks, mean, linewidth=3, color=color, label=nm)
            # figure out actual color used if color=None (mpl chooses from cycle)
            used_color = line.get_color()
            used_colors.append(used_color)
            ax.fill_between(tasks, lo, hi, alpha=0.18, color=used_color)

            if args.show_seeds and cohort["acc_per_seed"]:
                for a in cohort["acc_per_seed"]:
                    aa = np.asarray(a, dtype=float)
                    if args.first_tasks is not None:
                        aa = aa[:args.first_tasks]
                    ax.plot(np.arange(len(aa)), aa, linestyle="--", alpha=0.10, color=used_color)

        ax.set_xlabel("Task id", fontsize=16)
        ax.set_ylabel("Task accuracy", fontsize=16)
        ttl = f"{panel['title']}"
        if args.first_tasks is not None:
            ttl += f"  (first {args.first_tasks} tasks)"
        ax.set_title(ttl)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, linewidth=0.6, alpha=0.25)

        fig.tight_layout()
        return fig

    if target_kind == "pdf":
        with PdfPages(target_path) as pdf:
            for panel in panels:
                fig = plot_one_panel(panel)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"wrote {target_path} (multi-page)")
    else:
        out_dir = target_path
        for panel in panels:
            fig = plot_one_panel(panel)
            fname = os.path.join(out_dir, f"{_title_to_slug(panel['title'])}.pdf")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fname}")

# ------------------------- main -------------------------

def main():
    args = parse_args()

    # small fix: ensure parent directory exists for dir/pdf modes
    if args.mode == "two_cohorts":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    api = wandb.Api()

    if args.mode == "two_cohorts":
        if not args.name and not args.name2:
            raise SystemExit("two_cohorts mode requires --name (and optionally --name2)")
        plot_two_cohorts(args, api)
    else:
        plot_multipanel(args, api)

if __name__ == "__main__":
    main()