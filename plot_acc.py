# save as: plot_acc.py
"""
Seed aggregation BY TASK index. No cross-step leakage.

Modes:
  - two_cohorts (default): original behavior with up to 2 cohorts on one figure.
  - multipanel: multiple panels (pages) where each panel overlays N cohorts
                (e.g., 4 learning rates) for task_acc + overlay metric.

New flags:
  --overlay-native-steps           (plot overlay at native step resolution, mapped into fractional task x)
  --overlay-bins-per-task 80       (bins per task for native-step overlay aggregation)
  --max-samples 10000000           (ask W&B for many rows to avoid server-side downsampling)
  --scan-fallback-limit-rows 0     (cap when falling back to scan_history; 0 = no cap)

Examples:

    python plot_acc.py \
    --entity sheerio \
    --project random_label_MNIST \
    --group sweeps \
    --mode multipanel \
    --match-mode exact \
    --first-tasks 20 \
    --smooth-seeds 0.9 \
    --band iqr \
    --min-seeds 1 \
    --overlay-key-default lam_std \
    --overlay-native-steps \
    --overlay-bins-per-task 10 \
    --panel "L2 lr=1e-5|sweeps_l2_lr1e-5_wd1e-3|colors:tab:red" \
    --panel "L2 lr=1e-4|sweeps_l2_lr1e-4_wd1e-3|colors:tab:orange" \
    --panel "L2 lr=1e-3|sweeps_l2_lr1e-3_wd1e-3|colors:tab:purple" \
    --label-map "sweeps_l2_lr1e-5_wd1e-3=L2 lr=1e-5" \
    --label-map "sweeps_l2_lr1e-4_wd1e-3=L2 lr=1e-4" \
    --label-map "sweeps_l2_lr1e-3_wd1e-3=L2 lr=1e-3" \
    --ncols 3 --nrows 1 \
    --panel-size 5x4 \
    --max-samples 10000000 \
    --out plots/fig2_l2_lr_panels.pdf

  # Original two-cohort behavior still works:
  python plot_param.py \
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

import argparse, os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import math

# ------------------------- args -------------------------

LOG_OVERLAY_KEYS = {"sharpness_var", "ema_variation_inv", "lam_cstd", "lam_std"}
HISTORY_SAMPLES = 1_000_000
SCAN_FALLBACK_LIMIT_ROWS = 1000


def _maybe_set_log_y(ax, overlay_key: str):
    if overlay_key in LOG_OVERLAY_KEYS:
        ax.set_yscale("log")

def _pair_whx(s, default=(6.0, 5.0)):
    try:
        w, h = s.lower().replace('x', ' ').split()
        return float(w), float(h)
    except Exception:
        return default

def _parse_kv_list(pairs, sep='='):
    out = {}
    for item in pairs or []:
        if sep not in item:
            raise SystemExit(f"--label-map expects 'old{sep}new', got: {item}")
        k, v = item.split(sep, 1)
        out[k.strip()] = v.strip()
    return out

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)

    # Modes
    p.add_argument("--mode", choices=["two_cohorts", "multipanel"], default="two_cohorts")

    # Original (two_cohorts) args
    p.add_argument("--name", default=None, help="Primary cohort run name (exact or substring)")
    p.add_argument("--name2", default=None, help="Optional second cohort (e.g., *_reset)")

    p.add_argument("--label1", default=None, help="Legend label for cohort 1 (defaults to --name)")
    p.add_argument("--label2", default=None, help="Legend label for cohort 2 (defaults to --name2)")

    p.add_argument("--color1", default=None, help="Matplotlib color for cohort 1 (optional)")
    p.add_argument("--color2", default="tab:orange", help="Matplotlib color for cohort 2")

    # High-fidelity history controls
    p.add_argument("--max-samples", type=int, default=10_000_000,
                   help="Rows to request from W&B history() per run to avoid server-side downsampling.")
    p.add_argument("--scan-fallback-limit-rows", type=int, default=0,
                   help="If >0, cap rows when falling back to scan_history().")

    # Multipanel specs (repeatable)
    p.add_argument("--panel", action="append", default=[],
                help='Define one panel: "Title|name1,name2,..."  (optional third part: "|colors:c1,c2,..." )')
    p.add_argument("--panel-colors", default=None,
                help='Optional comma-separated colors to cycle within each panel (global default).')
    p.add_argument("--panel-colors-map", action="append", default=[],
                help='Per-panel colors: "Title:c1,c2,c3" (repeatable).')

    p.add_argument("--exclude", action="append", default=[],
                help="Substring(s) to exclude from run names (repeatable).")

    p.add_argument("--match-mode", choices=["exact","contains"], default="contains")

    p.add_argument("--acc-key", default="task_acc")
    p.add_argument("--rank-key", default="hessian_rank",   # or effective_rank (two_cohorts mode)
                help="Used in two_cohorts mode for the dashed line on the right axis.")
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

    # Layout & style
    p.add_argument("--panel-size", default="6x5",
                help="Width x Height inches per panel (e.g., 6x5 ~ almost-square).")
    p.add_argument("--ncols", type=int, default=1,
                help="For multipanel+PDF: number of columns per page (grid). If >1 => side-by-side.")
    p.add_argument("--nrows", type=int, default=1,
                help="For multipanel+PDF: number of rows per page (grid).")

    # Overlay (multipanel) selection
    p.add_argument("--overlay-key-default", default="sharpness_var",
                help="Dashed overlay metric for non-ReLU panels (title does NOT contain 'relu').")
    p.add_argument("--overlay-key-relu", default="ema_variation_inv",
                help="Dashed overlay metric for panels whose title contains 'relu' (case-insensitive).")

    # Native-step overlay options
    p.add_argument("--overlay-native-steps", action="store_true",
                help="Plot overlay at native step resolution (binned into fractional task x).")
    p.add_argument("--overlay-bins-per-task", type=int, default=50,
                help="Bins per task used to aggregate native-step overlay across seeds.")

    # Label remapping (multipanel)
    p.add_argument("--label-map", action="append", default=[],
                help="Repeatable: 'exact_run_name=Pretty Label' to rename legend entries.")

    p.add_argument("--out", default="plots/seed_agg_wandb.pdf",
                help="For multipanel: if path ends with .pdf => multi-page or grid per page; if ends with / or is a dir => individual PDFs inside.")
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

def _fetch_series(run, key):
    """Fetch one metric at high fidelity; fall back to unfiltered scan."""
    if not key:
        return pd.DataFrame()
    # Try high-fidelity history for just this key
    try:
        df = run.history(keys=[key, "_step"], pandas=True, samples=HISTORY_SAMPLES)
    except Exception:
        df = None
    # Fall back: unfiltered scan (don't pass keys= to avoid filtering sparse rows)
    if df is None or df.empty or key not in df.columns or "_step" not in df.columns:
        rows, n = [], 0
        try:
            for row in run.scan_history():
                rows.append(row); n += 1
                if SCAN_FALLBACK_LIMIT_ROWS > 0 and n >= SCAN_FALLBACK_LIMIT_ROWS:
                    break
        except Exception:
            pass
        df = pd.DataFrame(rows)
    if df is None or df.empty or key not in df.columns or "_step" not in df.columns:
        return pd.DataFrame(columns=["_step", key])
    return df[["_step", key]].dropna(subset=["_step"]).sort_values("_step")

def fetch_df(run, keys):
    """
    Fetch accuracy and overlay independently, then outer-merge on _step.
    This avoids the 'empty because one key is missing' failure mode.
    """
    keys = [k for k in keys if k]
    acc_key = keys[0] if keys else None
    aux_key = keys[1] if len(keys) >= 2 else None
    acc_df = _fetch_series(run, acc_key)
    if acc_df.empty:
        return pd.DataFrame()  # no accuracy = truly no data
    if aux_key:
        aux_df = _fetch_series(run, aux_key)
        if not aux_df.empty:
            df = pd.merge(acc_df, aux_df, on="_step", how="outer", sort=True)
        else:
            df = acc_df
    else:
        df = acc_df
    return df

def per_task_series(df, acc_key, aux_key, aux_stat="last",
                    downsample_steps=1, debug=False):
    """
    For ONE run:
      - 1 accuracy value per task (from logs)
      - summarize aux_key within each task window -> 1 value per task
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

    aux_tasks = np.full(T, np.nan)
    if aux_key and aux_key in df.columns:
        dfr = df[["_step", aux_key]].dropna().sort_values("_step")
        if downsample_steps > 1: dfr = dfr.iloc[::downsample_steps, :]
        rs, rv = dfr["_step"].to_numpy().astype(float), dfr[aux_key].to_numpy().astype(float)

        j = 0
        for t in range(T):
            l, r = left[t], right[t]
            while j < len(rs) and rs[j] < l: j += 1
            k = j
            vals = []
            while k < len(rs) and rs[k] < r:
                vals.append(rv[k]); k += 1
            if vals:
                if aux_stat == "last":
                    aux_tasks[t] = vals[-1]
                elif aux_stat == "mean":
                    aux_tasks[t] = float(np.mean(vals))
                else:
                    aux_tasks[t] = float(np.median(vals))

    if debug: print(f"[per_task] T={T} aux-missing={int(np.isnan(aux_tasks).sum())}")
    return np.arange(T), acc_tasks, aux_tasks

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

# --- native-step overlay helpers ---

def _task_windows(df, acc_key):
    """Return task boundary arrays: left[t], right[t] in step space, for mapping steps→fractional task x."""
    if acc_key not in df.columns: return None
    dfa = df[["_step", acc_key]].dropna().sort_values("_step")
    if dfa.empty: return None
    steps_tasks = dfa["_step"].to_numpy().astype(float)
    T = len(steps_tasks)
    max_step = float(np.nanmax(df["_step"])) if "_step" in df.columns else steps_tasks[-1]
    left = steps_tasks
    right = np.concatenate([steps_tasks[1:], np.array([max_step + 1.0])])
    return left, right, T

def _map_steps_to_fractional_task(rs, left, right, T_clip=None):
    """Map raw step array rs to fractional task x in [0, T)."""
    if rs is None or len(rs)==0: return np.array([]), np.array([])
    left = np.asarray(left); right = np.asarray(right)
    T = len(left)
    # task index via right-open windows
    t_idx = np.searchsorted(left, rs, side="right") - 1
    t_idx = np.clip(t_idx, 0, T-1)
    # keep only points that still lie in their window
    ok = rs < right[t_idx]
    rs = rs[ok]; t_idx = t_idx[ok]
    denom = np.maximum(right[t_idx] - left[t_idx], 1e-9)
    frac = (rs - left[t_idx]) / denom
    x = t_idx.astype(float) + frac
    if T_clip is not None:
        keep = x < float(T_clip)
        x = x[keep]; rs = rs[keep]
    return x, rs

def _bin_overlay_across_seeds(native_xy_list, T, bins_per_task, mode="std", q=0.1, min_seeds=2):
    """
    native_xy_list: list of (x_frac, y) arrays for each seed; x in [0,T).
    Bin each seed's y by x, then aggregate across seeds per bin.
    """
    if not native_xy_list:  # nothing logged
        return np.array([]), np.array([]), np.array([]), np.array([])
    total_bins = max(1, int(T) * max(1, int(bins_per_task)))
    edges = np.linspace(0.0, float(T), total_bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    per_seed_binned = []
    for x, y in native_xy_list:
        if x.size == 0:
            per_seed_binned.append(np.full(total_bins, np.nan)); continue
        idx = np.digitize(x, edges) - 1
        idx = np.clip(idx, 0, total_bins-1)
        # mean per bin for this seed
        accum = [[] for _ in range(total_bins)]
        for k, v in zip(idx, y):
            if not (np.isnan(v) or np.isinf(v)): accum[k].append(float(v))
        arr = np.full(total_bins, np.nan)
        for k in range(total_bins):
            if accum[k]:
                arr[k] = float(np.mean(accum[k]))
        per_seed_binned.append(arr)
    # aggregate across seeds per bin
    mean = np.full(total_bins, np.nan); lo = np.full(total_bins, np.nan); hi = np.full(total_bins, np.nan)
    mode = (mode or "std").lower()
    for b in range(total_bins):
        vals = np.array([s[b] for s in per_seed_binned if b < len(s) and not np.isnan(s[b])], dtype=float)
        if vals.size < min_seeds: continue
        m = vals.mean()
        if mode == "minmax":
            l, h = vals.min(), vals.max()
        elif mode == "iqr":
            l, h = np.percentile(vals, 25), np.percentile(vals, 75)
        elif mode == "q":
            l, h = np.percentile(vals, 100*q), np.percentile(vals, 100*(1.0-q))
        else:
            sd = vals.std(ddof=0); l, h = m - sd, m + sd
        mean[b], lo[b], hi[b] = m, l, h
    keep = ~np.isnan(mean)
    return centers[keep], mean[keep], lo[keep], hi[keep]

# ----- cohort pipeline -----

def load_and_aggregate_cohort(api, args, cohort_name, aux_key_override=None):
    """Fetch runs matching cohort_name, then return aggregated accuracy + aux per task.
    aux_key_override lets multipanel ask for a different dashed metric (e.g., sharpness_var)."""
    if cohort_name is None:
        return None

    runs = api.runs(f"{args.entity}/{args.project}")
    sel = [r for r in runs if match_run(r, args.group, cohort_name, args.match_mode, args.exclude)]
    sel = sorted(sel, key=lambda r: r.created_at)

    if not sel: return None  # allow missing cohort

    acc_per_seed, aux_per_seed = [], []
    native_xy_list = []  # for overlay-native-steps
    min_T_tasks = None
    overlay_key = (aux_key_override or args.rank_key)

    for r in sel:
        df = fetch_df(r, keys=[args.acc_key, overlay_key])
        if df.empty:
            if args.debug: print(f"[{cohort_name}] skip empty {r.id}")
            continue
        pts = per_task_series(df, args.acc_key, overlay_key,
                              aux_stat=args.rank_stat,
                              downsample_steps=args.downsample_steps,
                              debug=args.debug)
        if pts is None:
            if args.debug: print(f"[{cohort_name}] no '{args.acc_key}' in {r.id}")
            continue
        _, acc_t, aux_t = pts
        acc_per_seed.append(acc_t)
        aux_per_seed.append(aux_t)

        # prepare native-step overlay if requested
        if args.overlay_native_steps and (overlay_key in df.columns):
            tw = _task_windows(df, args.acc_key)
            if tw is not None:
                left, right, T = tw
                if min_T_tasks is None: min_T_tasks = T
                else: min_T_tasks = min(min_T_tasks, T)
                dfr = df[["_step", overlay_key]].dropna().sort_values("_step")
                if args.downsample_steps > 1:
                    dfr = dfr.iloc[::args.downsample_steps, :]
                rs = dfr["_step"].to_numpy().astype(float)
                rv = dfr[overlay_key].to_numpy().astype(float)
                x_frac, _ = _map_steps_to_fractional_task(rs, left, right, T_clip=None)
                if x_frac.size and rv.size:
                    n = min(len(x_frac), len(rv))
                    native_xy_list.append((x_frac[:n], rv[:n]))

    if not acc_per_seed and not aux_per_seed:
        return None

    # per-seed smoothing
    if acc_per_seed: acc_per_seed = smooth_seed_list(acc_per_seed, 0.3)
    if aux_per_seed: aux_per_seed = smooth_seed_list(aux_per_seed, args.smooth_seeds)

    # aggregate (across seeds per task)
    acc_tasks = acc_mean = acc_lo = acc_hi = None
    aux_tasks = aux_mean = aux_lo = aux_hi = None
    overlay_bins_x = overlay_mean = overlay_lo = overlay_hi = None

    if acc_per_seed:
        acc_tasks, acc_mean, acc_lo, acc_hi = aggregate_per_step(
            acc_per_seed, mode=args.band, q=args.quantile, min_seeds=args.min_seeds
        )
        if acc_mean.size: acc_mean = ema(acc_mean, args.smooth_mean)

    if aux_per_seed:
        aux_tasks, aux_mean, aux_lo, aux_hi = aggregate_per_step(
            aux_per_seed, mode=args.band, q=args.quantile, min_seeds=args.min_seeds
        )
        if aux_mean.size: aux_mean = ema(aux_mean, args.smooth_mean)

    # Native-step overlay aggregation (mapped & binned)
    if args.overlay_native_steps and native_xy_list and (min_T_tasks is not None):
        overlay_bins_x, overlay_mean, overlay_lo, overlay_hi = _bin_overlay_across_seeds(
            native_xy_list, T=min_T_tasks, bins_per_task=args.overlay_bins_per_task,
            mode=args.band, q=args.quantile, min_seeds=args.min_seeds
        )

    return {
        "acc_tasks": acc_tasks, "acc_mean": acc_mean, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "aux_tasks": aux_tasks, "aux_mean": aux_mean, "aux_lo": aux_lo, "aux_hi": aux_hi,
        "acc_per_seed": acc_per_seed, "aux_per_seed": aux_per_seed,
        "overlay_bins_x": overlay_bins_x, "overlay_mean": overlay_mean,
        "overlay_lo": overlay_lo, "overlay_hi": overlay_hi,
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

    cohort1 = load_and_aggregate_cohort(api, args, args.name, aux_key_override=args.rank_key)
    cohort2 = load_and_aggregate_cohort(api, args, args.name2, aux_key_override=args.rank_key) if args.name2 else None

    if cohort1 is None and cohort2 is None:
        raise SystemExit("No matching runs for either cohort.")

    panel_w, panel_h = _pair_whx(args.panel_size, default=(6.0,5.0))
    fig, axL = plt.subplots(figsize=(panel_w, panel_h))
    axR = axL.twinx()

    def plot_cohort(cohort, color, label_prefix):
        if cohort is None: return
        # optional per-seed overlays
        if args.show_seeds and cohort["acc_per_seed"]:
            for a in cohort["acc_per_seed"]:
                axL.plot(np.arange(len(a)), a, linestyle="-", alpha=0.10, color=color)
            if cohort["aux_per_seed"]:
                for r in cohort["aux_per_seed"]:
                    axR.plot(np.arange(len(r)), r, linestyle="-", alpha=0.10, color=color)

        # accuracy
        if cohort["acc_mean"] is not None and cohort["acc_mean"].size:
            acc_tasks, acc_mean, acc_lo, acc_hi = cohort["acc_tasks"], cohort["acc_mean"], cohort["acc_lo"], cohort["acc_hi"]
            T = args.first_tasks if args.first_tasks is not None else None
            if T is not None:
                acc_tasks, acc_mean, acc_lo, acc_hi = _slice_first_T(acc_tasks, acc_mean, acc_lo, acc_hi, T)
            axL.plot(acc_tasks, acc_mean, linewidth=1.2, linestyle="--", color=color, label=f"{label_prefix}")
            axL.fill_between(acc_tasks, acc_lo, acc_hi, alpha=0.18, color=color)

        # overlay
        if args.overlay_native_steps and cohort.get("overlay_bins_x") is not None and len(cohort["overlay_bins_x"]) > 0:
            ox, om, olo, ohi = cohort["overlay_bins_x"], cohort["overlay_mean"], cohort["overlay_lo"], cohort["overlay_hi"]
            if args.first_tasks is not None:
                msk = ox < float(args.first_tasks)
                ox, om, olo, ohi = ox[msk], om[msk], olo[msk], ohi[msk]
            axR.plot(ox, om, linewidth=2.2, linestyle="-", color=color)
            axR.fill_between(ox, olo, ohi, alpha=0.10, color=color)
        elif cohort["aux_mean"] is not None and cohort["aux_mean"].size:
            aux_tasks, aux_mean, aux_lo, aux_hi = cohort["aux_tasks"], cohort["aux_mean"], cohort["aux_lo"], cohort["aux_hi"]
            if args.first_tasks is not None:
                aux_tasks, aux_mean, aux_lo, aux_hi = _slice_first_T(aux_tasks, aux_mean, aux_lo, aux_hi, args.first_tasks)
            axR.plot(aux_tasks, aux_mean, linewidth=1.2, linestyle="-", color=color)
            axR.fill_between(aux_tasks, aux_lo, aux_hi, alpha=0.10, color=color)

    _maybe_set_log_y(axR, args.rank_key)

    label1 = args.label1 or (args.name or "cohort1")
    label2 = (args.label2 or args.name2) if args.name2 else None

    plot_cohort(cohort1, args.color1, label1)
    if cohort2: plot_cohort(cohort2, args.color2, label2)

    # labels and title
    axL.set_xlabel("Task id", fontsize=20)
    axL.set_ylabel("Task accuracy (dashed)", fontsize=20)
    axR.set_ylabel("Aux metric (dashed)", fontsize=20)
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
    Optional third part for colors: "|colors:c1,c2,c3"
    Returns list of dicts: {"title": str, "names": [str, ...], "colors": [str]|None}
    """
    panels = []
    for spec in panel_args:
        parts = spec.split("|")
        if len(parts) < 2:
            raise SystemExit(f'--panel must be "Title|name1,name2,...", got: {spec}')
        title = parts[0].strip()
        names_str = parts[1]
        names = [s.strip() for s in names_str.split(",") if s.strip()]
        if not names:
            raise SystemExit(f'--panel "{spec}" has no names')
        colors = None
        if len(parts) >= 3 and parts[2].strip():
            p2 = parts[2].strip()
            if p2.lower().startswith("colors:") or p2.lower().startswith("c:"):
                colors = [c.strip() for c in p2.split(":",1)[1].split(",") if c.strip()]
        panels.append({"title": title, "names": names, "colors": colors})
    return panels

def _color_cycle_from_arg(arg):
    if not arg: return None
    cols = [c.strip() for c in arg.split(",") if c.strip()]
    return cols if cols else None

def _ensure_out_target(path):
    # If endswith .pdf -> treat as multi-page (or grid) file
    # Else treat as directory (create it)
    if path.lower().endswith(".pdf"):
        parent = os.path.dirname(path)
        if parent: os.makedirs(parent, exist_ok=True)
        return ("pdf", path)
    # directory mode
    if not path.endswith("/"):
        path = path + "/"
    os.makedirs(path, exist_ok=True)
    return ("dir", path)

def _parse_panel_colors_map(items):
    # "Title:c1,c2,c3"
    out = {}
    for it in items or []:
        if ":" not in it:
            raise SystemExit(f'--panel-colors-map expects "Title:c1,c2,...", got: {it}')
        title, cols = it.split(":", 1)
        out[title.strip()] = [c.strip() for c in cols.split(",") if c.strip()]
    return out

def plot_multipanel(args, api):
    panels = _parse_panels(args.panel)
    if not panels:
        raise SystemExit("multipanel mode requires at least one --panel")

    target_kind, target_path = _ensure_out_target(args.out)
    global_color_cycle = _color_cycle_from_arg(args.panel_colors)
    panel_colors_map = _parse_panel_colors_map(args.panel_colors_map)
    label_map = _parse_kv_list(args.label_map, sep="=")
    panel_w, panel_h = _pair_whx(args.panel_size, default=(6.0,5.0))

    # Preload all cohorts per (panel-dependent) aux key
    cache = {}  # (name, aux_key) -> cohort dict
    def get_cohort(name, aux_key):
        key = (name, aux_key)
        if key not in cache:
            cache[key] = load_and_aggregate_cohort(api, args, name, aux_key_override=aux_key)
        return cache[key]

    def panel_overlay_key(title):
        return (args.overlay_key_relu
                if ("relu" in (title or "").lower())
                else args.overlay_key_default)

    def pick_panel_colors(panel):
        # per-panel in spec > --panel-colors-map > global cycle > mpl default cycle
        if panel.get("colors"):
            return panel["colors"]
        if panel["title"] in panel_colors_map:
            return panel_colors_map[panel["title"]]
        return global_color_cycle

    def plot_one_panel(ax, panel):
        # Base & dashed overlay on a twin axis
        overlay_key = panel_overlay_key(panel["title"])
        color_cycle = pick_panel_colors(panel)

        used_colors = []
        ax2 = ax.twinx()

        _maybe_set_log_y(ax2, overlay_key)

        for i, nm in enumerate(panel["names"]):
            cohort = get_cohort(nm, overlay_key)
            if cohort is None or cohort["acc_mean"] is None or not cohort["acc_mean"].size:
                print(f"[WARN] No data for '{nm}' (panel '{panel['title']}')")
                continue
            # Color pick
            color = (color_cycle[i % len(color_cycle)]) if color_cycle else None

            # First-T slicing
            tasks, mean, lo, hi = cohort["acc_tasks"], cohort["acc_mean"], cohort["acc_lo"], cohort["acc_hi"]
            if args.first_tasks is not None:
                tasks, mean, lo, hi = _slice_first_T(tasks, mean, lo, hi, args.first_tasks)

            label = label_map.get(nm, nm)

            line, = ax.plot(tasks, mean, linewidth=2, linestyle="--", color=color, label=label)
            used_color = line.get_color()
            used_colors.append(used_color)
            ax.fill_between(tasks, lo, hi, alpha=0.18, color=used_color)

            # dashed overlay (native-step if requested)
            if args.overlay_native_steps and cohort.get("overlay_bins_x") is not None and len(cohort["overlay_bins_x"])>0:
                ox, om, olo, ohi = cohort["overlay_bins_x"], cohort["overlay_mean"], cohort["overlay_lo"], cohort["overlay_hi"]
                if args.first_tasks is not None:
                    msk = ox < float(args.first_tasks)
                    ox, om, olo, ohi = ox[msk], om[msk], olo[msk], ohi[msk]
                ax2.plot(ox, om, linewidth=1.2, linestyle="-", color=used_color, label=None)
                ax2.fill_between(ox, olo, ohi, alpha=0.10, color=used_color)
            elif cohort["aux_mean"] is not None and cohort["aux_mean"].size:
                ot, om, olo, ohi = cohort["aux_tasks"], cohort["aux_mean"], cohort["aux_lo"], cohort["aux_hi"]
                if args.first_tasks is not None:
                    ot, om, olo, ohi = _slice_first_T(ot, om, olo, ohi, args.first_tasks)
                ax2.plot(ot, om, linewidth=1.2, linestyle="-", color=used_color, label=None)
                ax2.fill_between(ot, olo, ohi, alpha=0.10, color=used_color)

            if args.show_seeds and cohort["acc_per_seed"]:
                for a in cohort["acc_per_seed"]:
                    aa = np.asarray(a, dtype=float)
                    if args.first_tasks is not None:
                        aa = aa[:args.first_tasks]
                    ax.plot(np.arange(len(aa)), aa, linestyle="-", alpha=0.10, color=used_color)

        ax.set_xlabel("Task id", fontsize=12)
        ax.set_ylabel("Task accuracy (dashed)", fontsize=12)
        ax2.set_ylabel(r"$\mathrm{Vol}_\lambda$ (solid)", fontsize=12)
        ttl = f"{panel['title']}"
        if args.first_tasks is not None:
            ttl += f"  (first {args.first_tasks} tasks)"
        ax.set_title(ttl, fontsize=12)
        # Legend: one entry per cohort + proxies for line styles
        handles, labels = ax.get_legend_handles_labels()
        # style proxies
        proxy_acc = Line2D([0],[0], linestyle='-', linewidth=1.2, color='black')
        proxy_aux = Line2D([0],[0], linestyle='--', linewidth=1.2, color='black')
        handles = [proxy_acc, proxy_aux] + handles
        # ax.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(True, linewidth=0.6, alpha=0.25)
        return ax, ax2

    if target_kind == "pdf":
        # If ncols * nrows == 1 -> original: one panel per page.
        per_page = max(1, int(args.ncols) * max(1, int(args.nrows)))
        with PdfPages(target_path) as pdf:
            if per_page == 1:
                for panel in panels:
                    fig, ax = plt.subplots(figsize=(panel_w, panel_h))
                    plot_one_panel(ax, panel)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            else:
                # Grid pages
                for start in range(0, len(panels), per_page):
                    chunk = panels[start:start+per_page]
                    n = len(chunk)
                    rows = args.nrows
                    cols = args.ncols
                    # Compute figure size from per-panel size:
                    fig_w = panel_w * cols
                    fig_h = panel_h * rows
                    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
                    for idx, panel in enumerate(chunk):
                        r = idx // cols
                        c = idx % cols
                        ax = axes[r][c]
                        plot_one_panel(ax, panel)
                    # Turn off any unused axes
                    for idx in range(n, rows*cols):
                        r = idx // cols
                        c = idx % cols
                        axes[r][c].axis('off')
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        print(f"wrote {target_path} (pdf; grid {args.nrows}x{args.ncols})")
    else:
        # Save each panel as its own (almost-square) PDF in the directory
        out_dir = target_path
        for panel in panels:
            fig, ax = plt.subplots(figsize=(panel_w, panel_h))
            plot_one_panel(ax, panel)
            fig.tight_layout()
            fname = os.path.join(out_dir, f"{_title_to_slug(panel['title'])}.pdf")
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fname}")

# ------------------------- main -------------------------

def main():
    args = parse_args()

    # small fix: ensure parent directory exists for dir/pdf modes
    if args.mode == "two_cohorts":
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)

    global HISTORY_SAMPLES, SCAN_FALLBACK_LIMIT_ROWS
    HISTORY_SAMPLES = int(args.max_samples)
    SCAN_FALLBACK_LIMIT_ROWS = int(args.scan_fallback_limit_rows)

    api = wandb.Api()

    if args.mode == "two_cohorts":
        if not args.name and not args.name2:
            raise SystemExit("two_cohorts mode requires --name (and optionally --name2)")
        plot_two_cohorts(args, api)
    else:
        plot_multipanel(args, api)

if __name__ == "__main__":
    main()
