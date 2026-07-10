"""
One-run, three-metric plots from Weights & Biases.

EDIT THESE THREE KEYS at the top to match your metrics:
    METRIC1_KEY = "FC1/effective_lr"
    METRIC2_KEY = "FC1/tau"
    METRIC3_KEY = "FC1/alpha_crit_sqm10"

The script makes three PDFs:
  1) steps 0..11_700, y-log, clipped to [0.1, 0.3]
  2) steps 0..164_000, y-log, clipped, with EMA smoothing
  3) steps 153_000..164_000, y-log, clipped, with EMA smoothing

Example:
  python plot_eff.py \
  --entity sheerio --project workshop_MNIST --group crelu_1e-3_30_runs \
  --match-mode exact --run-name "crelu_1e-3lll" \
  --color1 "tab:blue" --color2 "tab:orange" --color3 "tab:red" --color4 "tab:green" \
  --out plots/fc1_crelu_metrics_subplots.pdf --aggregate-seeds --min-seeds 1 \
  --band std --keep-all-steps

"""
import argparse, os, time, signal
import numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb
from collections import defaultdict

# --------- METRIC KEYS (exact W&B names) ----------
METRIC1_KEY = "fc1/eff_lr"
METRIC2_KEY = "fc1/tau"
METRIC3_KEY = "fc1/alpha_crit_scv1"
METRIC4_KEY = "fc1/alpha_crit_rs10"
# --------------------------------------------------

# Legend labels
LABELS = [r"$\alpha_t$", r"$\alpha^{\star}_{\mathrm{vol}}$", r"$\alpha_g^*$", r"$\tilde\alpha^{\star}$"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--match-mode", choices=["exact","contains"], default="exact")

    # Aggregation across seeds (per STEP only)
    p.add_argument("--aggregate-seeds", action="store_true")
    p.add_argument("--min-seeds", type=int, default=2)

    # Colors
    p.add_argument("--color1", default=None, help="Matplotlib color for metric 1")
    p.add_argument("--color2", default=None, help="Matplotlib color for metric 2")
    p.add_argument("--color3", default=None, help="Matplotlib color for metric 3")
    p.add_argument("--color4", default=None, help="Matplotlib color for metric 4")
    # smoothing & ribbon
    p.add_argument("--smooth-seeds", type=float, default=0.0,
                   help="EMA beta applied to EACH seed before aggregation (0 disables).")
    p.add_argument("--band", choices=["std","iqr","q","minmax"], default="std",
                   help="Ribbon across seeds: std (±1σ), iqr (25–75), q ([q,1-q]), minmax.")
    p.add_argument("--quantile", type=float, default=0.1,
                   help="If --band q, use [q, 1-q] interval (e.g., 0.1).")
    p.add_argument("--keep-all-steps", action="store_true",
                   help="Keep mean for any step with ≥1 seed; hide ribbon when n < --min-seeds.")


    # Fetch / debug
    p.add_argument("--use-scan", action="store_true")
    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument("--debug", action="store_true")

    # Figure: width=8, height=5 (horizontal)
    p.add_argument("--figsize", type=float, nargs=2, default=(12.0, 2.5))
    p.add_argument("--wspace", type=float, default=0.08)
    p.add_argument("--out", default="plots/fc1_metrics_subplots.pdf")
    return p.parse_args()

# ---- utils ----

def ema(x, beta):
    if beta is None or beta <= 0.0 or np.isnan(beta): 
        return np.asarray(x, dtype=float)
    y = np.empty_like(x, dtype=float)
    m = float(x[0])
    for i, v in enumerate(x):
        m = beta*m + (1.0-beta)*float(v)
        y[i] = m
    return y

def _band_from_vals(vals, mode="std", q=0.1):
    vals = np.asarray(vals, float)
    m = vals.mean()
    mode = (mode or "std").lower()
    if mode == "minmax":
        lo, hi = vals.min(), vals.max()
    elif mode == "iqr":
        lo, hi = np.percentile(vals, 25), np.percentile(vals, 75)
    elif mode == "q":
        lo, hi = np.percentile(vals, 100*q), np.percentile(vals, 100*(1.0-q))
    else:
        sd = vals.std(ddof=0)
        lo, hi = m - sd, m + sd
    return m, lo, hi


def _timeout(seconds):
    class _Ctx:
        def __enter__(self2):
            if seconds <= 0: return
            def _handler(signum, frame): raise KeyboardInterrupt("timed out")
            signal.signal(signal.SIGALRM, _handler); signal.alarm(seconds)
        def __exit__(self2, exc_type, exc, tb):
            if seconds > 0: signal.alarm(0)
    return _Ctx()

def fetch_runs(entity, project, group, run_name, mode, keys, use_scan=False, limit_rows=0, timeout_seconds=0, debug=False):
    api = wandb.Api(); runs = api.runs(f"{entity}/{project}")
    def match(r):
        if r.group != group: return False
        nm = r.name or r.display_name or ""
        return (nm == run_name) if mode == "exact" else (run_name in nm)
    matched = sorted([r for r in runs if match(r)], key=lambda r: r.created_at)
    if not matched: raise SystemExit("No matching run(s).")
    if debug:
        print(f"[DEBUG] matched {len(matched)} run(s):")
        for r in matched: print(f"  - {r.id}  '{r.name}'  created={r.created_at}")

    want = list({"_step", *keys})
    dfs=[]
    for r in matched:
        try:
            with _timeout(timeout_seconds):
                if not use_scan:
                    # FULL fidelity: ask for a huge samples cap to avoid W&B downsampling
                    df = r.history(keys=want, pandas=True, samples=10_000_000)
                    if limit_rows>0 and df is not None and not df.empty:
                        df = df.head(limit_rows).copy()
                else:
                    rows=[]; n=0
                    for row in r.scan_history(keys=want):
                        rows.append(row); n+=1
                        if limit_rows>0 and n>=limit_rows: break
                    df=pd.DataFrame(rows)
        except KeyboardInterrupt as e:
            print(f"[WARN] aborted {r.id}: {e}"); continue

        if df is None or df.empty:
            if debug: print(f"[DEBUG] {r.id}: empty for requested keys")
            continue
        df = df.dropna(subset=["_step"]).sort_values("_step")
        if debug:
            present=[k for k in keys if k in df.columns]
            smin,smax=int(df["_step"].min()),int(df["_step"].max())
            print(f"[DEBUG] {r.id}: rows={len(df)} steps=[{smin},{smax}] present={present}")
        dfs.append((r, df))
    if not dfs: raise SystemExit("No run had the requested metrics.")
    return dfs

def prep_series(panel: pd.DataFrame, key: str):
    if key not in panel.columns: return None
    y = panel[key].to_numpy().astype(float)
    if np.all(np.isnan(y)): return None
    return pd.Series(y).ffill().bfill().to_numpy()

def aggregate_across_seeds_per_step(
    dfs, keys, min_seeds=2, debug=False, smooth_beta=0.0, band_mode="std", band_q=0.1,
    keep_all_steps=False
):
    """
    Aggregate ACROSS SEEDS per *global step*, preserving full x-density:
      - Build the union of all steps seen across matched runs.
      - For each run/key, align to that union with ffill/bfill, then
        optionally smooth per-seed with EMA, and contribute to that step.
      - Produce DataFrames with columns: _step, mean, lo, hi, n
        * If keep_all_steps: keep mean for any n>=1; lo/hi become NaN when n<min_seeds
        * Else: drop positions where n<min_seeds (classic behavior)
    """
    # 1) union of steps
    union_steps = sorted({
        int(s) for (_r, df) in dfs
        for s in df.get("_step", pd.Series([], dtype=float)).dropna().astype(int).tolist()
    })
    if not union_steps:
        return {k: pd.DataFrame(columns=["_step","mean","lo","hi","n"]) for k in keys}

    union_idx = pd.Index(union_steps, name="_step")

    # 2) per-key: collect per-step lists across seeds
    per_key_vals = {k: {s: [] for s in union_steps} for k in keys}

    for (_r, df) in dfs:
        if df is None or df.empty: 
            continue
        df = df.dropna(subset=["_step"]).copy()
        df["_step"] = df["_step"].astype(int)
        df = df.sort_values("_step")

        for k in keys:
            if k not in df.columns:
                continue
            s = pd.Series(df[k].astype(float).ffill().bfill().values, index=df["_step"].values)
            # align to union with ffill then bfill for any leading NaNs
            s = s.reindex(union_idx).ffill().bfill()
            # per-seed smoothing (EMA) along steps
            arr = ema(s.to_numpy(dtype=float), smooth_beta) if smooth_beta and smooth_beta > 0 else s.to_numpy(dtype=float)

            for step, v in zip(union_steps, arr):
                if not np.isnan(v):
                    per_key_vals[k][step].append(float(v))

    # 3) summarize across seeds per step
    out = {}
    for k in keys:
        rows = []
        for step in union_steps:
            vals = per_key_vals[k][step]
            n = len(vals)
            if n == 0:
                continue
            if keep_all_steps:
                # keep the mean; only show ribbon when we have enough seeds
                m, lo, hi = _band_from_vals(vals, band_mode, band_q)
                if n < min_seeds:
                    lo, hi = np.nan, np.nan
                rows.append([step, m, lo, hi, n])
            else:
                # classical: only keep positions with enough seeds
                if n >= min_seeds:
                    m, lo, hi = _band_from_vals(vals, band_mode, band_q)
                    rows.append([step, m, lo, hi, n])
        out[k] = pd.DataFrame(rows, columns=["_step","mean","lo","hi","n"]).sort_values("_step")
        if debug and not out[k].empty:
            smin, smax = int(out[k]["_step"].min()), int(out[k]["_step"].max())
            print(f"[DEBUG] agg {k}: rows={len(out[k])} steps=[{smin},{smax}] (band={band_mode}, keep_all={keep_all_steps})")
    return out

def format_k(x):
    x = float(x)
    if abs(x) >= 1000:
        val = x/1000.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-6 else f"{val:.1f}k"
    return f"{int(x)}"

# ---- plotting ----
def plot_three(axs, panels, keys, labels, colors, show_band=False, debug=False):
    for ax, (panel, x_lo, x_hi, aggregated) in zip(axs, panels):
        plotted_pts = 0
        if aggregated:
            for (k, lab, color) in zip(keys, labels, colors):
                d = panel.get(k, pd.DataFrame())
                d = d[(d["_step"]>=x_lo)&(d["_step"]<=x_hi)]
                if d.empty: continue
                ax.plot(d["_step"], d["mean"], lw=0.9, label=lab, color=color)
                if show_band:
                    ax.fill_between(d["_step"], d["lo"], d["hi"], alpha=0.12, color=color)
                plotted_pts += len(d)
        else:
            p = panel[(panel["_step"]>=x_lo)&(panel["_step"]<=x_hi)]
            if not p.empty:
                x = p["_step"].to_numpy().astype(float)
                for (k, lab, color) in zip(keys, labels, colors):
                    y = prep_series(p, k)
                    if y is None: continue
                    ax.plot(x, y, lw=0.9, label=lab, color=color)
                    plotted_pts += len(x)

        ax.set_xlim(x_lo, x_hi)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 70)

        # horizontal grid only, very faint
        ax.yaxis.grid(True, which="both", linestyle="-", linewidth=0.3, alpha=0.1)
        ax.xaxis.grid(False)

        # tidy spines
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # show ONLY first and last x tick, with "k" suffix
        ax.set_xticks([x_lo, x_hi])
        # ax.set_xlabel("Steps", labelpad=1)
        ax.set_xticklabels([format_k(x_lo), format_k(x_hi)])
        ymin, ymax = ax.get_ylim()
        ax.set_yticks([ymin, 1.0, ymax])
        ax.set_yticklabels([f"{ymin:g}", r"$10^0$", f"{ymax:g}"])

        if debug:
            print(f"[DEBUG] panel [{format_k(x_lo)}..{format_k(x_hi)}]: points={plotted_pts}")

def main():
    # global fidelity: keep every vertex
    plt.rcParams.update({
        "path.simplify": False,
        "path.simplify_threshold": 0.0,
        "agg.path.chunksize": 0,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 12,
    })

    args = parse_args()
    keys   = [METRIC1_KEY, METRIC2_KEY, METRIC3_KEY, METRIC4_KEY]
    labels = LABELS
    colors = [args.color1, args.color2, args.color3, args.color4]

    dfs = fetch_runs(args.entity, args.project, args.group, args.run_name, args.match_mode,
                     keys, use_scan=args.use_scan, limit_rows=args.limit_rows,
                     timeout_seconds=args.timeout_seconds, debug=args.debug)

    if not args.aggregate_seeds and len(dfs) != 1:
        raise SystemExit(f"Matched {len(dfs)} runs but --aggregate-seeds not set. "
                         f"Use --aggregate-seeds to aggregate per step, or narrow the match.")

    aggregating = bool(args.aggregate_seeds and len(dfs) > 1)
    print(f"[INFO] aggregating across seeds per step? {aggregating} (matched_runs={len(dfs)})")

    fig, axs = plt.subplots(
        1, 3, figsize=tuple(args.figsize),
        gridspec_kw={"width_ratios": [2, 3, 2]},  # 2:3:2
        sharey=True
    )
    # tighter horizontal gap
    fig.subplots_adjust(wspace=0.1)

    if aggregating:
        agg = aggregate_across_seeds_per_step(
            dfs, keys,
            min_seeds=args.min_seeds,
            debug=args.debug,
            smooth_beta=args.smooth_seeds,
            band_mode=args.band,
            band_q=args.quantile,
            keep_all_steps=args.keep_all_steps,
        )
        panels = [
            (agg,   0,      5_700,  True),
            (agg,   0,     63_100,  True),
            (agg, 58_800, 63_100,  True),
        ]
        # after computing `agg`, before plot_three(...)
        if args.debug: print("[DEBUG] plotting individual seeds as faint background")
        for ax, (x_lo, x_hi) in zip(axs, [(0,11700),(0,117500),(105800,117500)]):
            for (_r, df) in dfs:
                p = df[(df["_step"]>=x_lo)&(df["_step"]<=x_hi)]
                x = p["_step"].to_numpy().astype(float)
                for (k, color) in zip(keys, colors):
                    y = prep_series(p, k)
                    if y is None: continue
                    ax.plot(x, y, lw=0.4, alpha=0.2, color=color)

        plot_three(axs, panels, keys, labels, colors, show_band=True, debug=args.debug)
    else:
        run, df = dfs[0]
        if args.debug:
            smin, smax = int(df["_step"].min()), int(df["_step"].max())
            present = [k for k in keys if k in df.columns]
            print(f"[DEBUG] single-run {run.id}: steps=[{smin},{smax}] keys_present={present}")
        panels = [
            (df,   0,      5_700,  False),
            (df,   0,     63_100,  False),
            (df, 58_800, 63_100,  False),
        ]
        plot_three(axs, panels, keys, labels, colors, show_band=False, debug=args.debug)

    # One compact legend at top
    h, l = axs[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.text(0.5, 0.05, "Steps", ha="center", va="bottom")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
