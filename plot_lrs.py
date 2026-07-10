#!/usr/bin/env python3
"""
plot_lrs.py

One-panel plot of learning rates (W&B metric key: 'lr' by default).
- Fetches high-fidelity history per run (falls back to scan_history if needed).
- Bins native-step LR per seed, then aggregates across seeds (mean + band).
- Overlays three cohorts (run name patterns) with specified colors.
- Caps x-axis at a given limit (default: 24000).
pink: F07FDD
lime: 9BC750
red: F0434F
Example:
  python plot_lrs.py \
    --entity sheerio \
    --project workshop_MNIST \
    --group ka \
    --names "l2_f_1e-3_ply_ss10_250,crelu_1e-3_ply_rs1lll,wass_f_1e-3_ply_sqm10rrl_250" \
    --labels "L2, CReLU, Wass" \
    --colors "#f0434f,#9bc750,#f07fdd" \
    --bins 100 \
    --band minmax \
    --out plots/lr_three_overlaid.pdf

Notes:
- Matching is substring by default; use --match-mode exact for exact run-name match.
- You can change the metric key with --lr-key (default 'lr', lower case).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)

    # per-subplot titles
    p.add_argument("--title1", default=None, help="Optional title for left subplot")
    p.add_argument("--title2", default=None, help="Optional title for middle subplot")
    p.add_argument("--title3", default=None, help="Optional title for right subplot")

    p.add_argument("--group", default=None, help="Optional W&B group to filter runs")
    p.add_argument("--match-mode", choices=["exact","contains"], default="contains")

    # three panels, three runs each
    p.add_argument("--runs1", nargs="+", required=True)
    p.add_argument("--runs2", nargs="+", required=True)
    p.add_argument("--runs3", nargs="+", required=True)

    # optional labels
    p.add_argument("--labels1", nargs="*", default=None)
    p.add_argument("--labels2", nargs="*", default=None)
    p.add_argument("--labels3", nargs="*", default=None)

    # optional per-panel colors
    p.add_argument("--colors1", nargs="*", default=None)
    p.add_argument("--colors2", nargs="*", default=None)
    p.add_argument("--colors3", nargs="*", default=None)

    # dashed detection (useful to dash "reset" schedules etc.)
    p.add_argument("--dash-reset-substr", default="reset")

    # metrics & plotting
    p.add_argument("--lr-key", default="lr")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing factor (0 disables)")
    p.add_argument("--sort-by-step", action="store_true",
                   help="If set, sort points by '_step'. Recommended for LR.")
    p.add_argument("--show-points", action="store_true")
    p.add_argument("--max-points", type=int, default=None,
                   help="Plot at most this many points per run; uniformly subsamples if needed.")
    p.add_argument("--xmax", type=float, default=None,
                   help="Force x-axis max (step). If not set, uses max step over all plotted runs.")
    p.add_argument("--xmin", type=float, default=0.0,
                   help="Force x-axis min (step). Default: 0.0")

    # y-axis
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--ymin1", type=float, default=None)
    p.add_argument("--ymax1", type=float, default=None)
    p.add_argument("--ymin2", type=float, default=None)
    p.add_argument("--ymax2", type=float, default=None)
    p.add_argument("--ymin3", type=float, default=None)
    p.add_argument("--ymax3", type=float, default=None)

    p.add_argument("--title", default="", help="Optional suptitle")
    p.add_argument("--out", default="plots/lr_3x1.pdf")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ------------------------- helpers -------------------------

def ema(x, beta):
    if beta is None or beta <= 0.0 or np.isnan(beta): return np.asarray(x, dtype=float)
    y = np.empty_like(x, dtype=float)
    m = float(x[0])
    for i, v in enumerate(x):
        m = beta * m + (1.0 - beta) * float(v)
        y[i] = m
    return y

def fetch_df(run, key):
    want = ["_step", key]
    df = None
    try:
        # ask for plenty of samples to avoid server downsampling
        df = run.history(keys=want, pandas=True, samples=10_000_000)
    except Exception:
        df = None
    if df is None or df.empty or any(k not in df.columns for k in want):
        rows = []
        try:
            for row in run.scan_history():
                rows.append(row)
        except Exception:
            pass
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame(columns=["_step", key])
    keep = [k for k in want if k in df.columns]
    df = df[keep].dropna(subset=["_step"])
    return df.sort_values("_step")

def _uniform_subsample(x, y, max_points):
    if max_points is None or len(x) <= int(max_points):
        return x, y
    m = int(max_points)
    step = max(1, int(math.ceil(len(x) / m)))
    idx = np.arange(0, len(x), step)
    # ensure last sample included
    if idx[-1] != len(x)-1:
        idx = np.append(idx, len(x)-1)
    return x[idx], y[idx]

def find_matching_runs(api, entity, project, token, match_mode="contains", group=None):
    runs = api.runs(f"{entity}/{project}")
    token = token or ""
    out = []
    for r in runs:
        if group is not None and r.group != group:
            continue
        nm = r.display_name or r.name or ""
        ok = (nm == token) if match_mode == "exact" else (token in nm)
        if ok: out.append(r)
    out.sort(key=lambda r: r.created_at or 0, reverse=True)
    return out

def pick_most_recent_run(api, entity, project, token, match_mode="contains", group=None, debug=False):
    matches = find_matching_runs(api, entity, project, token, match_mode, group)
    if not matches and debug:
        print(f"[WARN] No run matched: '{token}'")
    return matches[0] if matches else None

def is_reset(label_or_token, needles=("reset",)):
    s = (label_or_token or "").lower()
    return any(n.lower() in s for n in needles if n)

def pick_linestyle(label_or_token, dash_needles):
    return "--" if is_reset(label_or_token, dash_needles) else "-"

def panel_colors_arg_to_list(arg):
    return list(arg) if arg is not None else []

# ------------------------- per-run series -------------------------

def collect_lr_series(api, args, token):
    r = pick_most_recent_run(api, args.entity, args.project, token,
                             args.match_mode, args.group, args.debug)
    if r is None:
        return {"label": token, "steps": None, "lr": None, "max_step": 0.0}
    df = fetch_df(r, key=args.lr_key)
    if df.empty or args.lr_key not in df.columns:
        return {"label": token, "steps": None, "lr": None, "max_step": 0.0}

    steps = df["_step"].to_numpy(dtype=float)
    vals  = df[args.lr_key].to_numpy(dtype=float)

    if args.sort_by_step:
        # already sorted; guard anyway
        idx = np.argsort(steps)
        steps, vals = steps[idx], vals[idx]

    # subsample if requested
    steps, vals = _uniform_subsample(steps, vals, args.max_points)

    # smooth (EMA on LR values, independent of step spacing)
    vals = ema(vals, args.smooth) if args.smooth and args.smooth > 0 else vals

    return {"label": token, "steps": steps, "lr": vals, "max_step": float(steps[-1]) if len(steps) else 0.0}

# ------------------------- main -------------------------

def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    api = wandb.Api()

    panels = [args.runs1, args.runs2, args.runs3]
    panel_labels = [args.labels1, args.labels2, args.labels3]
    panel_colors = [panel_colors_arg_to_list(args.colors1),
                    panel_colors_arg_to_list(args.colors2),
                    panel_colors_arg_to_list(args.colors3)]
    dash_needles = [args.dash_reset_substr] if args.dash_reset_substr else []

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)
    if not isinstance(axes, (list, np.ndarray)): axes = [axes]

    # collect first so we can align x-limits globally
    collected = []
    global_max_step = 0.0

    for tokens in panels:
        entries = [collect_lr_series(api, args, tok) for tok in tokens]
        collected.append(entries)
        for e in entries:
            global_max_step = max(global_max_step, e.get("max_step", 0.0))

    if args.xmax is not None:
        x_max = float(args.xmax)
    else:
        x_max = global_max_step if global_max_step > 0 else 1.0

    for pi, (ax, tokens, entries) in enumerate(zip(axes, panels, collected), start=1):
        labels = panel_labels[pi-1] if panel_labels[pi-1] else tokens
        colors = panel_colors[pi-1] if pi-1 < len(panel_colors) else []

        for idx, (label, token, entry) in enumerate(zip(labels, tokens, entries)):
            color = colors[idx] if idx < len(colors) else None
            ls = pick_linestyle(label or token, dash_needles)

            steps, lrs = entry["steps"], entry["lr"]
            if steps is None or lrs is None or len(steps) == 0: continue

            # clip to [xmin, xmax]
            mask = (steps >= float(args.xmin)) & (steps <= float(x_max))
            xs = steps[mask]
            ys = lrs[mask]
            if xs.size == 0: continue

            if args.show_points:
                ax.plot(xs, ys, linewidth=1.2, marker="o", markersize=3,
                        label=label, linestyle=ls, color=color)
            else:
                ax.plot(xs, ys, linewidth=1.2, label=label, linestyle=ls, color=color)

        ax.set_xlabel("Step")
        if pi == 1: ax.set_ylabel("Learning rate")

        # log scale y
        ax.set_yscale("log")

        panel_title = getattr(args, f"title{pi}")
        if panel_title: ax.set_title(panel_title)

        ymin_panel = getattr(args, f"ymin{pi}")
        ymax_panel = getattr(args, f"ymax{pi}")
        ymin = ymin_panel if ymin_panel is not None else args.ymin
        ymax = ymax_panel if ymax_panel is not None else args.ymax
        if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)

        # align x across panels
        ax.set_xlim(float(args.xmin), float(x_max))

    if args.title: fig.suptitle(args.title, y=1.02)
    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
