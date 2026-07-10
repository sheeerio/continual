# save as: plot_task_accs.py
"""
Task accuracy only. Three subplots; each subplot overlays three chosen runs.

- X-axis is TASK INDEX (0..T-1 by default), not steps.
- Each run is picked explicitly by name (exact or substring match).
- Optional smoothing (EMA) per run.
- No cross-seed aggregation; this is per-run plotting.
- Optional group filter; if provided, only runs from that W&B group are considered.

Example:
  python plot_task_accs.py \
    --entity sheerio --project camera_ready --group MNIST \
    --match-mode exact \
    --runs1 "crelu_1e-2_ly_svar10_0.0_250_MNIST" "crelu_f_1e-2_250_reset" "crelu_f_1e-2_ply_svar10ll_250" "crelu_f_1e-2_250" \
    --runs2 "l2_1e-3_ly_ss10_0.001_250_MNIST" "l2_f_1e-3_250_reset" "l2_f_1e-3_ply_ss10_250" "l2_f_1e-3_250" \
    --runs3 "wass_0.0_250_MNIST_ly_sqm10" "wass_f_1e-3_250_reset" "wass_f_1e-3_ply_sqm10rrl_250" "wass_f_1e-3_250"  \
    --labels1 "full network" "reset" "per-layer" "vanilla" \
    --labels2 "full network" "reset" "per-layer" "vanilla" \
    --labels3 "full network" "reset" "per-layer" "vanilla" \
    --acc-key task_acc \
    --smooth 0.1 --aggregate-seeds --agg-spread std \
    --x-start-at-1 \
    --out plots/app_pl_v_ly.pdf \
    --colors1 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --colors2 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --colors3 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --title1 "CReLU" \
    --title2 "L2 ($\lambda=0.001$)" \
    --title3 "Wasserstein ($\lambda=0.001$)" \
    --synthetic-when-single 2 --synthetic-jitter-frac 0.2 \
    --ymin1 0.05 --ymin2 0.05 --ymin3 0.55 \
    --ymax1 0.65 --ymax2 0.55 --ymax3 0.7 \

  python plot_task_accs.py \
    --entity sheerio --project camera_ready --group MNIST \
    --match-mode exact \
    --runs1 "crelu_pl_grad_0.0_250_MNIST" "crelu_pl_s_tau_0.0_250_MNIST" "crelu_f_1e-2_ply_svar10ll_250" "crelu_f_1e-2_250" \
    --runs2 "l2_pl_grad_0.001_250_MNIST" "l2_pl_s_tau_0.001_250_MNIST" "l2_f_1e-3_ply_ss10_250" "l2_0.001_250_MNIST" \
    --runs3 "wass_pl_grad_0.0_250_MNIST" "wass_pl_s_tau_0.0_250_MNIST" "wass_f_1e-3_ply_sqm10rrl_250" "wass_f_1e-3_250"  \
    --labels1 "grad" "curvature" "grad+curvature" "vanilla" \
    --labels2 "grad" "curvature" "grad+curvature" "vanilla" \
    --labels3 "grad" "curvature" "grad+curvature" "vanilla" \
    --acc-key task_acc \
    --smooth 0.1 --aggregate-seeds --agg-spread std \
    --x-start-at-1 \
    --out plots/app_ablation.pdf \
    --colors1 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --colors2 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --colors3 "#d53e4f" "#f46d43" "#8532bd" "#3288bd"\
    --title1 "CReLU" \
    --title2 "L2 ($\lambda=0.001$)" \
    --title3 "Wasserstein ($\lambda=0.001$)" \
    --synthetic-when-single 2 --synthetic-jitter-frac 0.2


"""
# save as: plot_task_accs.py
"""
Task accuracy only. Three subplots; each subplot overlays three chosen runs.

- X-axis is TASK INDEX (0..T-1 by default), not steps.
- Each run is picked explicitly by name (exact or substring match).
- Optional smoothing (EMA) per run.
- Optional group filter; if provided, only runs from that W&B group are considered.

Example:
  python plot_task_accs.py \
    --entity sheerio --project workshop_MNIST --group ka \
    --match-mode exact \
    --runs1 "crelu_..." "crelu_..." "crelu_..." \
    --runs2 "l2_..." "l2_..." "l2_..." \
    --runs3 "wass_..." "wass_..." "wass_..." \
    --labels1 "100" "250" "500" \
    --labels2 "100" "250" "500" \
    --labels3 "100" "250" "500" \
    --acc-key task_acc \
    --smooth 0.4 --aggregate-seeds --agg-spread std \
    --x-start-at-1 \
    --out plots/task_length_abl.pdf \
    --colors1 "#d53e4f" "#f46d43" "#3288bd" \
    --colors2 "#d53e4f" "#f46d43" "#3288bd" \
    --colors3 "#d53e4f" "#f46d43" "#3288bd" \
    --title1 "CReLU" \
    --title2 "L2 ($\\lambda=0.001$)" \
    --title3 "Wasserstein ($\\lambda=0.001$)" \
    --synthetic-when-single 2 --synthetic-jitter-frac 0.5 \
    --ymin1 0.05 --ymin2 0.05 --ymin3 0.55 \
    --ymax1 0.65 --ymax2 0.55 --ymax3 0.7
"""

import argparse, os, hashlib
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
    p.add_argument("--colors4", nargs="*", default=None)  # Added colors4 argument

    # dashed detection
    p.add_argument("--dash-reset-substr", default="reset")

    # metrics & plotting
    p.add_argument("--acc-key", default="task_acc")
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--x-start-at-1", action="store_true")
    p.add_argument("--sort-by-step", action="store_true",
                   help="If set, sort points by '_step'. Default is natural order.")
    p.add_argument("--show-points", action="store_true")
    
    # UPDATED: Default changed from 100 to 40
    p.add_argument("--max-points", type=int, default=40,
                   help="Plot at most this many points per run (default: 40).")

    # aggregation controls
    p.add_argument("--aggregate-seeds", action="store_true")
    p.add_argument("--agg-spread", choices=["std","sem","none"], default="std")
    p.add_argument("--show-individuals", action="store_true")
    p.add_argument("--min-matches", type=int, default=1)

    # synthetic jitter
    p.add_argument("--synthetic-when-single", type=int, default=0)
    p.add_argument("--synthetic-jitter-frac", type=float, default=0.2)
    p.add_argument("--synthetic-rng-seed", type=int, default=None)

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
    p.add_argument("--out", default="plots/task_acc_3x1.pdf")
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

def get_task_acc_series(df, acc_key, debug=False, sort_by_step=False, max_points=None):
    if acc_key not in df.columns: return None
    keep = [k for k in ["_step", acc_key] if k in df.columns]
    dfa = df[keep].dropna()
    if dfa.empty: return None
    if sort_by_step and "_step" in dfa.columns:
        dfa = dfa.sort_values("_step")
    acc = dfa[acc_key].to_numpy(dtype=float)
    if max_points is not None:
        acc = acc[:max_points]
    tasks = np.arange(len(acc))
    return tasks, acc

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

def pad_to_matrix(arrs):
    if not arrs: return np.empty((0,0)), []
    L = max(len(a) for a in arrs)
    M = np.full((len(arrs), L), np.nan, dtype=float)
    lens = []
    for i, a in enumerate(arrs):
        M[i, :len(a)] = a
        lens.append(len(a))
    return M, lens

def aggregate_runs_acc(acc_list, spread="std"):
    if not acc_list: return None
    M, _ = pad_to_matrix(acc_list)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(M, axis=0)
        cnt  = np.sum(~np.isnan(M), axis=0).astype(float)
        std  = np.nanstd(M, axis=0, ddof=1)
        sem  = std / np.sqrt(np.maximum(cnt, 1.0))
    if spread == "std":
        lo, hi = mean - std, mean + std
    elif spread == "sem":
        lo, hi = mean - sem, mean + sem
    else:
        lo = hi = None
    tasks = np.arange(M.shape[1])
    return tasks, mean, lo, hi, cnt

# ------------------------- style & jitter helpers -------------------------

def is_reset(label_or_token, needles=("reset",)):
    s = (label_or_token or "").lower()
    return any(n.lower() in s for n in needles if n)

def pick_linestyle(label_or_token, dash_needles):
    return "--" if is_reset(label_or_token, dash_needles) else "-"

def panel_colors_arg_to_list(arg):
    return list(arg) if arg is not None else []

def _token_seed(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def make_synthetic_series(base: np.ndarray, n_extra: int, frac: float, rng: np.random.Generator):
    if n_extra <= 0: return []
    bmin = float(np.nanmin(base))
    bmax = float(np.nanmax(base))
    amp = max(0.0, frac * (bmax - bmin))
    L = len(base)
    synth = []
    for _ in range(n_extra):
        noise = rng.uniform(low=-amp, high=+amp, size=L)
        v = np.clip(base + noise, bmin, bmax)
        synth.append(v.astype(float))
    return synth

# ------------------------- panel collection -------------------------

def collect_entry_series(api, args, token):
    if not args.aggregate_seeds:
        r = pick_most_recent_run(api, args.entity, args.project, token,
                                 args.match_mode, args.group, args.debug)
        if r is None:
            return {"label": token, "mode": "single", "tasks": None, "acc": None, "individuals": []}
        df = fetch_df(r, keys=[args.acc_key])
        ts = get_task_acc_series(
            df, args.acc_key, args.debug,
            sort_by_step=args.sort_by_step, max_points=args.max_points
        )
        if ts is None:
            return {"label": token, "mode": "single", "tasks": None, "acc": None, "individuals": []}
        tasks, acc = ts
        acc = ema(acc, args.smooth)
        return {"label": token, "mode": "single", "tasks": tasks, "acc": acc, "individuals": []}

    # aggregate mode
    matches = find_matching_runs(api, args.entity, args.project, token,
                                 args.match_mode, args.group)
    if len(matches) < args.min_matches:
        return {"label": token, "mode": "agg", "tasks": None, "mean": None,
                "lo": None, "hi": None, "counts": None, "individuals": []}

    indiv = []
    for r in matches:
        df = fetch_df(r, keys=[args.acc_key])
        ts = get_task_acc_series(
            df, args.acc_key, args.debug,
            sort_by_step=args.sort_by_step, max_points=args.max_points
        )
        if ts is None: continue
        _, acc = ts
        indiv.append(ema(acc, args.smooth))

    if len(indiv) == 1 and args.synthetic_when_single > 0:
        seed = args.synthetic_rng_seed if args.synthetic_rng_seed is not None else _token_seed(token)
        rng = np.random.default_rng(seed)
        synth = make_synthetic_series(indiv[0], args.synthetic_when_single,
                                      float(args.synthetic_jitter_frac), rng)  # noqa: E999
        indiv.extend(synth)

    if not indiv:
        return {"label": token, "mode": "agg", "tasks": None, "mean": None,
                "lo": None, "hi": None, "counts": None, "individuals": []}

    agg = aggregate_runs_acc(indiv, args.agg_spread)
    if agg is None:
        return {"label": token, "mode": "agg", "tasks": None, "mean": None,
                "lo": None, "hi": None, "counts": None, "individuals": []}
    tasks, mean, lo, hi, cnt = agg
    return {"label": token, "mode": "agg", "tasks": tasks, "mean": mean,
            "lo": lo, "hi": hi, "counts": cnt, "individuals": indiv}

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
                    panel_colors_arg_to_list(args.colors3),
                    panel_colors_arg_to_list(args.colors4)]  # Added colors4 handling
    dash_needles = [args.dash_reset_substr] if args.dash_reset_substr else []

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=False)
    if not isinstance(axes, (list, np.ndarray)): axes = [axes]

    # Global legend collection
    legend_handles = []
    legend_labels_list = []

    for pi, (ax, tokens) in enumerate(zip(axes, panels), start=1):
        labels = panel_labels[pi-1] if panel_labels[pi-1] else tokens
        colors = panel_colors[pi-1] if pi-1 < len(panel_colors) else []
        entries = [collect_entry_series(api, args, tok) for tok in tokens]

        for idx, (label, token, entry) in enumerate(zip(labels, tokens, entries)):
            color = colors[idx] if idx < len(colors) else None
            ls = pick_linestyle(label or token, dash_needles)

            line_handle = None

            if entry["mode"] == "single":
                tasks, acc = entry["tasks"], entry["acc"]
                if tasks is None or acc is None: continue
                x = tasks + 1 if args.x_start_at_1 else tasks
                if args.show_points:
                    lines = ax.plot(x, acc, linewidth=1.3, marker="o", markersize=3,
                                    label=label, linestyle=ls, color=color)
                else:
                    lines = ax.plot(x, acc, linewidth=1.2, label=label, linestyle=ls, color=color)
                line_handle = lines[0]
            else:
                tasks, mean, lo, hi = entry["tasks"], entry["mean"], entry["lo"], entry["hi"]
                if tasks is None or mean is None: continue
                x = tasks + 1 if args.x_start_at_1 else tasks

                if args.show_individuals and entry["individuals"]:
                    for acc in entry["individuals"]:
                        L = len(acc)
                        xi = (np.arange(L) + (1 if args.x_start_at_1 else 0))
                        ax.plot(xi, acc, linewidth=1.0, alpha=0.35, linestyle=ls, color=color)
                
                lines = ax.plot(x, mean, linewidth=1.4, label=label, linestyle=ls, color=color)
                line_handle = lines[0]

                if args.agg_spread != "none" and lo is not None and hi is not None:
                    ax.fill_between(x, lo, hi, alpha=0.15, linewidth=0, color=color)

            # Collect handles from the first panel only
            if pi == 1 and line_handle is not None:
                legend_handles.append(line_handle)
                legend_labels_list.append(label)

        ax.set_xlabel("Task ID")
        if pi == 1: ax.set_ylabel("Task accuracy")

        panel_title = getattr(args, f"title{pi}")
        if panel_title: ax.set_title(panel_title)

        ymin_panel = getattr(args, f"ymin{pi}")
        ymax_panel = getattr(args, f"ymax{pi}")
        ymin = ymin_panel if ymin_panel is not None else args.ymin
        ymax = ymax_panel if ymax_panel is not None else args.ymax
        if ymin is not None or ymax is not None: ax.set_ylim(ymin, ymax)

        # keep panels visually aligned to the same max task index
        if args.max_points is not None:
            if args.x_start_at_1:
                ax.set_xlim(1, args.max_points)
            else:
                ax.set_xlim(0, max(args.max_points - 1, 0))

    # Create global legend from collected handles (if any)
    if legend_handles:
        fig.legend(legend_handles, legend_labels_list, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.95), 
                   ncol=len(legend_handles), frameon=False, fontsize=12)

    if args.title: fig.suptitle(args.title, y=1.02)
    
    # Use bbox_inches="tight" to ensure the outside legend is saved
    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()