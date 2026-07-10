# save as: plot_effs_layers_robust.py
"""
Four-subplot comparison of Tau vs Effective LR.
Range: Steps 0 to 11,700 (Log Scale).

Structure:
  Panel 1: Global (tau, effective_lr)
  Panel 2: FC1    (fc1/tau, fc1/eff_lr)
  Panel 3: FC2    (fc2/tau, fc2/eff_lr)
  Panel 4: FC4    (fc4/tau, fc4/eff_lr)

Usage:
  python plot_effs_layers.py \
    --entity sheerio --project camera_ready --group MNIST \
    --run-name "relu" \
    --match-mode exact \
    --color-tau "tab:blue" --color-lr "tab:orange" \
    --out plots/relu_metrics_4x1.pdf \
    --debug --X_MAX 470

python plot_effs_layers.py \
    --entity sheerio --project camera_ready --group MNIST \
    --run-name "crelu_f_1e-2" \
    --match-mode exact \
    --color-tau "tab:blue" --color-lr "tab:orange" \
    --out plots/crelu_metrics_4x1.pdf \
    --debug --X_MAX 4400 --ymin 0.00001 --ymax 100

python plot_effs_layers.py \
    --entity sheerio --project camera_ready --group MNIST \
    --run-name "l2_0.001_250_MNIST" \
    --match-mode exact \
    --color-tau "tab:blue" --color-lr "tab:orange" \
    --out plots/l2_metrics_4x1.pdf \
    --debug --param "ss10"

python plot_effs_layers.py \
    --entity sheerio --project camera_ready --group MNIST \
    --run-name "wass_f_1e-3_250" \
    --match-mode exact \
    --color-tau "tab:blue" --color-lr "tab:orange" \
    --out plots/wass_metrics_4x1.pdf \
    --debug --param "sqm10"
"""

# save as: plot_effs_layers_tasks.py
"""
Four-subplot comparison of Tau vs Effective LR.
Range: Steps 0 to X_MAX, labeled as Task 1 and Task 2.
Scale: Logarithmic Y.

Structure:
  Panel 1: Full network
  Panel 2: Layer 1
  Panel 3: Layer 2
  Panel 4: Layer 3 (fc4)
"""

import argparse, os, signal
import numpy as np, pandas as pd, matplotlib.pyplot as plt, wandb

# --------- CONFIGURATION ----------

LABEL_TAU = r"$\tilde{\alpha}^*_t$"
LABEL_LR  = r"$\alpha_t$"
# ----------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--match-mode", choices=["exact","contains"], default="exact")

    p.add_argument("--aggregate-seeds", action="store_true")
    p.add_argument("--min-seeds", type=int, default=2)

    p.add_argument("--color-tau", default="tab:blue")
    p.add_argument("--color-lr", default="tab:orange")
    p.add_argument("--param", choices=["svar10","sqm10","ss10"], default="svar10")
    p.add_argument("--smooth-seeds", type=float, default=0.0)
    p.add_argument("--band", choices=["std","iqr","q","minmax"], default="std")
    p.add_argument("--quantile", type=float, default=0.1)
    p.add_argument("--keep-all-steps", action="store_true")
    p.add_argument("--X_MAX", type=int, default=11700)
    p.add_argument("--ymin", type=float, default=0.01)
    p.add_argument("--ymax", type=float, default=10.0)

    p.add_argument("--limit-rows", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=0)
    p.add_argument("--debug", action="store_true")

    p.add_argument("--figsize", type=float, nargs=2, default=(16.0, 3.0))
    p.add_argument("--out", default="plots/layer_metrics_4x1.pdf")
    return p.parse_args()

# ---- Utils ----

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

def fetch_runs(entity, project, group, run_name, mode, keys, limit_rows=0, timeout_seconds=0, debug=False):
    print(f" -> Connecting to WandB: {entity}/{project} (Group: {group})...")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    def match(r):
        if r.group != group: return False
        nm = r.name or r.display_name or ""
        return (nm == run_name) if mode == "exact" else (run_name in nm)
    
    matched = sorted([r for r in runs if match(r)], key=lambda r: r.created_at)
    if not matched: 
        raise SystemExit(f" [ERROR] No runs matched. Check your --group ('{group}') and --run-name ('{run_name}').")
    
    print(f" -> Found {len(matched)} matching run(s). Fetching ALL history (client-side filtering)...")
    
    dfs = []
    for r in matched:
        df = None
        
        # --- ATTEMPT 1: Fast Fetch (history) ---
        try:
            with _timeout(timeout_seconds):
                df = r.history(pandas=True, samples=100_000)
                if limit_rows > 0 and df is not None and not df.empty:
                    df = df.head(limit_rows).copy()
        except Exception as e:
            if debug: print(f"   [Debug] Fast fetch failed for {r.name}: {e}")

        if df is None or df.empty:
            # Fallback: Try scan_history WITHOUT keys
            if debug: print(f"   [Debug] {r.name}: History empty. Trying scan_history (full)...")
            try:
                rows = []
                count = 0
                for row in r.scan_history(): # No keys arg!
                    rows.append(row)
                    count += 1
                    if limit_rows > 0 and count >= limit_rows: break
                df = pd.DataFrame(rows)
            except Exception as e:
                print(f"   [WARN] scan_history failed for {r.name}: {e}")

        if df is None or df.empty:
            print(f"   [WARN] {r.name}: Could not retrieve any data.")
            continue
        
        # --- CLIENT SIDE FILTERING ---
        available_keys = set(df.columns)
        wanted_keys = set(keys)
        found = wanted_keys.intersection(available_keys)
        
        if not found:
            if debug: 
                print(f"   [Debug] {r.name}: Fetched {len(df)} rows, but none of the requested keys were found.")
            continue

        keep_cols = list(found)
        if "_step" in available_keys:
            keep_cols.append("_step")
        
        df_filtered = df[keep_cols].copy()

        if "_step" in df_filtered.columns:
            df_filtered = df_filtered.dropna(subset=["_step"]).sort_values("_step")
            dfs.append((r, df_filtered))
        else:
            if debug: print(f"   [Debug] {r.name}: Missing '_step' column.")
        
    if not dfs: 
        raise SystemExit(" [ERROR] Matched runs found, but failed to extract metrics.")
    return dfs

def prep_series(panel: pd.DataFrame, key: str):
    if key not in panel.columns: return None
    y = panel[key].to_numpy().astype(float)
    if np.all(np.isnan(y)): return None
    return pd.Series(y).ffill().bfill().to_numpy()

def aggregate_across_seeds_per_step(dfs, keys, min_seeds=2, debug=False, smooth_beta=0.0, band_mode="std", band_q=0.1, keep_all_steps=False):
    union_steps = sorted({
        int(s) for (_r, df) in dfs
        for s in df.get("_step", pd.Series([], dtype=float)).dropna().astype(int).tolist()
    })
    if not union_steps:
        return {k: pd.DataFrame(columns=["_step","mean","lo","hi","n"]) for k in keys}
    
    union_idx = pd.Index(union_steps, name="_step")
    per_key_vals = {k: {s: [] for s in union_steps} for k in keys}

    for (_r, df) in dfs:
        if df is None or df.empty: continue
        df = df.dropna(subset=["_step"]).copy()
        df["_step"] = df["_step"].astype(int)
        df = df.sort_values("_step")
        
        for k in keys:
            if k not in df.columns: continue
            s = pd.Series(df[k].astype(float).ffill().bfill().values, index=df["_step"].values)
            s = s.reindex(union_idx).ffill().bfill()
            arr = ema(s.to_numpy(dtype=float), smooth_beta) if smooth_beta > 0 else s.to_numpy(dtype=float)
            for step, v in zip(union_steps, arr):
                if not np.isnan(v): per_key_vals[k][step].append(float(v))

    out = {}
    for k in keys:
        rows = []
        for step in union_steps:
            vals = per_key_vals[k][step]
            n = len(vals)
            if n == 0: continue
            if keep_all_steps or n >= min_seeds:
                m, lo, hi = _band_from_vals(vals, band_mode, band_q)
                if keep_all_steps and n < min_seeds: lo, hi = np.nan, np.nan
                rows.append([step, m, lo, hi, n])
        out[k] = pd.DataFrame(rows, columns=["_step","mean","lo","hi","n"]).sort_values("_step")
    return out

def format_k(x):
    val = float(x) / 1000.0
    return f"{val:.1f}k" if abs(val) >= 1.0 else f"{int(x)}"

# ---- Plotting Logic ----

def main():
    # Style
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 11,
        "path.simplify": False
    })

    args = parse_args()
    X_MAX = args.X_MAX # Total steps for both tasks
    X_MIDPOINT = X_MAX / 2 # The dividing line between Task 1 and Task 2
    if args.out == "plots/l2_metrics_4x1.pdf":
        PANEL_CONFIG = [
            {"title": "Full network", "tau": f"alpha_crit_k_{args.param}", "lr": "effective_lr"},
            {"title": "Layer 1", "tau": f"fc1/alpha_crit_{args.param}", "lr": "fc1/eff_lr"},
            {"title": "Layer 2", "tau": f"fc2/alpha_crit_{args.param}", "lr": "fc2/eff_lr"},
            {"title": "Layer 3", "tau": f"fc4/alpha_crit_{args.param}", "lr": "fc4/eff_lr"},
        ]
    else:
        PANEL_CONFIG = [
            {"title": "Full network", "tau": f"alpha_crit_s_{args.param}", "lr": "effective_lr"},
            {"title": "Layer 1", "tau": f"fc1/alpha_crit_{args.param}", "lr": "fc1/eff_lr"},
            {"title": "Layer 2", "tau": f"fc2/alpha_crit_{args.param}", "lr": "fc2/eff_lr"},
            {"title": "Layer 3", "tau": f"fc4/alpha_crit_{args.param}", "lr": "fc4/eff_lr"},
        ]


    # Collect all unique keys needed for fetching
    all_keys = set()
    for pcfg in PANEL_CONFIG:
        all_keys.add(pcfg["tau"])
        all_keys.add(pcfg["lr"])
    all_keys = list(all_keys)

    # Fetch
    dfs = fetch_runs(args.entity, args.project, args.group, args.run_name, args.match_mode,
                     all_keys, limit_rows=args.limit_rows,
                     timeout_seconds=args.timeout_seconds, debug=args.debug)

    # Aggregate
    aggregating = bool(args.aggregate_seeds and len(dfs) > 1)
    if args.aggregate_seeds:
        print(f" -> Aggregating {len(dfs)} runs.")

    data_map = {}
    if aggregating:
        data_map = aggregate_across_seeds_per_step(
            dfs, all_keys, min_seeds=args.min_seeds, debug=args.debug,
            smooth_beta=args.smooth_seeds, band_mode=args.band, band_q=args.quantile,
            keep_all_steps=args.keep_all_steps
        )
    else:
        # Single run wrapper
        r, df = dfs[0]
        print(f" -> Plotting single run: {r.name}")
        for k in all_keys:
            if k in df.columns:
                sub = df[["_step", k]].dropna().rename(columns={k: "mean"})
                sub["lo"] = sub["mean"]
                sub["hi"] = sub["mean"]
                sub["n"]  = 1
                data_map[k] = sub
            else:
                data_map[k] = pd.DataFrame()

    # Plot 4 subplots
    fig, axs = plt.subplots(1, 4, figsize=tuple(args.figsize), sharey=True)
    if not isinstance(axs, (list, np.ndarray)): axs = [axs]
    
    # Reduce horizontal space
    fig.subplots_adjust(wspace=0.1, left=0.05, right=0.98, bottom=0.2)

    for i, (ax, pcfg) in enumerate(zip(axs, PANEL_CONFIG)):
        key_tau = pcfg["tau"]
        key_lr  = pcfg["lr"]
        
        # Plot Tau
        d_tau = data_map.get(key_tau, pd.DataFrame())
        if not d_tau.empty:
            d_tau = d_tau[d_tau["_step"] <= X_MAX]
            if not d_tau.empty:
                ax.plot(d_tau["_step"], d_tau["mean"], lw=1.5, color=args.color_tau, label=LABEL_TAU if i==0 else "")
                if aggregating:
                    ax.fill_between(d_tau["_step"], d_tau["lo"], d_tau["hi"], color=args.color_tau, alpha=0.15, lw=0)

        # Plot LR
        d_lr = data_map.get(key_lr, pd.DataFrame())
        if not d_lr.empty:
            d_lr = d_lr[d_lr["_step"] <= X_MAX]
            if not d_lr.empty:
                ax.plot(d_lr["_step"], d_lr["mean"], lw=1.5, color=args.color_lr, label=LABEL_LR if i==0 else "")
                if aggregating:
                    ax.fill_between(d_lr["_step"], d_lr["lo"], d_lr["hi"], color=args.color_lr, alpha=0.15, lw=0)

        # Styling
        if args.out == "plots/relu_metrics_4x1.pdf": ax.set_title(pcfg["title"]) 
        ax.set_xlim(0, X_MAX)
        ax.set_yscale("log")
        ax.set_ylim(args.ymin, args.ymax) 
        
        # Vertical line for Task boundary
        ax.axvline(X_MIDPOINT, color='k', linestyle=':', linewidth=1.0)
        ax.grid(True, which="major", axis="y", alpha=0.2, ls="-")
        
        # X Ticks: Remove current ticks and replace with task labels
        ax.set_xticks([]) # Remove numerical ticks
        
        # Add 'Task 1' and 'Task 2' labels below the axis
        task1_x_pos = X_MIDPOINT / 2 # Center of Task 1 range (0 to X_MIDPOINT)
        task2_x_pos = X_MIDPOINT + (X_MIDPOINT / 2) # Center of Task 2 range (X_MIDPOINT to X_MAX)
        
        # Use ax.text for static labels
        if args.out == "plots/l2_metrics_4x1.pdf":
            ax.text(task1_x_pos, -0.05, 'Task 1', 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='top', fontsize=9)
            ax.text(task2_x_pos, -0.05, 'Task 2', 
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='top', fontsize=9)

        # Remove the generic X label, which is no longer needed
        # ax.set_xlabel('', labelpad=50)


    # Y labels only on first plot
    axs[0].set_ylabel("Value (log scale)")
    
    # Single global legend at the top
    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        # Move legend slightly higher to avoid conflict with task labels
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False) 

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Use rect to give more space for the Task 1/2 labels
    plt.savefig(args.out, bbox_inches="tight")
    print(f" -> Wrote plot to {args.out}")

if __name__ == "__main__":
    main()