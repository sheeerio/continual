"""Plot NFM/AGOP correlation scalars (feature_diag/<layer>/raw_corr, delta_corr)
logged to wandb during nfm_agop_sweep.sh, as mean +/- std across seeds vs. task.
"""
import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "sheerio"
PROJECT = "conference"
METRICS = ["raw_corr", "delta_corr"]
NAME_RE = re.compile(r"^(?P<algo>[A-Za-z0-9]+)_seed(?P<seed>\d+)$")
LAYER_COL_RE = re.compile(r"^feature_diag/(?P<layer>.+)/(?P<metric>raw_corr|delta_corr)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=str, default="nfm_agop_diag_t20")
    p.add_argument(
        "--algos",
        nargs="+",
        default=["vanilla", "l2", "spectral", "parseval", "wass"],
    )
    p.add_argument("--out_dir", type=str, default="plots/nfm_agop_correlation")
    return p.parse_args()


def fetch_data(group, algos):
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})

    # data[metric][layer][algo][seed] = {task: value}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for run in runs:
        m = NAME_RE.match(run.name)
        if not m or m.group("algo") not in algos:
            continue
        algo = m.group("algo")
        seed = int(m.group("seed"))

        history = run.history(samples=100000)
        if "feature_diag/task" not in history.columns:
            continue

        layer_cols = [c for c in history.columns if LAYER_COL_RE.match(c)]
        if not layer_cols:
            continue

        sub = history[["feature_diag/task"] + layer_cols].dropna(
            subset=["feature_diag/task"]
        )
        for _, row in sub.iterrows():
            task = int(row["feature_diag/task"])
            for col in layer_cols:
                value = row[col]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                col_m = LAYER_COL_RE.match(col)
                layer = col_m.group("layer")
                metric = col_m.group("metric")
                data[metric][layer][algo].setdefault(seed, {})[task] = float(value)

    return data


def plot_metric(data, metric, algos, out_dir):
    layers = sorted(data[metric].keys())
    if not layers:
        print(f"No data found for metric '{metric}'.")
        return

    colors = plt.cm.tab10.colors
    for layer in layers:
        plt.figure(figsize=(7, 5))
        for idx, algo in enumerate(algos):
            seed_runs = data[metric][layer].get(algo, {})
            if not seed_runs:
                continue
            common_tasks = sorted(
                set.intersection(*[set(v.keys()) for v in seed_runs.values()])
            )
            if not common_tasks:
                continue
            mat = np.array(
                [[seed_runs[s][t] for t in common_tasks] for s in sorted(seed_runs)]
            )
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            color = colors[idx % len(colors)]
            plt.plot(common_tasks, mean, label=algo, color=color, linewidth=2)
            plt.fill_between(
                common_tasks, mean - std, mean + std, color=color, alpha=0.2
            )

        plt.xlabel("Task")
        plt.ylabel(metric)
        plt.title(f"{layer}: NFM/AGOP {metric} (mean ± std across seeds)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(out_dir, f"{layer}_{metric}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = fetch_data(args.group, args.algos)
    for metric in METRICS:
        plot_metric(data, metric, args.algos, args.out_dir)


if __name__ == "__main__":
    main()
