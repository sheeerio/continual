import csv
import itertools
import os
import subprocess
import sys

import yaml

PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "basin_sweep_protocol.yaml")


def load_protocol():
    with open(PROTOCOL_PATH) as f:
        return yaml.safe_load(f)


def completed_cells(results_csv):
    done = set()
    if not os.path.exists(results_csv):
        return done
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["regularizer"],
                row["model"],
                row["adaptive_multiplier"],
                row["reg_coeff"],
                row["seed"],
                row["dataset"],
            )
            done.add(key)
    return done


def cell_key(regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset):
    # Matches the string formatting csv.DictReader yields when re-reading rows
    # written by implicit_regularization.py's results_csv writer.
    return (
        regularizer,
        model,
        str(adaptive_multiplier),
        str(reg_coeff),
        str(seed),
        dataset,
    )


def run_name(regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset):
    mult_tag = "adaptive" if adaptive_multiplier else "fixed"
    return f"basin_{regularizer}_{model}_{dataset}_{mult_tag}_coeff{reg_coeff}_seed{seed}"


def build_command(protocol, regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset):
    fixed = protocol["fixed"]
    name = run_name(regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset)
    cmd = [
        sys.executable,
        protocol["script"],
        "--runs", str(fixed["runs"]),
        "--epochs", str(fixed["epochs"]),
        "--batch_size", str(fixed["batch_size"]),
        "--log_interval", str(fixed["log_interval"]),
        "--optimizer", fixed["optimizer"],
        "--exp_name", fixed["exp_name"],
        "--task_seed", str(fixed["task_seed"]),
        "--dataset", dataset,
        "--model", model,
        "--adaptive_reg",
        "--adaptive_scope", fixed["adaptive_scope"],
        "--adaptive_type", regularizer,
        "--reg_coeff", str(reg_coeff),
        "--seed", str(seed),
        "--name", name,
        "--results_csv", protocol["results_csv"],
    ]
    if adaptive_multiplier:
        cmd.append("--adaptive_multiplier")
    else:
        cmd.append("--no-adaptive_multiplier")
    return cmd


def main():
    protocol = load_protocol()
    axes = protocol["axes"]
    results_csv = protocol["results_csv"]

    all_cells = list(
        itertools.product(
            axes["regularizer"],
            axes["model"],
            axes["adaptive_multiplier"],
            axes["reg_coeff"],
            axes["seed"],
            axes["dataset"],
        )
    )
    done = completed_cells(results_csv)
    remaining = [
        cell
        for cell in all_cells
        if cell_key(*cell) not in done
    ]

    print(f"Total cells in protocol: {len(all_cells)}")
    print(f"Already completed (found in {results_csv}): {len(all_cells) - len(remaining)}")
    print(f"Remaining to run: {len(remaining)}")
    if len(sys.argv) > 1 and sys.argv[1] == "--count-only":
        return

    for i, (regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset) in enumerate(remaining):
        cmd = build_command(protocol, regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset)
        name = run_name(regularizer, model, adaptive_multiplier, reg_coeff, seed, dataset)
        print(f"[{i + 1}/{len(remaining)}] {name}")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
