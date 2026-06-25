#feature_diagnostics
import csv
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Subset


def _parse_layer_filter(spec):
    if spec is None:
        return None
    spec = spec.strip()
    if not spec or spec.lower() == "all":
        return None
    return {part.strip() for part in spec.split(",") if part.strip()}


def get_supported_layers(model, layer_filter="all"):
    requested = _parse_layer_filter(layer_filter)
    layers = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if requested is None or name in requested:
                layers[name] = module
    return layers


def capture_initial_linear_weights(model):
    weights = {}
    for name, module in get_supported_layers(model).items():
        weights[name] = module.weight.detach().clone()
    return weights


def should_run_feature_diagnostics(cfg, task_idx):
    if not getattr(cfg, "feature_diagnostics", False):
        return False
    every = max(1, int(getattr(cfg, "feature_diag_every", 1)))
    return (task_idx + 1) % every == 0


def _prepare_inputs(cfg, x, device):
    if cfg.model in ["CNN", "BatchNormCNN"]:
        return x.to(device).requires_grad_(True)
    return x.view(x.size(0), -1).to(device).requires_grad_(True)


def _sample_dataset(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _matrix_to_numpy(matrix):
    return matrix.detach().cpu().numpy().astype(np.float64, copy=False)


def _normalize_psd(matrix):
    trace = float(np.trace(matrix))
    if abs(trace) < 1e-12:
        return matrix.copy()
    return matrix / trace


def _pearson_corr(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _plot_task_matrices(task_dir, task_idx, layer_results):
    layer_names = list(layer_results.keys())
    if not layer_names:
        return None

    fig, axes = plt.subplots(
        len(layer_names),
        3,
        figsize=(12, 3.5 * len(layer_names)),
        squeeze=False,
    )

    col_titles = ["NFM: W^T W", "Delta NFM: (W-W0)^T (W-W0)", "AGOP"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title)

    for row_idx, layer_name in enumerate(layer_names):
        result = layer_results[layer_name]
        mats = [
            _normalize_psd(result["nfm"]),
            _normalize_psd(result["delta_nfm"]),
            _normalize_psd(result["agop"]),
        ]
        vmax = max(np.max(np.abs(mat)) for mat in mats)
        vmax = max(vmax, 1e-8)

        for col_idx, mat in enumerate(mats):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(
                    f"{layer_name}\nraw={result['raw_corr']:.3f}\ndelta={result['delta_corr']:.3f}",
                    rotation=0,
                    labelpad=45,
                    va="center",
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Task {task_idx + 1}: NFM vs AGOP", y=0.995)
    fig.tight_layout()
    out_path = os.path.join(task_dir, "nfm_agop.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_summary_csv(task_dir, layer_results):
    out_path = os.path.join(task_dir, "summary.csv")
    fieldnames = [
        "layer",
        "raw_corr",
        "delta_corr",
        "raw_diag_corr",
        "delta_diag_corr",
        "nfm_trace",
        "delta_nfm_trace",
        "agop_trace",
    ]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer_name, result in layer_results.items():
            writer.writerow(
                {
                    "layer": layer_name,
                    "raw_corr": result["raw_corr"],
                    "delta_corr": result["delta_corr"],
                    "raw_diag_corr": result["raw_diag_corr"],
                    "delta_diag_corr": result["delta_diag_corr"],
                    "nfm_trace": float(np.trace(result["nfm"])),
                    "delta_nfm_trace": float(np.trace(result["delta_nfm"])),
                    "agop_trace": float(np.trace(result["agop"])),
                }
            )
    return out_path


def _safe_name(name):
    return name.replace("/", "_").replace(".", "_")


def _collect_task_dirs(run_root):
    task_dirs = []
    if not os.path.isdir(run_root):
        return task_dirs
    for entry in sorted(os.listdir(run_root)):
        path = os.path.join(run_root, entry)
        if entry.startswith("task_") and os.path.isdir(path):
            task_dirs.append(path)
    return task_dirs


def _load_task_history(run_root, layer_name):
    history = []
    key_root = _safe_name(layer_name)
    for task_dir in _collect_task_dirs(run_root):
        matrix_path = os.path.join(task_dir, "matrices.npz")
        if not os.path.exists(matrix_path):
            continue
        payload = np.load(matrix_path)
        required = [
            f"{layer_name}__nfm",
            f"{layer_name}__delta_nfm",
            f"{layer_name}__agop",
        ]
        if not all(key in payload for key in required):
            continue
        task_name = os.path.basename(task_dir)
        history.append(
            {
                "task_name": task_name,
                "nfm": payload[f"{layer_name}__nfm"],
                "delta_nfm": payload[f"{layer_name}__delta_nfm"],
                "agop": payload[f"{layer_name}__agop"],
                "key_root": key_root,
            }
        )
    return history


def _plot_layer_task_trajectory(run_root, layer_name):
    history = _load_task_history(run_root, layer_name)
    if not history:
        return None

    trajectory_dir = os.path.join(run_root, "layer_trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)

    n_tasks = len(history)
    fig, axes = plt.subplots(
        3,
        n_tasks,
        figsize=(max(8, min(2.0 * n_tasks, 40)), 8),
        squeeze=False,
    )
    row_titles = ["NFM: W^T W", "Delta NFM", "AGOP"]
    keys = ["nfm", "delta_nfm", "agop"]

    for row_idx, row_title in enumerate(row_titles):
        axes[row_idx][0].set_ylabel(row_title)

    for col_idx, task_result in enumerate(history):
        matrices = [_normalize_psd(task_result[key]) for key in keys]
        vmax = max(np.max(np.abs(matrix)) for matrix in matrices)
        vmax = max(vmax, 1e-8)

        for row_idx, matrix in enumerate(matrices):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(task_result["task_name"].replace("task_", "T"))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{layer_name}: Representation Evolution Across Tasks", y=0.995)
    fig.tight_layout()
    out_path = os.path.join(
        trajectory_dir, f"{_safe_name(layer_name)}_task_evolution.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def update_task_trajectory_plots(run_root, layer_names):
    outputs = {}
    for layer_name in layer_names:
        out_path = _plot_layer_task_trajectory(run_root, layer_name)
        if out_path is not None:
            outputs[layer_name] = out_path
    return outputs


def run_feature_diagnostics(
    model,
    dataset,
    device,
    cfg,
    task_idx,
    init_weights,
    run_name,
):
    layers = get_supported_layers(model, getattr(cfg, "feature_diag_layers", "all"))
    if not layers:
        return {
            "skipped": True,
            "reason": "No supported linear layers matched the requested filter.",
        }

    sampled_dataset = _sample_dataset(
        dataset,
        getattr(cfg, "feature_diag_samples", 512),
        seed=int(getattr(cfg, "seed", 0)) + int(task_idx),
    )
    loader = data.DataLoader(
        sampled_dataset,
        batch_size=min(getattr(cfg, "feature_diag_batch_size", 64), len(sampled_dataset)),
        shuffle=False,
        num_workers=0,
    )

    agop_sums = {}
    for layer_name, module in layers.items():
        agop_sums[layer_name] = torch.zeros(
            module.in_features,
            module.in_features,
            dtype=torch.float64,
        )

    captured_inputs = {}
    hooks = []
    for layer_name, module in layers.items():
        hooks.append(
            module.register_forward_pre_hook(
                lambda mod, inputs, key=layer_name: captured_inputs.__setitem__(
                    key, inputs[0]
                )
            )
        )

    was_training = model.training
    model.eval()
    total_samples = 0

    try:
        for x, _ in loader:
            batch_inputs = _prepare_inputs(cfg, x, device)
            captured_inputs.clear()
            logits = model(batch_inputs)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            for layer_name in layers:
                h = captured_inputs[layer_name]
                if h.ndim != 2:
                    continue
                batch_agop = torch.zeros(
                    h.shape[1],
                    h.shape[1],
                    device=h.device,
                    dtype=torch.float64,
                )
                for out_idx in range(logits.shape[1]):
                    grad_h = torch.autograd.grad(
                        logits[:, out_idx].sum(),
                        h,
                        retain_graph=True,
                        allow_unused=False,
                    )[0]
                    batch_agop += grad_h.detach().to(torch.float64).T @ grad_h.detach().to(
                        torch.float64
                    )
                agop_sums[layer_name] += batch_agop.cpu()

            total_samples += int(batch_inputs.shape[0])
            model.zero_grad(set_to_none=True)
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()

    if total_samples == 0:
        return {"skipped": True, "reason": "Feature diagnostics received no samples."}

    base_dir = os.path.join(
        getattr(cfg, "feature_diag_dir", "plots/feature_diagnostics"),
        run_name,
        f"task_{task_idx + 1:03d}",
    )
    os.makedirs(base_dir, exist_ok=True)
    run_root = os.path.dirname(base_dir)

    layer_results = OrderedDict()
    npz_payload = {}
    for layer_name, module in layers.items():
        weight = module.weight.detach().view(module.weight.shape[0], -1)
        init_weight = init_weights.get(layer_name)
        if init_weight is None:
            init_weight = torch.zeros_like(weight)
        else:
            init_weight = init_weight.detach().to(weight.device).view_as(weight)

        nfm = _matrix_to_numpy(weight.T @ weight)
        delta = weight - init_weight
        delta_nfm = _matrix_to_numpy(delta.T @ delta)
        agop = (agop_sums[layer_name] / float(total_samples)).numpy()

        layer_results[layer_name] = {
            "nfm": nfm,
            "delta_nfm": delta_nfm,
            "agop": agop,
            "raw_corr": _pearson_corr(nfm, agop),
            "delta_corr": _pearson_corr(delta_nfm, agop),
            "raw_diag_corr": _pearson_corr(np.diag(nfm), np.diag(agop)),
            "delta_diag_corr": _pearson_corr(np.diag(delta_nfm), np.diag(agop)),
        }
        npz_payload[f"{layer_name}__nfm"] = nfm
        npz_payload[f"{layer_name}__delta_nfm"] = delta_nfm
        npz_payload[f"{layer_name}__agop"] = agop

    np.savez_compressed(os.path.join(base_dir, "matrices.npz"), **npz_payload)
    csv_path = _write_summary_csv(base_dir, layer_results)
    figure_path = _plot_task_matrices(base_dir, task_idx, layer_results)
    trajectory_paths = update_task_trajectory_plots(run_root, layer_results.keys())

    return {
        "skipped": False,
        "task_dir": base_dir,
        "run_root": run_root,
        "figure_path": figure_path,
        "trajectory_paths": trajectory_paths,
        "summary_csv": csv_path,
        "layers": list(layer_results.keys()),
        "summary": {
            layer_name: {
                "raw_corr": result["raw_corr"],
                "delta_corr": result["delta_corr"],
                "raw_diag_corr": result["raw_diag_corr"],
                "delta_diag_corr": result["delta_diag_corr"],
            }
            for layer_name, result in layer_results.items()
        },
    }
