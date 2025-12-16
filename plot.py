import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default=None)
parser.add_argument("--type", type=str, default="task", 
    choices=["normalize", "task", "idx0", "idx1", "idxm1", "range"])
config = parser.parse_args()


# --- Config ---
group_name = "cl"           
run_name = config.name   
metric_task = "task_acc"    
metric_snr1 = "snr_pct"     
metric_snr2 = "ly_snr_pct2" 
metric_snr3 = "k_snr_pct"

# --- Connect to W&B ---
api = wandb.Api()
runs = api.runs("sheerio/rand_label_MNIST")

# --- Find specific run ---
matches = [r for r in runs if r.group == group_name and r.name == run_name]
if not matches:
    raise ValueError(f"No run found with name '{run_name}' in group '{group_name}'")

target_run = sorted(matches, key=lambda r: r.created_at)[0]

# --- Pull metric history for all needed metrics ---
history_raw = target_run.history(keys=[metric_task, metric_snr1, metric_snr2, metric_snr3])
history_df = pd.DataFrame(history_raw)

# Ensure all required metrics exist
for metric in [metric_task, metric_snr1, metric_snr2, metric_snr3]:
    if metric not in history_df:
        raise ValueError(f"Metric '{metric}' not found in run '{run_name}'")

# Drop NaNs (aligned)
history_df = history_df.dropna(subset=[metric_task, metric_snr1, metric_snr2, metric_snr3])

# Convert to numpy arrays
task_acc = history_df[metric_task].to_numpy()
snr_pct = history_df[metric_snr1].to_numpy()
ly_snr_pct2 = history_df[metric_snr2].to_numpy()
k_snr_pct = history_df[metric_snr3].to_numpy()

if len(task_acc) < 2:
    raise ValueError(f"Need at least 2 values for scaling, found {len(task_acc)}")


min_task, max_task = task_acc.min(), task_acc.max()

def scale_to_task_range(metric, min_task, max_task):
    min_m, max_m = metric.min(), metric.max()
    if max_m == min_m:  # avoid division by zero
        return np.full_like(metric, (max_task + min_task) / 2)
    return ((metric - min_m) / (max_m - min_m)) * (max_task - min_task) + min_task

def normalize(arr):
    min_val, max_val = arr.min(), arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

if config.type == "normalize":
    task_acc_scaled      = normalize(task_acc)
    snr_pct_scaled       = normalize(snr_pct)
    ly_snr_pct2_scaled   = normalize(ly_snr_pct2)
    k_snr_pct_scaled     = normalize(k_snr_pct)
elif config.type == "idx1":
    scale_snr_pct = task_acc[1] / snr_pct[1] #/ task_acc[1]
    scale_ly_snr_pct2 = task_acc[1] / ly_snr_pct2[1]#/ task_acc[1]
    scale_k_snr_pct = task_acc[1] / k_snr_pct[1] #/ task_acc[1]
elif config.type == "idx0":
    scale_snr_pct = task_acc[0] / snr_pct[0] #/ task_acc[1]
    scale_ly_snr_pct2 = task_acc[0] / ly_snr_pct2[0]#/ task_acc[1]
    scale_k_snr_pct = task_acc[0] / k_snr_pct[0] #/ task_acc[1]
elif config.type == "idxm1":
    scale_snr_pct = task_acc[-1] / snr_pct[-1] #/ task_acc[1]
    scale_ly_snr_pct2 = task_acc[-1] / ly_snr_pct2[-1]#/ task_acc[1]
    scale_k_snr_pct = task_acc[-1] / k_snr_pct[-1] #/ task_acc[1]
elif config.type == "task":
    snr_pct_scaled = scale_to_task_range(snr_pct, min_task, max_task)
    ly_snr_pct2_scaled = scale_to_task_range(ly_snr_pct2, min_task, max_task)
    k_snr_pct_scaled = scale_to_task_range(k_snr_pct, min_task, max_task)
elif config.type == "range":
    snr_pct_scaled = (snr_pct - snr_pct.min()) / (snr_pct.max() - snr_pct.min())
    ly_snr_pct2_scaled = (ly_snr_pct2 - ly_snr_pct2.min()) / (ly_snr_pct2.max() - ly_snr_pct2.min())
    task_acc_scaled = (task_acc - task_acc.min()) / (task_acc.max() - task_acc.min())

if config.type in ["idx0", "idx1", "idxm1"]:
    snr_pct_scaled = snr_pct * scale_snr_pct
    ly_snr_pct2_scaled = ly_snr_pct2 * scale_ly_snr_pct2
    k_snr_pct_scaled = k_snr_pct * scale_k_snr_pct
    # snr_pct_scaled[0]=task_acc[0]
    # ly_snr_pct2_scaled[0]=task_acc[0] 
    # k_snr_pct_scaled[0]= task_acc[0]


# --- Plot ---
plt.figure(figsize=(8, 4))
if config.type == "normalize":
    plt.plot(task_acc_scaled, label=f"actual {metric_task}", linewidth=2)
else:
    plt.plot(task_acc, label=f"actual {metric_task}", linewidth=2) 
plt.plot(snr_pct_scaled, label=f"{metric_snr1} (scaled)", linestyle='--', color='orange')
plt.plot(ly_snr_pct2_scaled, label=f"{metric_snr2} (scaled)", linestyle='--', color='green')
plt.plot(k_snr_pct_scaled, label=f"{metric_snr3} (scaled)", linestyle='--', color='red')

if config.type == "normalize":
    plt.title(f"{metric_task}, {metric_snr1}, {metric_snr2}, {metric_snr3} (all scaled)\nRun '{run_name}'")
else:
    plt.title(f"{metric_task}, {metric_snr1} (scaled), {metric_snr2} (scaled), {metric_snr3} (scaled)\nRun '{run_name}'")

plt.xlabel(f"Predicted {metric_task}")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# Save + show
plt.savefig(f"plots/{config.type}/{run_name}_{group_name}_metrics_scaled_range.png", dpi=300, bbox_inches="tight")
plt.show()