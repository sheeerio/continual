# v1.1 verification: ONE cell, byte-identical flags to unit/tui_N5_s0 (which ran
# on the pre-fix working tree, git_sha 449dcc0-dirty). Compares task_acc_traj
# against that stored row -- a true A/B with no extra GPU time for the baseline.
#
# Both fixes are expected to be exact no-ops on every logged number:
#   (a) --make_plots default off: skips a wandb.log that is already a no-op
#       under WANDB_MODE=disabled.
#   (b) gpu eval loader: batch sequence and RNG consumption verified identical
#       on CPU (check_evalloader.py, check_rng.py).
import os, shlex

H = os.environ["HOME"]
ROOT = f"{H}/scratch/v11"
CD = f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"

FLAGS = ["--optimizer", "adam", "--activation", "relu", "--runs", "20", "--epochs", "50",
         "--dataset", "MNIST", "--model", "MLP", "--hidden", "256", "--lr", "0.001",
         "--batch_size", "256", "--log_interval", "100", "--ns", "1.0",
         "--diagnostics", "full", "--track_coherence", "--coherence_window", "20",
         "--hessian_tol", "1e-2", "--hessian_max_iters", "50", "--loader_mode", "gpu",
         "--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
         "--adaptive_scale", "saturating", "--sat_kappa", "1", "--reg_sensitivity", "1e-3",
         "--tau_ref_mode", "median", "--tau_ref_window", "100",
         "--tau_update_interval", "5", "--seed", "0"]

nm = "v11_tui_N5_s0"
out = f"{CD}/{nm}.csv"
cmd = [PY, "implicit_regularization.py", *FLAGS, "--name", nm, "--exp_name", "v11",
       "--results_csv", out,
       "--taskdiag_csv", f"{CD}/{nm}_taskdiag.csv",
       "--diag_csv", f"{CD}/{nm}_diag.csv"]

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    fc.write(" ".join(shlex.quote(a) for a in cmd) + "\n")
    fv.write(out + "\n")
print(f"1 cell -> {ROOT}")
print("array range: 0-0")
