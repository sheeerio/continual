# 1c/1d/1e re-run + TASK 2 tau_update_interval sweep + TASK 3 gpu residual.
#
#   0-2   fc2 arms with per-unit logging, HOOK_CHECK on      (1c/1d/1e)
#   3-11  --tau_update_interval N in {1,5,20} x seeds {0,1,2} (Task 2)
#   12    gpu-mode static under cProfile                      (Task 3)
#
# All cells use --loader_mode gpu (351 s/static cell vs 684-754 s workers4;
# AUC mean 0.46482 vs 0.46568, inside the 0.00553 seed std).
import os, shlex

H = os.environ["HOME"]
ROOT = f"{H}/scratch/unit"
CD = f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"

BASE = ["--optimizer", "adam", "--activation", "relu", "--runs", "20", "--epochs", "50",
        "--dataset", "MNIST", "--model", "MLP", "--hidden", "256", "--lr", "0.001",
        "--batch_size", "256", "--log_interval", "100", "--ns", "1.0",
        "--diagnostics", "full", "--track_coherence", "--coherence_window", "20",
        "--hessian_tol", "1e-2", "--hessian_max_iters", "50", "--loader_mode", "gpu"]

ST = ["--reg", "spectral", "--spectral_lambda", "1e-3"]
AD = ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
      "--adaptive_scale", "saturating", "--sat_kappa", "1", "--reg_sensitivity", "1e-3",
      "--tau_ref_mode", "median", "--tau_ref_window", "100"]

CELLS = [("u_vanilla",  ["--reg", "none", "--seed", "0"]),
         ("u_static",   ST + ["--seed", "0"]),
         ("u_adaptive", AD + ["--seed", "0"])]
for N in (1, 5, 20):
    for sd in (0, 1, 2):
        CELLS.append((f"tui_N{N}_s{sd}",
                      AD + ["--tau_update_interval", str(N), "--seed", str(sd)]))
CELLS.append(("prof_gpu_static", ST + ["--seed", "0"]))

cmds = []
for i, (nm, fl) in enumerate(CELLS):
    out = f"{CD}/{nm}.csv"
    # Cell 12 runs under cProfile; building it here keeps the sbatch script
    # free of fragile path substitution.
    launcher = [PY, "-m", "cProfile", "-o", f"{CD}/gpu_static.pstats"] \
        if nm == "prof_gpu_static" else [PY]
    cmds.append((out, launcher + ["implicit_regularization.py", *BASE, *fl,
                                  "--name", nm, "--exp_name", "unit",
                                  "--results_csv", out,
                                  "--taskdiag_csv", f"{CD}/{nm}_taskdiag.csv",
                                  "--diag_csv", f"{CD}/{nm}_diag.csv"]))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for o, c in cmds:
        fc.write(" ".join(shlex.quote(a) for a in c) + "\n")
        fv.write(o + "\n")
print(f"{len(cmds)} cells -> {ROOT}")
print("array range: 0-%d" % (len(cmds) - 1))
