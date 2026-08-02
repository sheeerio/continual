# MAIN SWEEP -- spectral only, extended coefficient axis, 3 seeds.
#
#   arm 1  static spectral                     --reg spectral --spectral_lambda C
#   arm 2  adaptive_sat, tau_ref_mode=median   (current default)
#   arm 3  adaptive_sat, tau_ref_mode=fixed    (drift-sensitive)
#   arm 4  vanilla reference                   (no coefficient)
#
# 8 coefficients x 3 arms x 3 seeds + 3 vanilla = 75 cells.
#
# --tau_update_interval 5, NOT 20: at N=20 the AUC std over seeds is 0.00360 vs
# 0.00191 at N=1, and this sweep has to resolve between-arm differences against
# a 0.00553 threshold. The extra 1.5x speedup is not worth doubling the noise
# floor.
#
# n_dead / n_alive / wnorm_dead / wnorm_alive land in the --taskdiag_csv per
# layer per task automatically; they cover fc1 and fc2, which are the only
# ReLU layers in this MLP (fc3 is commented out, fc4 is the linear output).
import os, shlex

H = os.environ["HOME"]
ROOT = f"{H}/scratch/sweep"
CD = f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"

SEEDS = (0, 1, 2)
COEFS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1"]

BASE = ["--optimizer", "adam", "--activation", "relu", "--runs", "20", "--epochs", "50",
        "--dataset", "MNIST", "--model", "MLP", "--hidden", "256", "--lr", "0.001",
        "--batch_size", "256", "--log_interval", "100", "--ns", "1.0",
        "--diagnostics", "full", "--track_coherence", "--coherence_window", "20",
        "--hessian_tol", "1e-2", "--hessian_max_iters", "50",
        "--loader_mode", "gpu", "--tau_update_interval", "5"]

AD = ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
      "--adaptive_scale", "saturating", "--sat_kappa", "1", "--tau_ref_window", "100"]

CELLS = []
for sd in SEEDS:
    CELLS.append((f"van_s{sd}", ["--reg", "none", "--seed", str(sd)]))
for c in COEFS:
    for sd in SEEDS:
        CELLS.append((f"stat_{c}_s{sd}",
                      ["--reg", "spectral", "--spectral_lambda", c, "--seed", str(sd)]))
for c in COEFS:
    for sd in SEEDS:
        CELLS.append((f"admed_{c}_s{sd}",
                      AD + ["--reg_sensitivity", c, "--tau_ref_mode", "median",
                            "--seed", str(sd)]))
for c in COEFS:
    for sd in SEEDS:
        CELLS.append((f"adfix_{c}_s{sd}",
                      AD + ["--reg_sensitivity", c, "--tau_ref_mode", "fixed",
                            "--seed", str(sd)]))

cmds = []
for nm, fl in CELLS:
    out = f"{CD}/{nm}.csv"
    cmds.append((out, [PY, "implicit_regularization.py", *BASE, *fl,
                       "--name", nm, "--exp_name", "sweep",
                       "--results_csv", out,
                       "--taskdiag_csv", f"{CD}/{nm}_taskdiag.csv",
                       "--diag_csv", f"{CD}/{nm}_diag.csv"]))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for o, c in cmds:
        fc.write(" ".join(shlex.quote(a) for a in c) + "\n")
        fv.write(o + "\n")

names = [n for n, _ in CELLS]
print(f"{len(cmds)} cells -> {ROOT}")
print(f"array range: 0-{len(cmds) - 1}")
print(f"  vanilla    {names.index('van_s0'):>3}-{names.index('van_s0')+2:<3}")
print(f"  static     {names.index('stat_1e-4_s0'):>3}-{names.index('stat_3e-1_s2'):<3}")
print(f"  ad median  {names.index('admed_1e-4_s0'):>3}-{names.index('admed_3e-1_s2'):<3}")
print(f"  ad fixed   {names.index('adfix_1e-4_s0'):>3}-{names.index('adfix_3e-1_s2'):<3}")
print("test pair (both at the static peak 1e-2, seed 0): "
      f"--array={names.index('stat_1e-2_s0')},{names.index('adfix_1e-2_s0')}"
      f"   [{names[names.index('stat_1e-2_s0')]}, {names[names.index('adfix_1e-2_s0')]}]")
