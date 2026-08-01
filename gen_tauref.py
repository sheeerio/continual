# TASKS 1b/1c, 2a, 2b, 3a -- 8 cells on the calibrated testbed.
#   0-2  tau_ref_mode sweep (Task 1b/1c): median-100, median-2000, fixed
#   3    static re-run with the fixed line-940 estimator (Task 3a)
#   4    adaptive cell with COST_PROFILE (Task 2a split)
#   5    static  cell with COST_PROFILE (Task 2a split)
#   6-7  sigma2 subsample N=32, N=64 (Task 2b); N=256 baseline is cell 5
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/tauref"
CD = f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"

BASE = ["--optimizer", "adam", "--activation", "relu", "--runs", "20", "--epochs", "50",
        "--dataset", "MNIST", "--model", "MLP", "--hidden", "256", "--lr", "0.001",
        "--batch_size", "256", "--log_interval", "100", "--ns", "1.0",
        "--diagnostics", "full", "--track_coherence", "--coherence_window", "20",
        "--hessian_tol", "1e-2", "--hessian_max_iters", "50", "--seed", "0"]

ADAPT = ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
         "--adaptive_scale", "saturating", "--sat_kappa", "1", "--reg_sensitivity", "1e-3"]
STATIC = ["--reg", "spectral", "--spectral_lambda", "1e-3"]

CELLS = [
    ("tauref_median100",  ADAPT + ["--tau_ref_mode", "median", "--tau_ref_window", "100"]),
    ("tauref_median2000", ADAPT + ["--tau_ref_mode", "median", "--tau_ref_window", "2000"]),
    ("tauref_fixed",      ADAPT + ["--tau_ref_mode", "fixed"]),
    ("static_fixed_est",  STATIC),
    ("prof_adaptive",     ADAPT),
    ("prof_static",       STATIC),
    ("sigma2_n32",        STATIC + ["--sigma2_subsample", "32"]),
    ("sigma2_n64",        STATIC + ["--sigma2_subsample", "64"]),
]

cmds = []
for name, flags in CELLS:
    out = f"{CD}/{name}.csv"
    cmds.append((out, [PY, "implicit_regularization.py", *BASE, *flags,
                       "--name", name, "--exp_name", "tauref",
                       "--results_csv", out,
                       "--taskdiag_csv", f"{CD}/{name}_taskdiag.csv",
                       "--diag_csv", f"{CD}/{name}_diag.csv"]))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for o, c in cmds:
        fc.write(" ".join(shlex.quote(a) for a in c) + "\n")
        fv.write(o + "\n")
print(f"{len(cmds)} cells -> {ROOT}")
print(f"array range: 0-{len(cmds)-1}")
