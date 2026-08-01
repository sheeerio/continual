# gen_centered.py -- TASK B3: --adaptive_scale centered swept over sat_kappa
# and the FULL coefficient axis.
#
# Full axis (both limbs), not a bracket around static's peak: the question is
# whether kappa changes the basin's SHAPE, and shape needs both limbs. A
# 3-point bracket could only show the peak moving, which is exactly the
# confound (level-shift vs reshape) this arm exists to remove.
#
# Config identical to the 93-cell grid and the Task 1 extension so all three
# drop into one figure: 20 tasks x 50 epochs, 3 seeds, ns=1.0, MNIST,
# MLP hidden=256, Adam lr=1e-3, --diagnostics light, log_interval 100.
#
# Fresh output dir: the results schema gained centered_clamp_* columns, and
# results_csv headers are written from the first row's keys, so these cells
# must not be appended into the older grids' CSVs. Merging happens at analysis
# time by reading each directory separately.
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/testbed_centered"
CSVDIR = f"{ROOT}/cells"
os.makedirs(CSVDIR, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"
SCRIPT = "implicit_regularization.py"

BASE = ["--optimizer", "adam", "--activation", "relu",
        "--runs", "20", "--epochs", "50",
        "--dataset", "MNIST", "--model", "MLP",
        "--hidden", "256", "--lr", "0.001", "--batch_size", "256",
        "--log_interval", "100", "--ns", "1.0",
        "--diagnostics", "light", "--track_coherence", "--coherence_window", "20"]

COEFFS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1"]
KAPPAS = ["1", "5", "20"]
SEEDS = [0, 1, 2]

cmds = []
for k in KAPPAS:
    for c in COEFFS:
        for s in SEEDS:
            name = f"centered_k{k}_c{c}_seed{s}"
            out = f"{CSVDIR}/{name}.csv"
            td = f"{CSVDIR}/{name}_taskdiag.csv"
            cmd = [PY, SCRIPT, *BASE,
                   "--reg", "none", "--adaptive_reg",
                   "--adaptive_type", "spectral",
                   "--adaptive_scale", "centered",
                   "--sat_kappa", k,
                   "--reg_sensitivity", c,
                   "--seed", str(s), "--name", name, "--exp_name", "testbed_centered",
                   "--results_csv", out, "--taskdiag_csv", td]
            cmds.append((out, cmd))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for out, c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c) + "\n")
        fv.write(out + "\n")

done = sum(1 for o, _ in cmds if os.path.exists(o) and os.path.getsize(o) > 0)
print(f"{len(cmds)} cells ({len(KAPPAS)} kappas x {len(COEFFS)} coeffs x {len(SEEDS)} seeds), "
      f"{done} done, {len(cmds)-done} to run -> {ROOT}")
print(f"array range: 0-{len(cmds)-1}")
