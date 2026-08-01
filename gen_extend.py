# gen_extend.py -- TASK 1: extend the coefficient axis right until static
# spectral turns over.
#
# The 93-cell grid left static spectral still IMPROVING at the right edge
# (AUC 0.282 / 0.361 / 0.466 / 0.553 / 0.574 at c = 1e-4 .. 1e-2), so its
# optimum is at or past 1e-2 and is not on the plotted axis. No basin claim is
# interpretable until the peak AND the falloff are both visible.
#
# Config is byte-identical to the 2026-07-29 testbed grid except for the
# coefficient values, so these cells drop straight into the same table and the
# same P1 figure: 20 tasks x 50 epochs, 3 seeds, ns=1.0, MNIST, MLP hidden=256,
# Adam lr=1e-3, --diagnostics light, log_interval 100.
#
# Only static spectral and adaptive_sat -- inv and the scheduler arms are not
# needed to locate the peak, per scope.
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/testbed_extend"
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

# New right-hand coefficients only; 1e-4..1e-2 already exist in ~/scratch/testbed.
COEFFS = ["3e-2", "1e-1", "3e-1"]
SEEDS = [0, 1, 2]

ARMS = [
    ("A_static", lambda c: ["--reg", "spectral", "--spectral_lambda", c]),
    ("C_adaptive_sat", lambda c: ["--reg", "none", "--adaptive_reg",
                                  "--adaptive_type", "spectral",
                                  "--adaptive_scale", "saturating",
                                  "--reg_sensitivity", c]),
]

cmds = []
for arm, flags in ARMS:
    for c in COEFFS:
        for s in SEEDS:
            name = f"{arm}_c{c}_seed{s}"
            out = f"{CSVDIR}/{name}.csv"
            td = f"{CSVDIR}/{name}_taskdiag.csv"
            cmd = [PY, SCRIPT, *BASE, *flags(c), "--seed", str(s),
                   "--name", name, "--exp_name", "testbed_extend",
                   "--results_csv", out, "--taskdiag_csv", td]
            cmds.append((out, cmd))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for out, c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c) + "\n")
        fv.write(out + "\n")

done = sum(1 for o, _ in cmds if os.path.exists(o) and os.path.getsize(o) > 0)
print(f"{len(cmds)} cells ({len(ARMS)} arms x {len(COEFFS)} coeffs x {len(SEEDS)} seeds), "
      f"{done} done, {len(cmds)-done} to run -> {ROOT}")
print(f"array range: 0-{len(cmds)-1}")
