# gen_calib.py -- STEP 1: calibrate the cheap config for the fast-iteration testbed.
#
# Question: what is the cheapest (tasks, epochs) cell that still PRESERVES THE
# ORDERING of {spectral, l2_loss, wass, vanilla} from the completed 100-task
# reference run? Those four span the AUC range (0.742 / 0.625 / 0.606 / 0.117),
# so an ordering that survives on them is evidence the cheap cell still
# separates methods.
#
# --diagnostics off throughout: diagnostics are purely observational and do not
# change the trained model, so they cannot affect AUC -- paying for them here
# would only distort the wall-clock numbers this step exists to measure.
# None of these four arms uses pl_lyapunov, so nothing method-critical is gated
# off (see the DIAG_NEED_SCHED note in implicit_regularization.py).
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/calib"
CSVDIR = f"{ROOT}/cells"
os.makedirs(CSVDIR, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"
SCRIPT = "implicit_regularization.py"

# Matches the reference run exactly except for runs/epochs, which are the axes
# under test. Same ns=1.0 MNIST subset, same MLP hidden=256, same Adam lr=1e-3.
BASE = ["--optimizer", "adam", "--activation", "relu",
        "--dataset", "MNIST", "--model", "MLP",
        "--hidden", "256", "--lr", "0.001", "--batch_size", "256",
        "--log_interval", "400", "--ns", "1.0",
        "--diagnostics", "off"]

SEEDS = [0, 1, 2]

# Same coefficient (1e-3) as the reference run, so AUCs are comparable.
# --reg l2 is NEVER used: it routes to Adam decoupled weight_decay, the
# known-bad path. Static L2 is --reg l2_loss.
ARMS = [
    ("spectral", ["--reg", "spectral", "--spectral_lambda", "1e-3"]),
    ("l2_loss",  ["--reg", "l2_loss",  "--l2_lambda", "1e-3"]),
    ("wass",     ["--reg", "wass",     "--wass_lambda", "1e-3"]),
    ("vanilla",  ["--reg", "none"]),
]

# tasks: 100 -> 50, 30, 20   x   epochs/task: 100 -> 50, 25, 10
GRID = [(t, e) for t in (50, 30, 20) for e in (50, 25, 10)]

cmds = []
for tasks, epochs in GRID:
    for arm, flags in ARMS:
        for seed in SEEDS:
            name = f"t{tasks}_e{epochs}_{arm}_seed{seed}"
            out = f"{CSVDIR}/{name}.csv"
            cmd = [PY, SCRIPT, *BASE,
                   "--runs", str(tasks), "--epochs", str(epochs),
                   *flags, "--seed", str(seed), "--name", name,
                   "--exp_name", "calib", "--results_csv", out]
            cmds.append((out, cmd))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for out, c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c) + "\n")
        fv.write(out + "\n")

done = sum(1 for o, _ in cmds if os.path.exists(o) and os.path.getsize(o) > 0)
print(f"{len(cmds)} cells ({len(GRID)} grid x {len(ARMS)} arms x {len(SEEDS)} seeds), "
      f"{done} already done, {len(cmds)-done} to run -> {ROOT}")
print(f"array range: 0-{len(cmds)-1}")
