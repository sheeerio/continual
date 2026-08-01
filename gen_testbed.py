# gen_testbed.py -- STEP 3: spectral-only fast-iteration testbed.
#
# Cell config is the Step 1 recommendation: t20_e50 (20 tasks x 50 epochs),
# which preserved the reference AUC ordering exactly (rho=+1.000) with seed
# noise at 0.73x the 10%-resolution budget, at 9.2 min/cell.
#
# NOTE: the cell is validated for AUC ONLY. The reference run supplied AUCs
# and no slopes, so slope was never compared against it -- not "compared and
# found wanting", simply not checkable. Slope is recorded here per spec, but
# any slope-based claim needs its own validation at 100 tasks first.
#
# Four arms for contribution attribution, on a shared log-spaced coefficient
# axis so A/B/C/D are directly comparable at matched c:
#   A  static spectral
#   B  A + pl_lyapunov scheduler
#   C  adaptive spectral            (inv and saturating as two sub-arms)
#   D  C + pl_lyapunov scheduler    (inv and saturating as two sub-arms)
# plus a vanilla reference line (no coefficient axis).
#
# CRITICAL: --reg l2 is never used anywhere -- it routes to Adam's decoupled
# weight_decay, the known-bad path. Static spectral is --reg spectral; the
# adaptive arms carry --reg none so the static branch stays off.
#
# A vs C is like-for-like by the mathematical-equivalence route (see the
# baseline-rigor rule in CLAUDE.md): identical parameter coverage (layer_map
# is built from named_parameters(); the MLP's fc3 is commented out), identical
# penalty form and estimator (power_iteration(p, iters=1), same spectral_k),
# and a global lambda on the summed penalty equals a constant per-layer
# factor. Accepted caveat: the two paths consume different amounts of RNG, so
# matched-c arms are a seed offset apart, absorbed by 3-seed averaging.
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/testbed"
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

SCHED = ["--lr_schedule", "pl_lyapunov", "--param", "t"]
COEFFS = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
SEEDS = [0, 1, 2]


def arm_flags(arm, c):
    if arm == "A_static":
        return ["--reg", "spectral", "--spectral_lambda", c]
    if arm == "B_static_sched":
        return ["--reg", "spectral", "--spectral_lambda", c] + SCHED
    if arm == "C_adaptive_inv":
        return ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
                "--adaptive_scale", "inv", "--reg_sensitivity", c]
    if arm == "C_adaptive_sat":
        return ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
                "--adaptive_scale", "saturating", "--reg_sensitivity", c]
    if arm == "D_adaptive_inv_sched":
        return ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
                "--adaptive_scale", "inv", "--reg_sensitivity", c] + SCHED
    if arm == "D_adaptive_sat_sched":
        return ["--reg", "none", "--adaptive_reg", "--adaptive_type", "spectral",
                "--adaptive_scale", "saturating", "--reg_sensitivity", c] + SCHED
    raise ValueError(arm)


ARMS = ["A_static", "B_static_sched", "C_adaptive_inv", "C_adaptive_sat",
        "D_adaptive_inv_sched", "D_adaptive_sat_sched"]

cells = []          # (name, flags)
for arm in ARMS:
    for c in COEFFS:
        for s in SEEDS:
            cells.append((f"{arm}_c{c}_seed{s}", arm_flags(arm, c) + ["--seed", str(s)]))
for s in SEEDS:
    cells.append((f"vanilla_seed{s}", ["--reg", "none", "--seed", str(s)]))

# STEP 4 timing gate goes first: array indices 0 and 1 are the two most
# expensive / least-certain cells (adaptive+scheduler, and adaptive alone) at
# the mid coefficient. Running --array=0-1 measures the corner that dominates
# the projection instead of a cheap cell that would flatter it.
GATE = ["D_adaptive_sat_sched_c1e-3_seed0", "C_adaptive_inv_c1e-3_seed0"]
cells.sort(key=lambda kv: (kv[0] not in GATE, GATE.index(kv[0]) if kv[0] in GATE else 0))

cmds = []
for name, flags in cells:
    out = f"{CSVDIR}/{name}.csv"
    td = f"{CSVDIR}/{name}_taskdiag.csv"
    cmd = [PY, SCRIPT, *BASE, *flags, "--name", name, "--exp_name", "testbed",
           "--results_csv", out, "--taskdiag_csv", td]
    cmds.append((out, cmd))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for out, c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c) + "\n")
        fv.write(out + "\n")

done = sum(1 for o, _ in cmds if os.path.exists(o) and os.path.getsize(o) > 0)
print(f"{len(cmds)} cells ({len(ARMS)} arms x {len(COEFFS)} coeffs x {len(SEEDS)} seeds "
      f"+ {len(SEEDS)} vanilla), {done} done, {len(cmds)-done} to run -> {ROOT}")
print(f"array range: 0-{len(cmds)-1}")
print(f"gate cells (--array=0-1): {cmds[0][0].split('/')[-1]}, {cmds[1][0].split('/')[-1]}")
