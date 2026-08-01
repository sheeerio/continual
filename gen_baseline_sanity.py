# gen_baseline_sanity.py
# Baseline sanity run: one coefficient per method, 3 seeds, 100 tasks,
# random-label MNIST (ns=1.0), MLP hidden=256. NOT a sweep, NOT a basin --
# confirms every vetted baseline moves in the expected direction before any
# of it goes into the coefficient-axis harness.
import os, shlex
H = os.environ["HOME"]
ROOT = f"{H}/scratch/baseline_sanity"
CSVDIR = f"{ROOT}/cells"
os.makedirs(CSVDIR, exist_ok=True)
PY = f"{H}/venv/continual/bin/python"
SCRIPT = "implicit_regularization.py"

BASE = ["--optimizer", "adam", "--activation", "relu", "--runs", "100",
        "--epochs", "100", "--dataset", "MNIST", "--model", "MLP",
        "--hidden", "256", "--lr", "0.001", "--batch_size", "256",
        "--log_interval", "400", "--ns", "1.0"]

SEEDS = [0, 1, 2]

# (arm_name, extra_flags) -- extra_flags override/extend BASE per-arm.
# CRITICAL: static L2 arms use --reg l2_loss (honest loss-based path), never
# --reg l2 (Adam decoupled weight_decay -- known-bad, puts static L2 at
# chance). shrink_perturb uses its own sp_weight_decay/sp_noise_std scale,
# not the 1e-3 lambda used by every other regularizer. cbp has no lambda.
ARMS = [
    ("vanilla", ["--reg", "none"]),
    ("l2_loss", ["--reg", "l2_loss", "--l2_lambda", "1e-3"]),
    ("l2_init", ["--reg", "l2_init", "--l2_lambda", "1e-3"]),
    ("spectral", ["--reg", "spectral", "--spectral_lambda", "1e-3"]),
    ("wass", ["--reg", "wass", "--wass_lambda", "1e-3"]),
    ("ortho", ["--reg", "ortho", "--ortho_lambda", "1e-3"]),
    ("parseval", ["--reg", "parseval", "--parseval_lambda", "1e-3"]),
    ("shrink_perturb", ["--reg", "shrink_perturb",
                         "--sp_weight_decay", "0.4", "--sp_noise_std", "0.01"]),
    ("layernorm_l2", ["--model", "LayerNormMLP", "--reg", "l2_loss", "--l2_lambda", "1e-3"]),
    ("cbp", ["--reg", "none", "--use_cbp", "--cbp_replacement_rate", "1e-4"]),
    ("adaptive_sat", ["--reg", "none", "--l2_lambda", "0", "--adaptive_reg",
                       "--adaptive_type", "l2", "--adaptive_scale", "saturating",
                       "--reg_sensitivity", "1e-3"]),
]

cmds = []
for arm, flags in ARMS:
    for seed in SEEDS:
        name = f"{arm}_seed{seed}"
        out = f"{CSVDIR}/{name}.csv"
        cmd = [PY, SCRIPT, *BASE, *flags, "--seed", str(seed), "--name", name,
               "--exp_name", "baseline_sanity", "--results_csv", out]
        cmds.append((out, cmd))

with open(f"{ROOT}/commands.txt", "w") as fc, open(f"{ROOT}/csvs.txt", "w") as fv:
    for out, c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c) + "\n")
        fv.write(out + "\n")

done = sum(1 for o, _ in cmds if os.path.exists(o) and os.path.getsize(o) > 0)
print(f"{len(cmds)} cells ({len(ARMS)} arms x {len(SEEDS)} seeds), {done} already done, {len(cmds)-done} to run -> {ROOT}")
