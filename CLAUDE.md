# Cluster conventions (Narval / Alliance)

- Always use --account=def-schmidtm for test/dev jobs, --account=rrg-schmidtm for priority sweeps.
- Never guess partition names or time limits — check `sinfo` or ask before submitting.
- Before any `sbatch`, run a two-cell test submission (--array=0-1) first. Never launch a full array untouched.
- Compute nodes have no internet: always set WANDB_MODE=offline in submission scripts, sync with `wandb sync` from the login node after.
- Use absolute venv python paths in SLURM scripts (~/venv/continual/bin/python), never rely on `source activate`.
- Resume logic: check per-cell CSV existence before running; skip if already done.
- Before submitting, show me the full sbatch script and the exact `sbatch` command — don't run sbatch without me reviewing it first.
- To check job status use `squeue -u gbaveja`; don't guess job IDs, pull them from the sbatch output or squeue.
