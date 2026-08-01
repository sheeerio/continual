#!/bin/bash
#SBATCH --job-name=basesanity
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --array=0-32
#SBATCH --output=/home/gbaveja/scratch/baseline_sanity/logs/%x_%A_%a.out

# Baseline sanity run: one coefficient per method, 3 seeds, 100 tasks,
# ns=1.0 MNIST, MLP hidden=256. One array task per cell (33 total) -- small
# run, no need for the CHUNK/P multiplexing used by the big sweeps.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

ROOT=$HOME/scratch/baseline_sanity
line=$(( SLURM_ARRAY_TASK_ID + 1 ))
csv=$(sed -n "${line}p" "$ROOT/csvs.txt")
cmd=$(sed -n "${line}p" "$ROOT/commands.txt")

if [ -s "$csv" ]; then
  echo "skip done (task $SLURM_ARRAY_TASK_ID): $csv"
  exit 0
fi

echo "RUN task $SLURM_ARRAY_TASK_ID -> $csv"
bash -c "$cmd"
echo "task $SLURM_ARRAY_TASK_ID done"
