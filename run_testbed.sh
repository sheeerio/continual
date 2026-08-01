#!/bin/bash
#SBATCH --job-name=testbed
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --array=0-92
#SBATCH --output=/home/gbaveja/scratch/testbed/logs/%x_%A_%a.out

# STEP 3/4 harness. Same line-indexed pattern as run_calib.sh: one array task
# per cell, resume by per-cell CSV existence.
#
# --array=0-1 is the STEP 4 TIMING GATE and runs the two most expensive cells
# (see gen_testbed.py's GATE list). Do not widen to 0-92 until those two have
# reported and the projection has been checked.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

ROOT=$HOME/scratch/testbed
line=$(( SLURM_ARRAY_TASK_ID + 1 ))
csv=$(sed -n "${line}p" "$ROOT/csvs.txt")
cmd=$(sed -n "${line}p" "$ROOT/commands.txt")

if [ -s "$csv" ]; then
  echo "skip done (task $SLURM_ARRAY_TASK_ID): $csv"
  exit 0
fi

echo "RUN task $SLURM_ARRAY_TASK_ID -> $csv"
start=$(date +%s)
bash -c "$cmd"
rc=$?
end=$(date +%s)
echo "CELLTIME task=$SLURM_ARRAY_TASK_ID rc=$rc elapsed_s=$(( end - start )) csv=$csv"
