#!/bin/bash
#SBATCH --job-name=calib
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --array=0-107
#SBATCH --output=/home/gbaveja/scratch/calib/logs/%x_%A_%a.out

# STEP 1 calibration harness. Same line-indexed pattern as
# run_baseline_sanity.sh: one array task per cell, resume by per-cell CSV
# existence. Array range is set deliberately -- 0-1 is the two-cell test,
# 0-107 is the full grid. Do not widen without re-reading gen_calib.py's
# printed count.
#
# Wall clock per cell is the thing this step measures, so each cell is timed
# and the elapsed seconds echoed into the log for the summary table.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

ROOT=$HOME/scratch/calib
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
echo "task $SLURM_ARRAY_TASK_ID done"
