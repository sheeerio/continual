#!/bin/bash
#SBATCH --job-name=tauref
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/gbaveja/scratch/tauref/logs/%x_%A_%a.out

# Cells 4 and 5 carry COST_PROFILE=1 (Task 2a). The profiler inserts a
# cuda synchronize at each wrapped call, so those two cells are NOT valid
# wall-clock references for the others -- they exist to attribute the split,
# not to price a cell.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

ROOT=$HOME/scratch/tauref
line=$(( SLURM_ARRAY_TASK_ID + 1 ))
csv=$(sed -n "${line}p" "$ROOT/csvs.txt")
cmd=$(sed -n "${line}p" "$ROOT/commands.txt")

if [ -s "$csv" ]; then
  echo "skip done (task $SLURM_ARRAY_TASK_ID): $csv"
  exit 0
fi

if [ "$SLURM_ARRAY_TASK_ID" = "4" ] || [ "$SLURM_ARRAY_TASK_ID" = "5" ]; then
  export COST_PROFILE=1
  echo "COST_PROFILE enabled for task $SLURM_ARRAY_TASK_ID"
fi

echo "RUN task $SLURM_ARRAY_TASK_ID -> $csv"
start=$(date +%s)
bash -c "$cmd"
rc=$?
end=$(date +%s)
echo "CELLTIME task=$SLURM_ARRAY_TASK_ID rc=$rc elapsed_s=$(( end - start )) csv=$csv"
