#!/bin/bash
#SBATCH --job-name=unit
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --array=0-12
#SBATCH --output=/home/gbaveja/scratch/unit/logs/%x_%A_%a.out

# Cells 0-2 print the runtime hook-attribution check (Task 1c) at startup.
# Cell 12 is launched under cProfile by the generator, not by this script.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

ROOT=$HOME/scratch/unit
line=$(( SLURM_ARRAY_TASK_ID + 1 ))
csv=$(sed -n "${line}p" "$ROOT/csvs.txt")
cmd=$(sed -n "${line}p" "$ROOT/commands.txt")

if [ -s "$csv" ]; then
  echo "skip done (task $SLURM_ARRAY_TASK_ID): $csv"
  exit 0
fi

case $SLURM_ARRAY_TASK_ID in
  0|1|2) export HOOK_CHECK=1; echo "HOOK_CHECK on" ;;
esac

echo "RUN task $SLURM_ARRAY_TASK_ID -> $csv"
start=$(date +%s)
bash -c "$cmd"
rc=$?
end=$(date +%s)
echo "CELLTIME task=$SLURM_ARRAY_TASK_ID rc=$rc elapsed_s=$(( end - start )) csv=$csv"
