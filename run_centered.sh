#!/bin/bash
#SBATCH --job-name=centered
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --array=0-71
#SBATCH --output=/home/gbaveja/scratch/testbed_centered/logs/%x_%A_%a.out

# TASK B3: 72 cells. Same line-indexed / resume-by-CSV pattern as
# run_testbed.sh and run_extend.sh. B2 measured 822 s for this config, so
# 01:00:00 leaves ~4x margin.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

ROOT=$HOME/scratch/testbed_centered
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
