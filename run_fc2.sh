#!/bin/bash
#SBATCH --job-name=fc2dead
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/gbaveja/scratch/fc2dead/logs/%x_%A_%a.out
set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled
ROOT=$HOME/scratch/fc2dead
line=$(( SLURM_ARRAY_TASK_ID + 1 ))
csv=$(sed -n "${line}p" "$ROOT/csvs.txt"); cmd=$(sed -n "${line}p" "$ROOT/commands.txt")
if [ -s "$csv" ]; then echo "skip done"; exit 0; fi
echo "RUN task $SLURM_ARRAY_TASK_ID -> $csv"
start=$(date +%s); bash -c "$cmd"; rc=$?; end=$(date +%s)
echo "CELLTIME task=$SLURM_ARRAY_TASK_ID rc=$rc elapsed_s=$(( end - start )) csv=$csv"
