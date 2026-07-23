#!/bin/bash
#SBATCH --job-name=cbp_permuted
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/gbaveja/scratch/cbp_permuted/logs/%x_%A_%a.out

# Vanilla vs Continual Backprop on PermutedMNIST, 500 tasks, 1 epoch/task.
# One comparison baseline (AUC/slope vs vanilla), not a coefficient sweep --
# array index 0 = vanilla, 1 = CBP. Same seed, same everything else.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

ROOT=$HOME/scratch/cbp_permuted
SCRIPT="implicit_regularization.py"

COMMON=(
  --dataset PermutedMNIST
  --runs 500
  --epochs 1
  --model MLP
  --activation relu
  --seed 0
  --reg none
  --results_csv "${ROOT}/results.csv"
)

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  NAME="vanilla"
  EXTRA_FLAGS=()
else
  NAME="cbp"
  EXTRA_FLAGS=(--use_cbp --cbp_replacement_rate 1e-4 --cbp_maturity_threshold 100 --cbp_decay_rate 0.99)
fi

echo "RUN ${NAME}"
python "${SCRIPT}" \
  "${COMMON[@]}" \
  --name "${NAME}" \
  --exp_name "cbp_permuted_500tasks" \
  "${EXTRA_FLAGS[@]}" \
  > "${ROOT}/logs/${NAME}.log" 2>&1

echo "${NAME} done"
