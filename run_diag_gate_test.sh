#!/bin/bash
#SBATCH --job-name=diag_gate
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/home/gbaveja/scratch/diag_gate_test/logs/%x_%A_%a.out

# Integration test for the diagnostic-block gating fix. Same short run
# twice: array index 0 = old behavior (diag_interval == log_interval,
# i.e. unthrottled), index 1 = throttled (diag_interval 8x log_interval).
# Same seed, same everything else -- compares wall-clock and confirms no
# crash either way.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

COMMON=(--optimizer adam --activation relu --runs 3 --epochs 50
        --dataset MNIST --model MLP --hidden 64 --lr 0.001 --batch_size 256
        --log_interval 50 --ns 1.0 --track_coherence
        --reg l2_loss --l2_lambda 1e-3 --seed 0
        --diag_csv "/home/gbaveja/scratch/diag_gate_test/diag_${SLURM_ARRAY_TASK_ID}.csv")

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
  NAME="unthrottled"
  EXTRA=(--diag_interval 50 --diag_task_interval 1)
else
  NAME="throttled"
  EXTRA=(--diag_interval 400 --diag_task_interval 3)
fi

START=$(date +%s)
python implicit_regularization.py "${COMMON[@]}" "${EXTRA[@]}" \
  --name "${NAME}" --exp_name diag_gate_test
END=$(date +%s)
echo "RESULT ${NAME} elapsed=$(( END - START ))s"
