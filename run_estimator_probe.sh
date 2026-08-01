#!/bin/bash
#SBATCH --job-name=estprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/gbaveja/scratch/estimator_probe/logs/%x_%j.out

# TASKS 1-3 in one run on the CALIBRATED config (50 epochs/task, ~2100
# steps/task) so the >=50-logged-step window fits inside a single task --
# the defect that limited the previous probe to 6 samples.
#
# 3h limit: the warm-chain windows carry 30 chains x (1+2+5) iters x 3 layers
# = 720 HVPs per step for 60 steps before each of 3 checkpoints, on top of
# the per-checkpoint 30-repeat fixed-point and batch-spread measurements.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

mkdir -p "$HOME/scratch/estimator_probe/logs"

start=$(date +%s)
~/venv/continual/bin/python estimator_probe.py \
  --optimizer adam --activation relu --runs 3 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 10 --ns 1.0 --seed 0 \
  --reg spectral --spectral_lambda 1e-3 \
  --name estimator_probe --exp_name estimator_probe
rc=$?
end=$(date +%s)
echo "CELLTIME rc=$rc elapsed_s=$(( end - start ))"
