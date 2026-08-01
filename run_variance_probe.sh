#!/bin/bash
#SBATCH --job-name=varprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/gbaveja/scratch/variance_probe/logs/%x_%j.out

# TASK 1: estimator-noise vs real-curvature decomposition for tau.
# epochs=3 (not 50): the probes are checkpoint-based, so the run only needs to
# reach mid-task-1 and mid-task-2, not train to convergence.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

mkdir -p "$HOME/scratch/variance_probe/logs"

start=$(date +%s)
~/venv/continual/bin/python variance_probe.py \
  --optimizer adam --activation relu --runs 3 --epochs 3 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 10 --ns 1.0 --seed 0 \
  --name variance_probe --exp_name variance_probe
rc=$?
end=$(date +%s)
echo "CELLTIME rc=$rc elapsed_s=$(( end - start ))"
