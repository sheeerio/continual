#!/bin/bash
#SBATCH --job-name=specprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=03:00:00
#SBATCH --output=/home/gbaveja/scratch/spectrum_probe/logs/%x_%j.out

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled
mkdir -p "$HOME/scratch/spectrum_probe/logs"

start=$(date +%s)
~/venv/continual/bin/python spectrum_probe.py \
  --optimizer adam --activation relu --runs 3 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 10 --ns 1.0 --seed 0 \
  --reg spectral --spectral_lambda 1e-3 \
  --hessian_tol 1e-3 --hessian_max_iters 20 \
  --name spectrum_probe --exp_name spectrum_probe
rc=$?
end=$(date +%s)
echo "CELLTIME rc=$rc elapsed_s=$(( end - start ))"
