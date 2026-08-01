#!/bin/bash
#SBATCH --job-name=estfinal
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=05:00:00
#SBATCH --output=/home/gbaveja/scratch/estfinal_probe/logs/%x_%j.out
set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled
mkdir -p "$HOME/scratch/estfinal_probe/logs"
start=$(date +%s)
~/venv/continual/bin/python estimator_final_probe.py \
  --optimizer adam --activation relu --runs 3 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 10 --ns 1.0 --seed 0 \
  --reg spectral --spectral_lambda 1e-3 \
  --name estfinal --exp_name estfinal
rc=$?
end=$(date +%s)
echo "CELLTIME rc=$rc elapsed_s=$(( end - start ))"
