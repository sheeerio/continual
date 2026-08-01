#!/bin/bash
#SBATCH --job-name=ortho_qc
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/gbaveja/scratch/ortho_quickcheck/logs/%x_%j.out

# Quick single-seed, reduced-task sanity check that the fixed ortho
# regularizer actually beats vanilla, before committing to the full
# 3-seed/100-task reproduction. 15 tasks is well past vanilla's task~2-3
# collapse point, so it's enough to see divergence if the fix works.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

ROOT=$HOME/scratch/ortho_quickcheck
python implicit_regularization.py \
  --optimizer adam --activation relu --runs 15 --epochs 100 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 400 --ns 1.0 \
  --reg ortho --ortho_lambda 1e-3 \
  --seed 0 --name ortho_quickcheck_seed0 --exp_name ortho_quickcheck \
  --results_csv "${ROOT}/cells/ortho_quickcheck_seed0.csv"

echo "done"
