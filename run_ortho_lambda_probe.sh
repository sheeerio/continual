#!/bin/bash
#SBATCH --job-name=ortho_lprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/gbaveja/scratch/ortho_lambda_probe/logs/%x_%A_%a.out

# ortho_lambda=1e-3 showed no visible effect vs vanilla (quick 15-task check).
# Probe 3 much larger coefficients, single seed, 10 tasks each, to see if
# the fixed ortho penalty can move the needle at all with enough strength.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

ROOT=$HOME/scratch/ortho_lambda_probe
LAMBDAS=(0.1 1.0 10.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
NAME="ortho_lam${LAM}_seed0"

python implicit_regularization.py \
  --optimizer adam --activation relu --runs 10 --epochs 100 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 400 --ns 1.0 \
  --reg ortho --ortho_lambda "${LAM}" \
  --seed 0 --name "${NAME}" --exp_name ortho_lambda_probe \
  --results_csv "${ROOT}/cells/${NAME}.csv"

echo "done: lambda=${LAM}"
