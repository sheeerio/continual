#!/bin/bash
#SBATCH --job-name=diag_test
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:20:00
#SBATCH --output=/home/gbaveja/scratch/diag_csv_test/logs/%x_%j.out

# Minimal integration test for the diagnostic CSV sink: tiny run, small
# hidden width, log_interval=1 so the diagnostic block fires almost every
# step, --track_coherence on so both the per-layer and coherence rows get
# exercised. Just checking it doesn't crash and produces sane rows.

set -uo pipefail
module load python/3.12
source ~/venv/continual/bin/activate
export WANDB_MODE=disabled

python implicit_regularization.py \
  --optimizer adam --activation relu --runs 2 --epochs 2 \
  --dataset MNIST --model MLP --hidden 32 --lr 0.001 --batch_size 256 \
  --log_interval 1 --ns 1.0 --track_coherence \
  --reg l2_loss --l2_lambda 1e-3 \
  --seed 0 --name diag_csv_test --exp_name diag_csv_test \
  --diag_csv "/home/gbaveja/scratch/diag_csv_test/diag.csv"

echo "done"
