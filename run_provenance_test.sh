#!/bin/bash
#SBATCH --job-name=prov_test
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:40:00
#SBATCH --output=/home/gbaveja/scratch/prov_test/logs/%x_%j.out

# Schema verification for TASK 1 (git_sha + full_config on the per-run
# results row) and TASK 2 (per-task diagnostic snapshots: long-format
# taskdiag sink + ;-joined trajectory columns on the results row).
#
# Deliberately tiny -- 4 tasks, 1 epoch, hidden 32. This is a schema and
# sanity check, NOT an experiment; the numbers carry no scientific meaning.
# --adaptive_reg and --track_coherence are both on so the adaptive_factor
# and mean_off_diag columns are actually exercised rather than left blank.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

OUT=/home/gbaveja/scratch/prov_test
mkdir -p "$OUT/logs"
# Fresh files: the results_csv header is derived from the first row's keys,
# so appending new columns onto a pre-existing CSV would misalign it.
rm -f "$OUT/results.csv" "$OUT/taskdiag.csv"

~/venv/continual/bin/python implicit_regularization.py \
  --optimizer adam --activation relu --runs 4 --epochs 2 \
  --dataset MNIST --model MLP --hidden 32 --lr 0.001 --batch_size 256 \
  --log_interval 10 --ns 1.0 \
  --reg l2_loss --l2_lambda 1e-3 \
  --adaptive_reg --adaptive_type l2 --adaptive_scale inv --reg_sensitivity 0.001 \
  --track_coherence --coherence_window 5 \
  --seed 0 --name prov_test --exp_name prov_test \
  --results_csv "$OUT/results.csv" \
  --taskdiag_csv "$OUT/taskdiag.csv"

echo "exit=$?"
echo "done"
