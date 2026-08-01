#!/bin/bash
#SBATCH --job-name=efflrprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/gbaveja/scratch/efflr_probe/logs/%x_%j.out

# TASK 2 probe -- ONE cell. The question is binary, not distributional: does
# eff_lr ever approach 0.12 in this setup at all?
#
# eff_lr = lr / (bc1 * (sqrt(v_hat_mean) + eps)), so at lr=1e-3 crossing 0.12
# requires per-element RMS gradient below ~8.33e-3, i.e. >120x preconditioner
# amplification. If the observed max is orders of magnitude short of that, the
# 0.12 floor is an SGD-regime constant (lr=0.01 + momentum 0.9 gives eff_lr
# ~0.1) that does not transfer to Adam at lr=1e-3.
#
# 3 tasks only, --diagnostics full so the per-step diag_csv records eff_lr per
# layer (the light schema now carries per-task mean/min/max, but percentile-
# style questions need the per-step series).

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

OUT=$HOME/scratch/efflr_probe
mkdir -p "$OUT/logs" "$OUT/cells"

~/venv/continual/bin/python implicit_regularization.py \
  --optimizer adam --activation relu --runs 3 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 100 --ns 1.0 \
  --diagnostics full \
  --reg spectral --spectral_lambda 1e-3 \
  --lr_schedule pl_lyapunov --param t \
  --seed 0 --name efflr_probe --exp_name efflr_probe \
  --results_csv "$OUT/cells/probe.csv" \
  --taskdiag_csv "$OUT/cells/probe_taskdiag.csv" \
  --diag_csv "$OUT/cells/probe_diag.csv"

echo "CELLTIME rc=$? "
