#!/bin/bash
#SBATCH --job-name=centchk
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:40:00
#SBATCH --output=/home/gbaveja/scratch/centered_check/logs/%x_%j.out

# TASK B2: verify the centering BEFORE spending 72 cells on B3.
# centered => factor = sensitivity * (1 + kappa*(g - 0.5)), and g is centered
# at ~0.5 by construction, so mean(adaptive_factor) should land at
# ~sensitivity = 1e-3 regardless of kappa. If it doesn't, the centering is
# wrong and B3 does not run.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

OUT=$HOME/scratch/centered_check
mkdir -p "$OUT/logs" "$OUT/cells"

start=$(date +%s)
~/venv/continual/bin/python implicit_regularization.py \
  --optimizer adam --activation relu --runs 20 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 100 --ns 1.0 \
  --diagnostics light --track_coherence --coherence_window 20 \
  --reg none --adaptive_reg --adaptive_type spectral \
  --adaptive_scale centered --sat_kappa 5 --reg_sensitivity 1e-3 \
  --seed 0 --name centered_k5_c1e-3 --exp_name centered_check \
  --results_csv "$OUT/cells/centered_k5_c1e-3.csv" \
  --taskdiag_csv "$OUT/cells/centered_k5_c1e-3_taskdiag.csv"
rc=$?
end=$(date +%s)
echo "CELLTIME rc=$rc elapsed_s=$(( end - start ))"
