#!/bin/bash
#SBATCH --job-name=stabprobe
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/home/gbaveja/scratch/stability_probe2/logs/%x_%A_%a.out

# TASK A: vanilla stability probe, vs static spectral c=1e-3 as the control.
#
# TWO cells, not one. The Task 2 probe predates alpha_crit_t being a logged
# column, so its diag CSV cannot supply the side-by-side per-task table Task A
# asks for -- the static arm has to be re-run under the updated schema for the
# two to be comparable. Same config as that probe otherwise, so the re-run is
# a drop-in replacement.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled
export SHADOW_CHECK=1

OUT=$HOME/scratch/stability_probe2
mkdir -p "$OUT/logs" "$OUT/cells"

case $SLURM_ARRAY_TASK_ID in
  0) TAG=vanilla;  ARM="--reg none" ;;
  1) TAG=static_c1e-3; ARM="--reg spectral --spectral_lambda 1e-3" ;;
  *) echo "bad array id"; exit 1 ;;
esac

echo "RUN $TAG"
start=$(date +%s)
~/venv/continual/bin/python implicit_regularization.py \
  --optimizer adam --activation relu --runs 3 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 100 --ns 1.0 \
  --diagnostics full \
  $ARM \
  --seed 0 --name "stab_$TAG" --exp_name stability_probe \
  --results_csv "$OUT/cells/${TAG}.csv" \
  --taskdiag_csv "$OUT/cells/${TAG}_taskdiag.csv" \
  --diag_csv "$OUT/cells/${TAG}_diag.csv"
rc=$?
end=$(date +%s)
echo "CELLTIME tag=$TAG rc=$rc elapsed_s=$(( end - start ))"
