#!/bin/bash
#SBATCH --job-name=diagcost
#SBATCH --account=def-schmidtm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/gbaveja/scratch/diagcost/logs/%x_%A_%a.out

# STEP 2: quantify the --diagnostics cost lever on the cell the testbed will
# actually run (t20_e50, the Step 1 recommendation), not on a config we won't use.
#
# log_interval=100 deliberately, NOT the 400 used for calibration. At 400 the
# lever is nearly invisible: t20_e50 is 2100 steps/task, so light's coarse grid
# resolves to max(400, 2100//10=210) = 400 -- identical to full's cadence, and
# the two modes would differ only in CSV writes. At 100 the three modes are
# genuinely distinct: full fires the expensive estimates 21x/task, light ~10x,
# off never. 100 is also what Step 3 should use.
#
# Cells 0-2: the lever itself, on the static spectral arm.
# Cell 3: adaptive spectral at light -- Step 3 is mostly adaptive arms, which
# carry a per-layer iters=1 Hessian estimate that --diagnostics never gates
# (it is method, not diagnostics), so the static numbers alone would
# under-project the Step 3 grid.

set -uo pipefail
module load python/3.12
export WANDB_MODE=disabled

OUT=$HOME/scratch/diagcost
mkdir -p "$OUT/logs" "$OUT/cells"

PY=$HOME/venv/continual/bin/python
BASE="--optimizer adam --activation relu --runs 20 --epochs 50 \
  --dataset MNIST --model MLP --hidden 256 --lr 0.001 --batch_size 256 \
  --log_interval 100 --ns 1.0 --seed 0 --exp_name diagcost"

case $SLURM_ARRAY_TASK_ID in
  0) TAG=static_off;      ARM="--reg spectral --spectral_lambda 1e-3"; DIAG=off   ;;
  1) TAG=static_light;    ARM="--reg spectral --spectral_lambda 1e-3"; DIAG=light ;;
  2) TAG=static_full;     ARM="--reg spectral --spectral_lambda 1e-3"; DIAG=full  ;;
  3) TAG=adaptive_light;  ARM="--reg none --adaptive_reg --adaptive_type spectral --adaptive_scale inv --reg_sensitivity 1e-3"; DIAG=light ;;
  *) echo "bad array id"; exit 1 ;;
esac

CSV="$OUT/cells/${TAG}.csv"
if [ -s "$CSV" ]; then
  echo "skip done: $CSV"
  exit 0
fi

EXTRA=""
if [ "$DIAG" != "off" ]; then
  EXTRA="--taskdiag_csv $OUT/cells/${TAG}_taskdiag.csv --track_coherence --coherence_window 20"
fi
if [ "$DIAG" = "full" ]; then
  EXTRA="$EXTRA --diag_csv $OUT/cells/${TAG}_diag.csv"
fi

echo "RUN $TAG (diagnostics=$DIAG)"
start=$(date +%s)
$PY implicit_regularization.py $BASE $ARM --diagnostics "$DIAG" \
    --name "$TAG" --results_csv "$CSV" $EXTRA
rc=$?
end=$(date +%s)
echo "CELLTIME tag=$TAG diagnostics=$DIAG rc=$rc elapsed_s=$(( end - start ))"
