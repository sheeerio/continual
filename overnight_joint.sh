#!/bin/bash
set -euo pipefail

# Requires: git checkout joint-controller (on Machine B)
SCRIPT="implicit_regularization.py"
EXP_NAME="overnight_joint_$(date +%Y%m%d)"
LOG_DIR="./logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

COMMON=(
  --reg l2_init
  --adaptive_reg
  --adaptive_scope local
  --adaptive_type l2
  --adaptive_form saturating
  --tau_ref_mode median
  --tau_ref_window 500
  --track_coherence
  --coherence_window 100
  --runs 5
  --epochs 200
  --ns 1.0
  --lr 0.01
  --reg_sensitivity 1e-3
  --exp_name "${EXP_NAME}"
)

# Baseline: adaptive_saturating alone, no joint controller
# Then joint controller with varying knobs
SHOCK_THRESHOLDS=(0.05 0.1 0.2)
MAX_LR_COOLS=(0.02 0.05 0.1)
MAX_REG_BOOSTS=(2.0 4.0 8.0)
SEEDS=(0 1 2)

COUNT=0
START=$(date +%s)

# --- baseline runs (no joint controller) ---
for seed in "${SEEDS[@]}"; do
  COUNT=$((COUNT + 1))
  NAME="baseline_seed-${seed}"
  MARKER="${LOG_DIR}/${NAME}.done"
  [ -f "${MARKER}" ] && { echo "SKIP ${NAME}"; continue; }
  echo "[${COUNT}] RUN ${NAME}"
  python "${SCRIPT}" "${COMMON[@]}" --name "${NAME}" --seed "${seed}" \
    > "${LOG_DIR}/${NAME}.log" 2>&1
  touch "${MARKER}"
done

# --- joint controller sweep ---
for shock in "${SHOCK_THRESHOLDS[@]}"; do
  for cool in "${MAX_LR_COOLS[@]}"; do
    for boost in "${MAX_REG_BOOSTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        NAME="joint_shock-${shock}_cool-${cool}_boost-${boost}_seed-${seed}"
        MARKER="${LOG_DIR}/${NAME}.done"
        [ -f "${MARKER}" ] && { echo "SKIP ${NAME}"; continue; }
        ELAPSED=$(( $(date +%s) - START ))
        echo "[${COUNT}] (${ELAPSED}s) RUN ${NAME}"
        python "${SCRIPT}" "${COMMON[@]}" \
          --name "${NAME}" --seed "${seed}" \
          --joint_controller --joint_alpha_variant t \
          --joint_shock_threshold "${shock}" \
          --joint_max_lr_cool "${cool}" \
          --joint_max_reg_boost "${boost}" \
          > "${LOG_DIR}/${NAME}.log" 2>&1
        touch "${MARKER}"
      done
    done
  done
done

echo "JOINT SWEEP DONE."
