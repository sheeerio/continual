#!/bin/bash
# Q2: Does the joint LR-reg controller beat static reg AND beat adaptive_inv?
# Attribution: joint controller is added ON TOP of static or adaptive_inv.
# Saturating is deliberately absent; the joint controller is being tested
# as an independent contribution.

set -euo pipefail

# Verify we're on the joint-controller branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${BRANCH}" != "joint-controller" ]; then
  echo "ERROR: expected joint-controller branch, got ${BRANCH}"
  exit 1
fi

SCRIPT="implicit_regularization.py"
EXP_NAME="joint_$(date +%Y%m%d)"
LOG_DIR="./logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

COMMON=(
  --track_coherence
  --coherence_window 100
  --runs 5
  --epochs 200
  --ns 1.0
  --lr_schedule constant
  --exp_name "${EXP_NAME}"
)

REGS=(l2 spectral wass)
# Four cells per reg:
#   static           = plain static reg
#   static+joint     = static reg + joint controller
#   adaptive_inv     = plain adaptive
#   adaptive+joint   = adaptive + joint controller
METHODS=(static static_joint adaptive_inv adaptive_inv_joint)
COEFS=(3e-4 1e-3 3e-3 1e-2 3e-2)
LRS=(0.001 0.01)
SEEDS=(0 1 2)

# Joint controller: fixed defaults for the main comparison.
# Sensitivity to these knobs is a separate ablation (joint_ablation.sh).
JOINT_FLAGS=(
  --joint_controller
  --joint_alpha_variant t
  --joint_shock_threshold 0.1
  --joint_max_lr_cool 0.05
  --joint_max_reg_boost 4.0
)

TOTAL=$(( ${#REGS[@]} * ${#METHODS[@]} * ${#COEFS[@]} * ${#LRS[@]} * ${#SEEDS[@]} ))
COUNT=0
START=$(date +%s)

for reg in "${REGS[@]}"; do
  for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
      for coef in "${COEFS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          COUNT=$((COUNT + 1))
          NAME="reg-${reg}_method-${method}_lr-${lr}_coef-${coef}_seed-${seed}"
          MARKER="${LOG_DIR}/${NAME}.done"

          if [ -f "${MARKER}" ]; then
            echo "[${COUNT}/${TOTAL}] SKIP ${NAME}"
            continue
          fi

          # Base reg flags (independent of joint controller)
          case "${reg}" in
            l2)       BASE_STATIC=(--reg l2 --l2_lambda "${coef}") ;;
            spectral) BASE_STATIC=(--reg spectral --spectral_lambda "${coef}" --spectral_k 1.0) ;;
            wass)     BASE_STATIC=(--reg wass --wass_lambda "${coef}") ;;
          esac

          case "${reg}" in
            l2)
              BASE_ADAPTIVE=(--reg none --l2_lambda 0 --adaptive_reg
                             --adaptive_scope local --adaptive_type l2
                             --adaptive_form inv --reg_sensitivity "${coef}")
              ;;
            spectral)
              BASE_ADAPTIVE=(--reg none --l2_lambda 0 --adaptive_reg
                             --adaptive_scope local --adaptive_type spectral
                             --adaptive_form inv --spectral_k 1.0
                             --reg_sensitivity "${coef}")
              ;;
            wass)
              BASE_ADAPTIVE=(--reg none --l2_lambda 0 --adaptive_reg
                             --adaptive_scope local --adaptive_type wass
                             --adaptive_form inv --reg_sensitivity "${coef}")
              ;;
          esac

          case "${method}" in
            static)             METHOD_FLAGS=("${BASE_STATIC[@]}") ;;
            static_joint)       METHOD_FLAGS=("${BASE_STATIC[@]}" "${JOINT_FLAGS[@]}") ;;
            adaptive_inv)       METHOD_FLAGS=("${BASE_ADAPTIVE[@]}") ;;
            adaptive_inv_joint) METHOD_FLAGS=("${BASE_ADAPTIVE[@]}" "${JOINT_FLAGS[@]}") ;;
          esac

          ELAPSED=$(( $(date +%s) - START ))
          echo "[${COUNT}/${TOTAL}] (${ELAPSED}s) RUN ${NAME}"

          python "${SCRIPT}" \
            "${COMMON[@]}" \
            --name "${NAME}" \
            --lr "${lr}" \
            --seed "${seed}" \
            "${METHOD_FLAGS[@]}" \
            > "${LOG_DIR}/${NAME}.log" 2>&1

          touch "${MARKER}"
        done
      done
    done
  done
done

echo "JOINT SWEEP DONE."
