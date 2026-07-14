#!/bin/bash
# Q1: Does adaptive_saturating beat adaptive_inv and static across regularizers?
# Fixed: no scheduler, constant LR. That isolates the reg contribution.
# The scheduler crossing is a separate sweep (main_sweep_sched.sh below).

set -euo pipefail

SCRIPT="implicit_regularization.py"
EXP_NAME="main_$(date +%Y%m%d)"
LOG_DIR="./logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

COMMON=(
  --track_coherence
  --coherence_window 100
  --runs 10
  --epochs 200
  --ns 1.0
  --lr_schedule constant
  --exp_name "${EXP_NAME}"
)

REGS=(l2 spectral wass)
METHODS=(static adaptive_inv adaptive_saturating)
COEFS=(3e-4 1e-3 3e-3 1e-2 3e-2)
LRS=(0.001 0.01)
SEEDS=(0 1 2)

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

          case "${method}_${reg}" in
            # ---- STATIC ----
            static_l2)
              METHOD_FLAGS=(--reg l2 --l2_lambda "${coef}")
              ;;
            static_spectral)
              METHOD_FLAGS=(--reg spectral --spectral_lambda "${coef}" --spectral_k 1.0)
              ;;
            static_wass)
              METHOD_FLAGS=(--reg wass --wass_lambda "${coef}")
              ;;

            # ---- ADAPTIVE INV ----
            adaptive_inv_l2)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type l2
                            --adaptive_form inv --reg_sensitivity "${coef}")
              ;;
            adaptive_inv_spectral)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type spectral
                            --adaptive_form inv --spectral_k 1.0
                            --reg_sensitivity "${coef}")
              ;;
            adaptive_inv_wass)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type wass
                            --adaptive_form inv --reg_sensitivity "${coef}")
              ;;

            # ---- ADAPTIVE SATURATING ----
            adaptive_saturating_l2)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type l2
                            --adaptive_form saturating --tau_ref_mode median
                            --tau_ref_window 500 --reg_sensitivity "${coef}")
              ;;
            adaptive_saturating_spectral)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type spectral
                            --adaptive_form saturating --tau_ref_mode median
                            --tau_ref_window 500 --spectral_k 1.0
                            --reg_sensitivity "${coef}")
              ;;
            adaptive_saturating_wass)
              METHOD_FLAGS=(--reg none --l2_lambda 0 --adaptive_reg
                            --adaptive_scope local --adaptive_type wass
                            --adaptive_form saturating --tau_ref_mode median
                            --tau_ref_window 500 --reg_sensitivity "${coef}")
              ;;
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

echo "MAIN SWEEP DONE."
