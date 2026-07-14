#!/bin/bash
# Q2: Does the joint LR-reg controller beat the plain versions?
# 4 cells per reg: static, static+joint, adaptive_inv, adaptive_inv+joint.
# No saturating anywhere. Joint controller knobs fixed at defaults.
# Assumes: git checkout joint-controller.
# 3 regs x 4 methods x 2 LRs x 5 coefs x 3 seeds = 360 runs.

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${BRANCH}" != "joint-controller" ]; then
    echo "ERROR: expected joint-controller branch, got ${BRANCH}"
    exit 1
fi

MAX_PARALLEL_JOBS=3
ACT="relu"
EXP_NAME="joint_$(date +%Y%m%d)"
LOG_DIR="./logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

JOINT_FLAGS="--joint_controller --joint_alpha_variant t --joint_shock_threshold 0.1 --joint_max_lr_cool 0.05 --joint_max_reg_boost 4.0"

COMMANDS=()

REGS=("l2" "spectral" "wass")
METHODS=("static" "static_joint" "adaptive_inv" "adaptive_inv_joint")
COEFS=(3e-4 1e-3 3e-3 1e-2 3e-2)
LRS=(0.001 0.01)
SEEDS=(0 1 2)

for reg in "${REGS[@]}"; do
  for method in "${METHODS[@]}"; do
    for lr in "${LRS[@]}"; do
      for coef in "${COEFS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          NAME="reg-${reg}_method-${method}_lr-${lr}_coef-${coef}_seed-${seed}"

          # base static flags
          case "${reg}" in
            l2)       BASE_STATIC="--reg l2 --l2_lambda ${coef}" ;;
            spectral) BASE_STATIC="--reg spectral --spectral_lambda ${coef} --spectral_k 1.0" ;;
            wass)     BASE_STATIC="--reg wass --wass_lambda ${coef}" ;;
          esac

          # base adaptive_inv flags
          case "${reg}" in
            l2)
              BASE_ADAPTIVE="--reg none --l2_lambda 0 --adaptive_reg --adaptive_scope local --adaptive_type l2 --adaptive_form inv --reg_sensitivity ${coef}" ;;
            spectral)
              BASE_ADAPTIVE="--reg none --l2_lambda 0 --adaptive_reg --adaptive_scope local --adaptive_type spectral --adaptive_form inv --spectral_k 1.0 --reg_sensitivity ${coef}" ;;
            wass)
              BASE_ADAPTIVE="--reg none --l2_lambda 0 --adaptive_reg --adaptive_scope local --adaptive_type wass --adaptive_form inv --reg_sensitivity ${coef}" ;;
          esac

          case "${method}" in
            static)             METHOD_FLAGS="${BASE_STATIC}" ;;
            static_joint)       METHOD_FLAGS="${BASE_STATIC} ${JOINT_FLAGS}" ;;
            adaptive_inv)       METHOD_FLAGS="${BASE_ADAPTIVE}" ;;
            adaptive_inv_joint) METHOD_FLAGS="${BASE_ADAPTIVE} ${JOINT_FLAGS}" ;;
          esac

          CMD="python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation ${ACT} --runs 10 --epochs 200 --ns 1.0 --lr ${lr} --dataset MNIST --lr_schedule constant --track_coherence --coherence_window 100 ${METHOD_FLAGS} --name=${NAME} --exp_name=${EXP_NAME}"
          COMMANDS+=("${CMD}")
        done
      done
    done
  done
done

run_and_monitor() {
    local cmd="$1"
    local name="$2"
    local log="$3"
    local marker="${LOG_DIR}/${name}.done"

    if [ -f "${marker}" ]; then
        echo "SKIP (already done): $name"
        return 0
    fi

    echo "Now running: $name"
    eval "$cmd" > "$log" 2>&1

    if [ $? -ne 0 ]; then
        echo "Error running: $name (check $log)"
    else
        echo "Completed: $name"
        touch "${marker}"
    fi
}

# --- PARALLEL EXECUTION LOOP ---
PIDS=()
TOTAL=${#COMMANDS[@]}
IDX=0
for cmd in "${COMMANDS[@]}"; do
    IDX=$((IDX + 1))
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
                break
            fi
        done
        sleep 2
        PIDS=("${PIDS[@]}")
    done

    exp_name=$(echo "$cmd" | sed -n 's/.*--name=\([^ ]*\).*/\1/p')
    log_file="${LOG_DIR}/${exp_name}.log"

    echo "[${IDX}/${TOTAL}] Launching ${exp_name}"
    run_and_monitor "$cmd" "$exp_name" "$log_file" &
    PIDS+=($!)
done
wait
echo "JOINT SWEEP DONE."