#!/bin/bash
# Variant of plot.sh dedicated to the NFM/AGOP feature-diagnostics comparison:
# vanilla vs. l2/spectral/parseval/wass, 3 seeds each, 20 tasks, 200 epochs/task.
# After all jobs finish, pulls feature_diag/*/raw_corr and feature_diag/*/delta_corr
# from wandb and plots mean +/- std across seeds vs. task in matplotlib.

MAX_PARALLEL_JOBS=3
SCRIPT_NAME="implicit_regularization.py"

SEEDS=(1 2 3)
ALGOS=("vanilla" "l2" "spectral" "parseval" "wass")

RUNS=20
EPOCHS=200
LR="1e-3"
L2_LAMBDA="1e-3"
SPECTRAL_LAMBDA="1e-3"
WASS_LAMBDA="1e-3"

EXP_NAME="nfm_agop_diag_t20"

COMMANDS=()

for seed in "${SEEDS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        REG_FLAGS=""
        case "$algo" in
            vanilla)
                REG_FLAGS=""
                ;;
            l2)
                REG_FLAGS="--reg l2 --l2_lambda ${L2_LAMBDA}"
                ;;
            spectral)
                REG_FLAGS="--reg spectral --spectral_lambda ${SPECTRAL_LAMBDA}"
                ;;
            parseval)
                REG_FLAGS="--reg parseval"
                ;;
            wass)
                REG_FLAGS="--reg wass --wass_lambda ${WASS_LAMBDA}"
                ;;
        esac

        CMD="python3 ${SCRIPT_NAME} --runs ${RUNS} --dataset MNIST \
--name ${algo}_seed${seed} --exp_name ${EXP_NAME} --epochs ${EPOCHS} \
--optimizer adam --lr ${LR} --seed ${seed} ${REG_FLAGS} \
--feature_diagnostics --feature_diag_every 1 --feature_diag_max_plot_tasks 10"
        COMMANDS+=("$CMD")
    done
done

run_and_monitor() {
    local cmd="$1"
    local log="$2"
    local info=$(echo "$cmd" | sed 's/python3 implicit_regularization.py //')

    echo "[$(date +'%H:%M:%S')] Starting: $info"
    eval "$cmd" > "$log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[ERROR] Job failed. Check: $log"
    else
        echo "[SUCCESS] Job completed."
    fi
}

mkdir -p logs
PIDS=()

for cmd in "${COMMANDS[@]}"; do
    log_name=$(echo "$cmd" | md5sum | cut -d' ' -f1)
    log_file="logs/job_${log_name}.log"

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

    run_and_monitor "$cmd" "$log_file" &
    PIDS+=($!)
done

wait
echo "Sweep Complete. Total Jobs: ${#COMMANDS[@]}"

echo "Plotting NFM/AGOP correlation (mean +/- std across seeds vs. task)..."
python3 plot_nfm_agop_correlation.py --group "${EXP_NAME}" --out_dir "plots/nfm_agop_correlation_${EXP_NAME}"
