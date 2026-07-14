#!/bin/bash

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
MAX_PARALLEL_JOBS=4
SCRIPT_NAME="implicit_regularization.py"

# Grid Dimensions
SEEDS=(1 2 3)
MODES=("td")
LEARNING_RATES=("1e-2" "1e-3" "1e-4")
REG_COEFS=("1e2" "1e1" "1" "1e-1")

# Special Branch Parameters
SCHED_PARAMS=("t")
ADAPTIVE_TYPES=("l2" "parseval" "spectral" "wasserstein")
SENSITIVITIES=("1e-4" "1e-3" "1e-2" "1e-1" "1" "10" "100")
ADAPTIVE_TARGETS=("1e-1" "1e-2")

COMMANDS=()

# --- BUILD COMMAND LIST ---
# Move seed to the outside to ensure we see progress across different initializations early
for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            
            # # 1. BASELINE ALGORITHMS (No Reg Coef needed)
            for algo in "random"; do
                CMD="python3 ${SCRIPT_NAME} --mode ${mode} --algo ${algo} --lr ${lr} --seed ${seed} --lr_schedule constant --reg_coef 0.0"
                COMMANDS+=("$CMD")
            done

            # # 2. STATIC REGULARIZATION (Iterate over REG_COEFS)
            # for algo in "parseval" "l2" "spectral" "wasserstein"; do
            #     for reg in "${REG_COEFS[@]}"; do
            #         CMD="python3 ${SCRIPT_NAME} --mode ${mode} --algo ${algo} --lr ${lr} --seed ${seed} --lr_schedule constant --reg_coef ${reg}"
            #         COMMANDS+=("$CMD")
            #     done
            # done

            # 3. PER-LAYER LYAPUNOV SCHEDULING (No Reg Coef needed)
            # for param in "${SCHED_PARAMS[@]}"; do
            #     CMD="python3 ${SCRIPT_NAME} --mode ${mode} --algo vanilla --lr ${lr} --seed ${seed} --lr_schedule pl_lyapunov --sched_param ${param}"
            #     COMMANDS+=("$CMD")
            # done

            # 4. LOCAL ADAPTIVE REGULARIZATION (Independent of STATIC REG_COEFS)
            for a_type in "${ADAPTIVE_TYPES[@]}"; do
                for sens in "${SENSITIVITIES[@]}"; do
                    for target in "${ADAPTIVE_TARGETS[@]}"; do
                        CMD="python3 ${SCRIPT_NAME} --algo vanilla --lr ${lr} --seed ${seed} --adaptive_reg --adaptive_type ${a_type} --reg_sensitivity ${sens} --adaptive_target ${target}"
                        COMMANDS+=("$CMD")
                    done
                done
            done

        done
    done
done

# =============================================================================
# EXECUTION ENGINE
# =============================================================================

run_and_monitor() {
    local cmd="$1"
    local log="$2"
    # Simple extraction for console output
    local info=$(echo "$cmd" | sed 's/python3 plasticity_experiment.py //')
    
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
    # Generate a unique hash or name for the log file to avoid collisions
    log_name=$(echo "$cmd" | md5sum | cut -d' ' -f1)
    log_file="logs/job_${log_name}.log"

    # Parallel Management
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        for i in "${!PIDS[@]}"; do 
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then 
                unset PIDS[$i]
                break
            fi
        done
        sleep 2
        PIDS=("${PIDS[@]}") # Re-index array
    done
    
    run_and_monitor "$cmd" "$log_file" &
    PIDS+=($!) 
done

wait
echo "Sweep Complete. Total Jobs: ${#COMMANDS[@]}"