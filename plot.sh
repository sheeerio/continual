#!/bin/bash

MAX_PARALLEL_JOBS=2
SCRIPT_NAME="implicit_regularization.py"

SEEDS=(1 2 3)
# MODES=("mc" "td")
LEARNING_RATES=("1e-2" "1e-3" "1e-4")
REG_COEFS=("1e-1" "1e-2" "1e-3" "1e-4")

SCHED_PARAMS=("tau" "svar" "t")
ADAPTIVE_TYPES=("l2" "parseval" "spectral" "wass")
SENSITIVITIES=("1e-3" "1e-2" "1e-1" "1" "10")
ADAPTIVE_TARGETS=("1e-1" "1e-2")

COMMANDS=()

for seed in "${SEEDS[@]}"; do
    # for mode in "${MODES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            

            # 2. STATIC REGULARIZATION (Iterate over REG_COEFS)
            for algo in "l2" "spectral" "parseval" "wass"; do
                for reg in "${REG_COEFS[@]}"; do
                    if [ "$lr" = "1e-2" ] && [ "$reg" != "1e-1" ]; then
                        continue
                    fi
                    if [ "$algo" = "l2" ]; then
                        LAMBDA_FLAG="--l2_lambda"
                    else
                        LAMBDA_FLAG="--spectral_lambda"
                    fi
                    CMD="python3 ${SCRIPT_NAME} --runs 30 --dataset PermutedMNIST --name ${algo}_coef${reg}_lr${lr} --exp_name ema_agg_permuted --epochs 200 --optimizer adam --lr ${lr} --seed ${seed} --reg ${algo} ${LAMBDA_FLAG} ${reg}"
                    COMMANDS+=("$CMD")
                done
            done

            # 3. PER-LAYER LYAPUNOV SCHEDULING (No Reg Coef needed)
            # for param in "${SCHED_PARAMS[@]}"; do
            #     CMD="python3 ${SCRIPT_NAME} --mode ${mode} --algo vanilla --lr ${lr} --seed ${seed} --lr_schedule pl_lyapunov --sched_param ${param}"
            #     COMMANDS+=("$CMD")
            # done

            # 4. LOCAL ADAPTIVE REGULARIZATION (Independent of STATIC REG_COEFS)
            for a_type in "${ADAPTIVE_TYPES[@]}"; do
                # For lr=1e-2, only spectral and wass remain (l2 and parseval already done)
                if [ "$lr" = "1e-2" ] && [ "$a_type" != "spectral" ] && [ "$a_type" != "wass" ]; then
                    continue
                fi
                for sens in "${SENSITIVITIES[@]}"; do
                    CMD="python3 ${SCRIPT_NAME} --runs 30 --dataset PermutedMNIST --name adaptive_${a_type}_sens${sens}_lr${lr} --exp_name ema_agg_permuted --epochs 200 --optimizer adam --lr ${lr} --seed ${seed} --adaptive_reg --adaptive_type ${a_type} --reg_sensitivity ${sens}"
                    COMMANDS+=("$CMD")
                done
            done

            # 1. BASELINE ALGORITHMS (No Reg Coef needed)
            CMD="python3 ${SCRIPT_NAME} --runs 30 --dataset PermutedMNIST --name vanilla_lr${lr} --exp_name ema_agg_permuted --epochs 200 --optimizer adam --lr ${lr} --seed ${seed}"
            COMMANDS+=("$CMD")

            CMD="python3 ${SCRIPT_NAME} --runs 30 --dataset PermutedMNIST --name reset_lr${lr} --exp_name ema_agg_permuted --epochs 200 --optimizer adam --lr ${lr} --seed ${seed} --reset_model"
            COMMANDS+=("$CMD")


        done
    # done
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