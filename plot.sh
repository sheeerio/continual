#!/bin/bash
MAX_PARALLEL_JOBS=3
SEED=2
ACT="relu"
COMMANDS=()

for lr in 1e-2; do
    EXP_GROUP="exp_${ACT}_lr_${lr}"
    # --- STATIC ---
    for reg_type in "l2" "spectral" "none"; do
        if [ "$reg_type" == "none" ]; then
            for sched in "none" "pl_grad" "reset"; do
                s_f=""; s_s=""
                [[ "$sched" == "pl_grad" ]] && { s_f="--lr_schedule pl_lyapunov --param t"; s_s="_pl_grad"; }
                [[ "$sched" == "reset" ]] && { s_f="--reset_model"; s_s="_reset"; }
                COMMANDS+=("python3 implicit_regularization.py --seed ${SEED} --optimizer adam --activation ${ACT} --runs 100 --name=${ACT}_none${s_s}_lr_${lr} --exp_name=${EXP_GROUP} --epochs 100 --lr=${lr} --dataset=MNIST ${s_f}")
            done
        else
            for lam in 1e-4 1e-3 1e-2; do
                for sched in "none" "grad" "svar"; do
                    s_f=""; s_s=""; reg_f="--reg ${reg_type}"
                    [[ "$reg_type" == "l2" ]] && reg_f="${reg_f} --l2_lambda ${lam}"
                    [[ "$reg_type" == "spectral" ]] && reg_f="${reg_f} --spectral_lambda ${lam}"
                    [[ "$sched" == "grad" ]] && { s_f="--lr_schedule pl_lyapunov --param t"; s_s="_pl_grad"; }
                    [[ "$sched" == "svar" ]] && { s_f="--lr_schedule pl_lyapunov --param svar10"; s_s="_pl_svar"; }
                    COMMANDS+=("python3 implicit_regularization.py --seed ${SEED} --optimizer adam --activation ${ACT} ${reg_f} --runs 100 --name=${ACT}_${reg_type}_lam${lam}${s_s}_lr_${lr} --exp_name=${EXP_GROUP} --epochs 100 --lr=${lr} --dataset=MNIST ${s_f}")
                done
            done
        fi
    done
    # --- ADAPTIVE ---
    for reg_type in "l2" "spectral"; do
        for sens in 1e-4 1e-3 1e-2; do
            for sched in "none" "grad"; do
                s_f=""; s_s=""
                [[ "$sched" == "grad" ]] && { s_f="--lr_schedule pl_lyapunov --param t"; s_s="_pl_grad"; }
                COMMANDS+=("python3 implicit_regularization.py --seed ${SEED} --optimizer adam --activation ${ACT} --adaptive_reg --adaptive_type ${reg_type} --reg_sensitivity ${sens} --runs 100 --name=${ACT}_adaptive_${reg_type}_sens${sens}${s_s}_lr_${lr} --exp_name=${EXP_GROUP} --epochs 100 --lr=${lr} --dataset=MNIST ${s_f}")
            done
        done
    done
done

run_and_monitor() {
    local cmd="$1"
    local name="$2"
    local log="$3"
    
    echo "Now running: $name"
    # Execute the command and redirect logs
    eval "$cmd" > "$log" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Error running: $name (check $log)"
    else
        echo "Completed: $name"
    fi
}

# --- PARALLEL EXECUTION LOOP ---
PIDS=(); mkdir -p logs
for cmd in "${COMMANDS[@]}"; do
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        for i in "${!PIDS[@]}"; do if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then unset PIDS[$i]; break; fi; done
        sleep 2; PIDS=("${PIDS[@]}")
    done
    
    exp_name=$(echo "$cmd" | sed -n 's/.*--name[= ]\([^ ]*\).*/\1/p')
    log_file="logs/${exp_name}_s${SEED}.log"
    
    # Run the wrapper in the background
    run_and_monitor "$cmd" "$exp_name" "$log_file" &
    PIDS+=($!) 
done

wait
echo "All experiments in this file have completed."