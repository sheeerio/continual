#!/bin/bash

MAX_PARALLEL_JOBS=3
COMMANDS=()

# ==============================================================================
# COMMAND GENERATION
# ==============================================================================
for seed in 1 2 3; do
    for lr in 1e-3 1e-2; do
        # ======================================================================
        # PART 2: ADAPTIVE REGULARIZERS (Run for EACH Sensitivity)
        # These depend on the sensitivity loop.
        # ======================================================================
        
        for sens in 5e-2 1e-1; do
            
            # Repeat the schedule loop for adaptive runs
            for schedule_type in "none" "pl_grad"; do
                
                # Define Schedule Flags and Suffix (Resetting ensuring no leakage)
                if [ "$schedule_type" == "none" ]; then
                    s_flags=""
                    s_suffix=""
                elif [ "$schedule_type" == "pl_grad" ]; then
                    s_flags="--lr_schedule pl_lyapunov --param t"
                    s_suffix="_pl_grad"
                fi
                # --- 5. Adaptive L2 ---
                # COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --adaptive_reg --adaptive_type l2 --reg_sensitivity ${sens} --runs 50 --name=relu+adreg_l2${s_suffix}_lr_${lr} --exp_name=sens_${sens} --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

                # --- 6. Adaptive Spectral ---
                COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --adaptive_reg --adaptive_type spectral --reg_sensitivity ${sens} --runs 50 --name=relu+adreg_spectral${s_suffix}_lr_${lr} --exp_name=sens_${sens} --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")
            done
        done

    done
done

echo "Generated ${#COMMANDS[@]} commands."

# ==============================================================================
# PARALLEL EXECUTION LOOP
# ==============================================================================

PIDS=()
mkdir -p logs

for cmd in "${COMMANDS[@]}"; do
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
                break
            fi
        done
        sleep 5
        PIDS=("${PIDS[@]}")
    done

    # Extract name for logging (Handles both --name=val and --name val)
    exp_name=$(echo "$cmd" | sed -n 's/.*--name[= ]\([^ ]*\).*/\1/p')
    timestamp=$(date +%s)
    log_file="logs/${exp_name}_${timestamp}.log"
    
    echo "Starting: $exp_name"
    
    nohup $cmd > "$log_file" 2>&1 &
    PIDS+=($!) 
done

echo "Waiting for all experiments to finish..."
wait
echo "All experiments completed."