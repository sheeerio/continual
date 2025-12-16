#!/bin/bash

MAX_PARALLEL_JOBS=3
COMMANDS=()

# ==============================================================================
# COMMAND GENERATION
# ==============================================================================

for seed in 1 3; do
    for lr in 1e-4 1e-3 1e-2; do

        # ======================================================================
        # PART 1: STATIC REGULARIZERS (Run ONCE per LR)
        # These do NOT depend on sensitivity.
        # ======================================================================
        
        # We loop through the 3 schedule types here
        # 1. None ("")
        # 2. PL T ("_pl_grad")
        # 3. PL Svar10 ("_pl")
        for schedule_type in "none" "pl_grad" "pl_svar"; do
            
            # Define Schedule Flags and Suffix
            if [ "$schedule_type" == "none" ]; then
                s_flags=""
                s_suffix=""
            elif [ "$schedule_type" == "pl_grad" ]; then
                s_flags="--lr_schedule pl_lyapunov --param t"
                s_suffix="_pl_grad"
            elif [ "$schedule_type" == "pl_svar" ]; then
                s_flags="--lr_schedule pl_lyapunov --param svar10"
                s_suffix="_pl"
            fi

            # --- 1. Baseline (ReLU) ---
            COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --runs 50 --name=relu${s_suffix}_lr_${lr} --exp_name=static --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

            # --- 2. L2 ---
            COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --reg l2 --l2_lambda 1e-3 --runs 50 --name=relu+l2${s_suffix}_lr_${lr} --exp_name=static --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

            # --- 3. Spectral ---
            COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --reg spectral --runs 50 --name=relu+spectral${s_suffix}_lr_${lr} --exp_name=static --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

            # --- 4. Wasserstein ---
            COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --reg wass --runs 50 --name=relu+wass${s_suffix}_lr_${lr} --exp_name=static --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")
        
        done


        # ======================================================================
        # PART 2: ADAPTIVE REGULARIZERS (Run for EACH Sensitivity)
        # These depend on the sensitivity loop.
        # ======================================================================
        
        for sens in 5e-3 1e-2 5e-2; do
            
            # Repeat the schedule loop for adaptive runs
            for schedule_type in "none" "pl_grad" "pl_svar"; do
                
                # Define Schedule Flags and Suffix (Resetting ensuring no leakage)
                if [ "$schedule_type" == "none" ]; then
                    s_flags=""
                    s_suffix=""
                elif [ "$schedule_type" == "pl_grad" ]; then
                    s_flags="--lr_schedule pl_lyapunov --param t"
                    s_suffix="_pl_grad"
                elif [ "$schedule_type" == "pl_svar" ]; then
                    s_flags="--lr_schedule pl_lyapunov --param svar10"
                    s_suffix="_pl"
                fi

                # --- 5. Adaptive L2 ---
                COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --adaptive_reg --adaptive_type l2 --reg_sensitivity ${sens} --runs 50 --name=relu+adreg_l2${s_suffix}_lr_${lr} --exp_name=sens_${sens} --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

                # --- 6. Adaptive Spectral ---
                COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --adaptive_reg --adaptive_type spectral --reg_sensitivity ${sens} --runs 50 --name=relu+adreg_spectral${s_suffix}_lr_${lr} --exp_name=sens_${sens} --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

                # --- 7. Adaptive Wasserstein ---
                COMMANDS+=("python3 implicit_regularization.py --seed ${seed} --optimizer adam --activation relu --adaptive_reg --adaptive_type wass --reg_sensitivity ${sens} --runs 50 --name=relu+adreg_wass${s_suffix}_lr_${lr} --exp_name=sens_${sens} --epochs 100 --lr=${lr} --dataset=MNIST ${s_flags}")

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