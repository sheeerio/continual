#!/bin/bash

###############################################################################
# EXPERIMENT 1: FREEZE LAYERS + BASELINES
# Scope: Iterate Optimizers -> Freeze Layers -> Reg Baselines
###############################################################################

MAX_PARALLEL_JOBS=4
PIDS=()
COMMANDS=()

# 1. Definitions
SEEDS=(1)
OPTIMIZERS=("sgd" "adam")

# Define layers to freeze. "None" is the baseline (no freezing).
# Adjust "fc1" "fc2" "fc3" based on your actual model layer names.
LAYERS_TO_FREEZE=("None" "fc1" "fc2" "fc4") 
CAP_TYPES=("none" "hard" "soft")

SCHED_CONFIGS=(
    "fc1:wsd,fc3:cosine"
    "fc1:cosine,fc3:wsd"
    "fc1:power,fc3:wsd"
    "fc1:wsd,fc3:power"
    "fc1:cosine,fc3:cosine" # Control 1
    "fc1:wsd,fc3:wsd"       # Control 2
)

# Common Settings
EPOCHS=200
RUNS=25
BATCH=256
DATASET="MNIST"
MODEL="MLP"
ACTIVATION="relu"

# 2. Build Commands
for seed in "${SEEDS[@]}"; do
    for opt in "${OPTIMIZERS[@]}"; do
        for layer in "${LAYERS_TO_FREEZE[@]}"; do
            
            # Construct a suffix for the experiment name to keep wandb organized
            # e.g., "adam_freeze_fc1" or "sgd_freeze_None"
            base_name="${opt}_freeze_${layer}"

            # --- A. Classic (No Reg) ---
            COMMANDS+=("python3 implicit_regularization.py \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --name ${base_name}_classic --exp_name exp1_freeze")

            # --- B. Reset Model ---
            COMMANDS+=("python3 implicit_regularization.py --reset_model \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --name ${base_name}_reset --exp_name exp1_freeze")

            # --- C. L2 Regularization ---
            COMMANDS+=("python3 implicit_regularization.py \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --reg l2 --l2_lambda 1e-3 \
                --name ${base_name}_l2 --exp_name exp1_freeze")

            # --- D. Spectral Regularization ---
            COMMANDS+=("python3 implicit_regularization.py \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --reg spectral --spectral_lambda 1e-3 \
                --name ${base_name}_spectral --exp_name exp1_freeze")

            # --- E. Wasserstein Regularization ---
            COMMANDS+=("python3 implicit_regularization.py \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --reg wass --wass_lambda 1e-3 \
                --name ${base_name}_wass --exp_name exp1_freeze")

            # --- F. Normalize Gradients ---
            COMMANDS+=("python3 implicit_regularization.py \
                --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET --freeze_layer $layer \
                --normalize_gradients \
                --name ${base_name}_gradnorm --exp_name exp1_freeze")

        done
        
        # --- Baseline (Control) ---
        COMMANDS+=("python3 implicit_regularization.py \
            --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
            --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
            --lr 1e-3 --dataset $DATASET \
            --hessian_cap_type none \
            --name ${opt}_control --exp_name exp2_hessian")

        # --- A. Hard Cap (Edge of Stability) ---
        # Limit eta * lambda <= 2.0 (Theoretical limit)
        COMMANDS+=("python3 implicit_regularization.py \
            --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
            --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
            --lr 1e-3 --dataset $DATASET \
            --hessian_cap_type hard --hessian_cap_val 2.0 \
            --name ${opt}_hardcap_2.0 --exp_name exp2_hessian")

        # Limit eta * lambda <= 1.0 (Very stable)
        COMMANDS+=("python3 implicit_regularization.py \
            --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
            --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
            --lr 1e-3 --dataset $DATASET \
            --hessian_cap_type hard --hessian_cap_val 1.0 \
            --name ${opt}_hardcap_1.0 --exp_name exp2_hessian")

        # --- B. Soft Cap (Penalty Term) ---
        # Loss += 0.01 * Lambda_max
        COMMANDS+=("python3 implicit_regularization.py \
            --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
            --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
            --lr 1e-3 --dataset $DATASET \
            --hessian_cap_type soft --hessian_cap_val 0.01 \
            --name ${opt}_softcap_0.01 --exp_name exp2_hessian")

        # Loss += 0.1 * Lambda_max
        COMMANDS+=("python3 implicit_regularization.py \
            --seed $seed --model $MODEL --activation $ACTIVATION --runs $RUNS \
            --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
            --lr 1e-3 --dataset $DATASET \
            --hessian_cap_type soft --hessian_cap_val 0.1 \
            --name ${opt}_softcap_0.1 --exp_name exp2_hessian")
        
        for freeze in "${LAYERS_TO_FREEZE[@]}"; do
            for cap in "${CAP_TYPES[@]}"; do
                
                # Determine value based on cap type
                CAP_VAL=0.0
                if [ "$cap" == "hard" ]; then CAP_VAL=2.0; fi
                if [ "$cap" == "soft" ]; then CAP_VAL=0.01; fi

                EXP_NAME="${opt}_F-${freeze}_C-${cap}"
                
                COMMANDS+=("python3 implicit_regularization.py \
                    --seed $seed --model $MODEL --runs $RUNS \
                    --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                    --lr 1e-3 --dataset $DATASET \
                    --freeze_layer $freeze \
                    --hessian_cap_type $cap --hessian_cap_val $CAP_VAL \
                    --name $EXP_NAME --exp_name exp3_combo")
            done
        done

        for config in "${SCHED_CONFIGS[@]}"; do
            clean_name=$(echo $config | tr ':,' '-')
            EXP_NAME="${opt}_${clean_name}"

            COMMANDS+=("python3 implicit_regularization.py \
                --seed 1 --model $MODEL --runs $RUNS \
                --batch_size $BATCH --epochs $EPOCHS --optimizer $opt \
                --lr 1e-3 --dataset $DATASET \
                --layer_sched_config \"$config\" \
                --name $EXP_NAME --exp_name exp4_scheduling")
        done
    done
done

# 3. Execution Loop
for cmd in "${COMMANDS[@]}"; do
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset PIDS[$i]
                break
            fi
        done
        sleep 1
        PIDS=("${PIDS[@]}")
    done

    echo "Starting: $cmd"
    
    # Extract name for log file
    exp_name=$(echo "$cmd" | grep -oP '(--name[ =][^ ]+)' | head -n1 | awk '{print $2}')
    mkdir -p logs
    timestamp=$(date +%s)
    log_file="logs/${exp_name}_${timestamp}.log"
    
    nohup $cmd > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for experiments to finish..."
wait
echo "Experiment 1 Completed."