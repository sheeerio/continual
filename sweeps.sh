###############################################################################
# SWEEPS EXPERIMENT
###############################################################################
MAX_PARALLEL_JOBS=4     # keep your GPU happy
COMMANDS=()

lrs=1e-4
seeds=(1 2 3)  # for reproducibility

# ──────────────────────── 1.  MLP + Wasserstein  ───────────────────────────
for seed in "${seeds[@]}"; do
    COMMANDS+=("python3 implicit_regularization.py \
        --seed $seed --model MLP --activation relu --runs 50 \
        --batch_size 256 --epochs 500 --optimizer adam \
        --lr 1e-4 --dataset MNIST \
        --name sweeps_relu_lr_1e-4 --exp_name sweeps")
done

PIDS=()

for cmd in "${COMMANDS[@]}"; do
    # Check if we have reached the maximum number of parallel jobs
    while [[ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]]; do
        # Wait for any background job to finish
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                # Job has finished, remove its PID
                unset PIDS[$i]
                break
            fi
        done
        sleep 1 # Avoid busy-waiting
        PIDS=("${PIDS[@]}") # Re-index array
    done

    echo "Starting: $cmd"
    # Run the command in the background and store its PID
    # Redirect output to separate log files for each run
    exp_name=$(echo "$cmd" | grep -oP '(--name[ =][^ ]+)' | head -n1 | awk '{print $2}')
    mkdir -p logs
    timestamp=$(date +%s)
    log_file="logs/${exp_name}_${timestamp}.log"
    
    nohup $cmd > "$log_file" 2>&1 &
    PIDS+=($!) # Add the PID of the last background command
done

# Wait for all remaining background jobs to complete
echo "Waiting for all experiments to finish..."
wait
echo "All experiments completed."
