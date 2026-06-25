AX_PARALLEL_JOBS=2
COMMANDS=()

lrs=1e-3
lambdas=1e-3 
seeds=(1)

for seed in "${seeds[@]}"; do
    COMMANDS+=("python3 implicit_regularization.py \
        --seed $seed --model MLP --activation relu --runs 25 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd --exp_name sgd")
    COMMANDS+=("python3 implicit_regularization.py --reset_model \
        --seed $seed --model MLP --activation relu --runs 25 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd_reset --exp_name sgd")
    COMMANDS+=("python3 implicit_regularization.py \
        --seed $seed --model MLP --activation relu --runs 25 \
        --reg l2 --l2_lambda 1e-3 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd_l2 --exp_name sgd")
done

for seed in "${seeds[@]}"; do
    COMMANDS+=("python3 implicit_regularization.py \
        --seed $seed --model MLP --activation relu --runs 25 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd_normgrads --exp_name sgd --normalize_gradients")
    COMMANDS+=("python3 implicit_regularization.py --reset_model \
        --seed $seed --model MLP --activation relu --runs 25 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd_reset_normgrads --exp_name sgd --normalize_gradients")
    COMMANDS+=("python3 implicit_regularization.py \
        --seed $seed --model MLP --activation relu --runs 25 \
        --reg l2 --l2_lambda 1e-3 \
        --batch_size 256 --epochs 200 --optimizer sgd \
        --lr 1e-3 --dataset MNIST \
        --name nomomem_sgd_l2_normgrads --exp_name sgd --normalize_gradients")
done


# ───────────────────── 2.  BatchNormMLP + L2  ──────────────────────────────
# for lam in "${lambdas[@]}"; do
#   COMMANDS+=("python3 implicit_regularization.py \
#     --seed 2025 --model BatchNormMLP --activation relu --runs 50 \
#     --reg l2 --l2_lambda 1e-3 \
#     --batch_size 256 --epochs 500 --optimizer adam \
#     --lr 1e-3 --dataset MNIST \
#     --name sweeps_bn_l2_lr1e-3_wd1e-3_seed2025 --exp_name sweeps")
# done

PIDS=()

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
    exp_name=$(echo "$cmd" | grep -oP '(--name[ =][^ ]+)' | head -n1 | awk '{print $2}')
    mkdir -p logs
    timestamp=$(date +%s)
    log_file="logs/${exp_name}_${timestamp}.log"
    
    nohup $cmd > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for all experiments to finish..."
wait
echo "All experiments completed."
