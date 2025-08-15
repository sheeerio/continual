#!/bin/bash

# Define the number of parallel jobs you want to run.
# This should be balanced between your CPU cores and GPU memory.
# Start with a conservative number (e.g., 2-4) and increase if your GPU utilization allows.
MAX_PARALLEL_JOBS=4 # Adjust this based on your RTX 3070 Ti's memory and performance

# List of commands to run
COMMANDS=(
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-2_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --name=wass_1e-3_ply_f --exp_name sigma --lr 1e-3 --dataset=MNIST --lr_schedule pl_lyapunov "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --name=wass_1e-3_ly_f --exp_name sigma --lr 1e-3 --dataset=MNIST --lr_schedule lyapunov"
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --name=wass_1e-3_ly_f --exp_name sigma --lr 1e-3 --dataset=MNIST --lr_schedule lyapunov"
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-2_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --name=wass_1e-3_ply_f --exp_name sigma --lr 1e-3 --dataset=MNIST --lr_schedule pl_lyapunov "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-3 --name=spectral_1e-3_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-4 --name=spectral_1e-4_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 1 --activation=crelu --runs=10 --name=crelu_f --exp_name sigma --epochs 100 --lr=1e-3 --dataset=MNIST"
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-3 --name=spectral_1e-3_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-4 --name=spectral_1e-4_f --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --activation=crelu --runs=10 --name=crelu_fmaw --exp_name sigma --epochs 100 --lr=1e-3 --dataset=MNIST"

    "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --exp_name sigma --name ortho_frob_f --epochs=100"
    "python3 implicit_regularization.py --seed 2 --model MLP --reg orthofrob --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --exp_name sigma --name ortho_frob_f --epochs=100"

    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-2_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-2_f --exp_name sigma "
    
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-3_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-2_f --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-2_f --exp_name sigma "

    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name relu_f_1e-4 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name relu_f_1e-4 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name relu_f_1e-2 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name relu_f_1e-2 --exp_name sigma "
    
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-3_f_1e-4 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-3_f_1e-4 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name l2_1e-3_f_1e-2 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name l2_1e-3_f_1e-2 --exp_name sigma "

    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f_1e-3 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name wass_1e-4_f_1e-2 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name wass_1e-4_f_1e-2 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4 --exp_name sigma "
    "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 10 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4 --exp_name sigma "

    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-3 --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-3 --exp_name sigma --lr=1e-3 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-2 --exp_name sigma --lr=1e-2 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-2 --exp_name sigma --lr=1e-2 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 1 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-4 --exp_name sigma --lr=1e-4 --dataset=MNIST "
    "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=10 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_f_1e-4 --exp_name sigma --lr=1e-4 --dataset=MNIST "
    # "python3 plot.py --type task --name wass_1e-3_f"
    # "python3 plot.py --type task --name wass_1e-3_ply_f"
    # "python3 plot.py --type task --name wass_1e-3_ly_f"
    # "python3 plot.py --type task --name wass_1e-2_f"
    # "python3 plot.py --type task --name relu_f"
    # "python3 plot.py --type task --name l2_1e-3_f"
    # "python3 plot.py --type idx1 --name spectral_1e-3_f"
    # "python3 plot.py --type idx1 --name spectral_1e-2_f"
    # "python3 plot.py --type task --name spectral_1e-4_f"
    # "python3 plot.py --type task --name crelu_f" 

 )

# Array to keep track of background PIDs
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
