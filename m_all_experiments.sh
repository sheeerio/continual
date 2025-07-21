#!/bin/bash

# Define the number of parallel jobs you want to run.
# This should be balanced between your CPU cores and GPU memory.
# Start with a conservative number (e.g., 2-4) and increase if your GPU utilization allows.
MAX_PARALLEL_JOBS=4 # Adjust this based on your RTX 3070 Ti's memory and performance

# List of commands to run
COMMANDS=(
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=3 --name=3uniform_adalin7 --alpha=0.7 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=adalin --runs=3 --name=3uniform_adalin7+l2 --alpha=0.7 --exp_name=partial_adam --reg=l2 --l2_lambda=5e-4"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=3 --name=3uniform_bn+adalin7+l2 --alpha=0.7 --exp_name=partial_adam --reg=l2 --l2_lambda=5e-4"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=3 --name=3uniform_bn+adalin7 --alpha=0.7 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=relu --reg=l2 --l2_lambda=3e-2 --runs=3 --name=3uniform_bn+l2 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=3 --name=3uniform_adalin3 --alpha=0.3 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=3 --name=3uniform_bn+adalin3 --alpha=0.3 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=relu --runs=3 --name=3uniform_bn --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --initialization=kaiming --model=MLP --activation=crelu --runs=30 --name=3uniform_crelu --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --initialization=kaiming --model=BatchNormMLP --activation=crelu --runs=3 --name=3uniform_bn+crelu --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_s_p --exp_name=partial_adam --reg=shrink_perturb --sp_weight_decay 1e-5 --sp_noise_std 1e-6"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=2 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=1e-5 --name=3uniform_wass=5 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --activation=softplus --runs=3 --name=3uniform_softplus+l2 --exp_name='partial_adam' --reg=l2 --l2_lambda=3e-2"
    # "python3 implicit_regularization.py --seed=2025 --activation=swish --runs=3 --name=3uniform_swish+l2 --exp_name='partial_adam' --reg=l2 --l2_lambda=3e-2"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_relu+sam --exp_name='partial_adam' --sam"
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=3 --name=3uniform_adalin5 --alpha=0.5 --exp_name=adam --lr=0.0032"

    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_relu_1e-3 --exp_name=eos --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 -name=relu_3e-2 --exp_name=sgd --dataset=MNIST --lr=0.0001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_relu_1e-5 --exp_name=sgd --dataset=MNIST --lr=0.00001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_relu_1e-6 --exp_name=sgd --dataset=MNIST --lr=0.000001 --epochs=250"

    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_wass_3e-2 --exp_name=sgd --dataset=MNIST --lr=0.01 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_wass_1e-3 --exp_name=sgd --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=adalin --alpha=0.5 --runs=3 --name=3uniform_adalin5_1e-3_reset --exp_name=lr --dataset=MNIST --lr=0.0001 --epochs=250 --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3+wsd_sched --exp_name=lr --dataset=MNIST --lr=0.0001 --epochs=250 --lr_schedule=wsd"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3+power_sched --exp_name=lr --dataset=MNIST --lr=0.001 --epochs=250 --lr_schedule=power"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_relu_3e-2+skew_sched --exp_name=eos --dataset=MNIST --lr=0.000008 --epochs=250 --lr_schedule=skew --skew_peak_frac=0.4"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3 --exp_name=lr --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=l2 --l2_lambda=0.0035 --name=3uniform_l2_1e-3 --exp_name=eos --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_wass_1e-5 --exp_name=sgd --dataset=MNIST --lr=0.00001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=l2 --l2_lambda=5e-4 --name=3uniform_l2_1e-3 --exp_name=cu --dataset=MNIST --lr=0.000008 --epochs=250 --lr_schedule=skew --skew_peak_frac=0.4"


    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=4 --exp_name=cu --epochs=1500 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=4 --exp_name=cu --epochs=250 --lr=3e-2 --dataset=MNIST"

    # bad init experiment
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_std_normal_base --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_std_normal --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=0std=0.01_base --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=0std=0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=0std=3_base --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=0std=3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0_reset  --ns=0. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0  --ns=0. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0.25_reset  --ns=0.25 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0.25  --ns=0.25 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0._reset5  --ns=0.5 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0.5  --ns=0.5 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0.75_reset  --ns=0.75 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_0.75  --ns=0.75 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_1_reset  --ns=1. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3ns_1  --ns=1. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=3 --name=3ns_0_reset  --ns=0. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=3 --name=3ns_0  --ns=0. --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=3 --name=3ns_0.25_reset  --ns=0.25 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=3 --name=3ns_0.25  --ns=0.25 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=3 --name=3ns_0.5_reset  --ns=0.5 --exp_name=ns --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=3_base --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=3 --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=1_base --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=1 --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=3 --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=1_base --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3snormal_m=0std=1 --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_m=0std=5_base --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_m=0std=5 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_m=10std=5_base --uniform_b=10 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_m=10std=5 --uniform_b=10 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer"
    
#    "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_0.01_base --uniform_a=-0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer --reset_model"
 #   "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_0.01 --uniform_a=-0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer"

  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_3_base --uniform_a=-3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer --reset_model"
   # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_3 --uniform_a=-3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer"

  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_10+-0.01_base --uniform_a=9.99 --uniform_b=10.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer --reset_model"
  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_10+-0.01 --uniform_a=9.99 --uniform_b=10.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer"

   # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_xavier_base --initialization=xavier --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=3 --name=3uniform_xavier --initialization=xavier --exp_name=scratch_v --epochs=250 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=1_base --uniform_b=10 --uniform_b=1 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=1 --uniform_b=10 --uniform_b=1 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=0.01_base --uniform_b=10 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=0.01 --uniform_b=10 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=3_base --uniform_b=10 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --name=3uniform_m=10std=3 --uniform_b=10 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST"

    # tai papa experimental
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=tai --epochs=250 --lr=1e-3 --dataset=MNIST --reset_model --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=l2 --l2_lambda=1e-3 --name=3l2 --exp_name=tai --epochs=700 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=tai --epochs=250 --lr=1e-3 --dataset=MNIST --reset_model --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3wass_250 --exp_name=tai --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=tai --epochs=250 --lr=1e-3 --dataset=MNIST --reset_model --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=3 --reg=wass --wass_lambda=3e-2 --name=3wass_250 --exp_name=tai --epochs=250 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    
    # tp slides
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=M_relu_reset --exp_name=tai --epochs=1000 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=wadk --exp_name=tai --epochs=100 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=5 --name=wadk --exp_name=tai --epochs=100 --lr=1e-3 --dataset=CIFAR10 --reset_optimizer"
    "python3 implicit_regularization.py --seed=2025 --model=CNN --activation=relu --runs=5 --name=cnn --exp_name=tai --epochs=100 --lr=1e-3 --dataset=MNIST --reset_optimizer"
    "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=ala --exp_name=tai --epochs=100 --lr=1e-3 --dataset=MNIST --reset_optimizer --lr_schedule=wsd"

    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=M_relu_reset --exp_name=tai --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=M_relu --exp_name=tai --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=5 --name=M_relu_reset --exp_name=tai --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=5 --name=M_relu --exp_name=tai --epochs=500 --lr=1e-3 --dataset=MNIST --reset_optimizer"
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
    log_file="logs/$(echo "$cmd" | tr -c '[:alnum:]' '_').log"
    mkdir -p logs
    nohup $cmd > "$log_file" 2>&1 &
    PIDS+=($!) # Add the PID of the last background command
done

# Wait for all remaining background jobs to complete
echo "Waiting for all experiments to finish..."
wait
echo "All experiments completed."
