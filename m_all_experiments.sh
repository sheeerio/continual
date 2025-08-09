#!/bin/bash

# Define the number of parallel jobs you want to run.
# This should be balanced between your CPU cores and GPU memory.
# Start with a conservative number (e.g., 2-4) and increase if your GPU utilization allows.
MAX_PARALLEL_JOBS=4 # Adjust this based on your RTX 3070 Ti's memory and performance

# List of commands to run
COMMANDS=(
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=4 --name=3uniform_adalin7 --alpha=0.7 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=adalin --runs=4 --name=3uniform_adalin7+l2 --alpha=0.7 --exp_name=partial_adam --reg=l2 --l2_lambda=5e-4"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=4 --name=3uniform_bn+adalin7+l2 --alpha=0.7 --exp_name=partial_adam --reg=l2 --l2_lambda=5e-4"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=4 --name=3uniform_bn+adalin7 --alpha=0.7 --exp_name=partial_adam"
    ## "python3 implicit_regularization.py --seed=2025 --runs=5 --epochs=100 --model=BatchNormMLP --activation=relu --reg=l2 --l2_lambda=1e-4 --name=bn+l2 --exp_name=new"
    ## "python3 implicit_regularization.py --seed=2026 --runs=5 --epochs=100 --model=BatchNormMLP --activation=relu --reg=l2 --l2_lambda=1e-4 --name=bn+l2 --exp_name=new"
    ## "python3 implicit_regularization.py --seed=2027 --runs=5 --epochs=100 --model=BatchNormMLP --activation=relu --reg=l2 --l2_lambda=1e-4 --name=bn+l2 --exp_name=new"
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=4 --name=3uniform_adalin3 --alpha=0.3 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=adalin --runs=4 --name=3uniform_bn+adalin3 --alpha=0.3 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=BatchNormMLP --activation=relu --runs=4 --name=3uniform_bn --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --initialization=kaiming --model=MLP --activation=crelu --runs=40 --name=3uniform_crelu --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --initialization=kaiming --model=BatchNormMLP --activation=crelu --runs=4 --name=3uniform_bn+crelu --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_s_p --exp_name=partial_adam --reg=shrink_perturb --sp_weight_decay 1e-5 --sp_noise_std 1e-6"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=2 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=1e-5 --name=3uniform_wass=5 --exp_name=partial_adam"
    # "python3 implicit_regularization.py --seed=2025 --activation=softplus --runs=4 --name=3uniform_softplus+l2 --exp_name='partial_adam' --reg=l2 --l2_lambda=3e-2"
    # "python3 implicit_regularization.py --seed=2025 --activation=swish --runs=4 --name=3uniform_swish+l2 --exp_name='partial_adam' --reg=l2 --l2_lambda=3e-2"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_relu+sam --exp_name='partial_adam' --sam"
    # "python3 implicit_regularization.py --seed=2025 --activation=adalin --runs=4 --name=3uniform_adalin5 --alpha=0.5 --exp_name=adam --lr=0.0032"

    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_relu_1e-3 --exp_name=eos --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 -name=relu_3e-2 --exp_name=sgd --dataset=MNIST --lr=0.0001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_relu_1e-5 --exp_name=sgd --dataset=MNIST --lr=0.00001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_relu_1e-6 --exp_name=sgd --dataset=MNIST --lr=0.000001 --epochs=250"

    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_wass_3e-2 --exp_name=sgd --dataset=MNIST --lr=0.01 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_wass_1e-3 --exp_name=sgd --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=adalin --alpha=0.5 --runs=4 --name=3uniform_adalin5_1e-3_reset --exp_name=lr --dataset=MNIST --lr=0.0001 --epochs=250 "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3+wsd_sched --exp_name=lr --dataset=MNIST --lr=0.0001 --epochs=250 --lr_schedule=wsd"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3+power_sched --exp_name=lr --dataset=MNIST --lr=0.001 --epochs=250 --lr_schedule=power"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_relu_3e-2+skew_sched --exp_name=eos --dataset=MNIST --lr=0.000008 --epochs=250 --lr_schedule=skew --skew_peak_frac=0.4"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass_1e-3 --exp_name=lr --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=l2 --l2_lambda=0.0035 --name=3uniform_l2_1e-3 --exp_name=eos --dataset=MNIST --lr=0.001 --epochs=250"
    # "python3 implicit_regularization.py --optimizer=sgd --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_wass_1e-5 --exp_name=sgd --dataset=MNIST --lr=0.00001 --epochs=250"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=l2 --l2_lambda=5e-4 --name=3uniform_l2_1e-3 --exp_name=cu --dataset=MNIST --lr=0.000008 --epochs=250 --lr_schedule=skew --skew_peak_frac=0.4"


    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=4 --exp_name=cu --epochs=1500 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3uniform_wass=4 --exp_name=cu --epochs=250 --lr=3e-2 --dataset=MNIST"

    # bad init experiment
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_std_normal_base --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_std_normal --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=0std=0.01_base --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=0std=0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=0std=3_base --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=0std=3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0_reset  --ns=0. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0  --ns=0. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST"
    
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0.25_reset  --ns=0.25 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0.25  --ns=0.25 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0._reset5  --ns=0.5 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0.5  --ns=0.5 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0.75_reset  --ns=0.75 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_0.75  --ns=0.75 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3ns_1_reset  --ns=1. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=lya  --ns=1. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --lr_schedule lyapunov"

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=4 --name=3ns_0_reset  --ns=0. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=4 --name=3ns_0  --ns=0. --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST "
    
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=4 --name=3ns_0.25_reset  --ns=0.25 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=4 --name=3ns_0.25  --ns=0.25 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=4 --name=3ns_0.5_reset  --ns=0.5 --exp_name=ns --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=3_base --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=3 --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=1_base --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=1 --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=3 --normal_std=3 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=1_base --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3snormal_m=0std=1 --normal_std=1 --initialization=normal --exp_name=scratch_v --epochs=200 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_m=0std=5_base --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_m=0std=5 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_m=10std=5_base --uniform_b=10 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_m=10std=5 --uniform_b=10 --uniform_b=5 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST "
    
#    "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_0.01_base --uniform_a=-0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST  --reset_model"
 #   "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_0.01 --uniform_a=-0.01 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST "

  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_3_base --uniform_a=-3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST  --reset_model"
   # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_3 --uniform_a=-3 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST "

  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_10+-0.01_base --uniform_a=9.99 --uniform_b=10.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST  --reset_model"
  #  "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_10+-0.01 --uniform_a=9.99 --uniform_b=10.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --dataset=MNIST "

   # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_xavier_base --initialization=xavier --exp_name=scratch_v --epochs=250 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=4 --name=3uniform_xavier --initialization=xavier --exp_name=scratch_v --epochs=250 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=1_base --uniform_b=10 --uniform_b=1 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=1 --uniform_b=10 --uniform_b=1 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=0.01_base --uniform_b=10 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=0.01 --uniform_b=10 --uniform_b=0.01 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST "
    
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=3_base --uniform_b=10 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --name=3uniform_m=10std=3 --uniform_b=10 --uniform_b=3 --initialization=uniform --exp_name=scratch_v --epochs=250 --lr=1e-3 --dataset=MNIST"

    # under papa experimental
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST --reset_model "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=l2 --l2_lambda=1e-4 --name=3l2 --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=new --epochs=250 --lr=1e-3 --dataset=MNIST --reset_model "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3wass_250 --exp_name=new --epochs=250 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=4 --reg=wass --wass_lambda=3e-2 --name=3wass_baseline_250 --exp_name=new --epochs=250 --lr=1e-3 --dataset=MNIST --reset_model "
    # "python3 implicit_regularization.py --seed=0 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-4 --name=spectral_random_epoch_seed0_1e-4 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-4 --name=spectral_random_epoch_seed0_1e-4_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=0 --epochs=100 --model=MLP --activation=relu --runs=15 --name=relu_seed0 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=250 --model=MLP --activation=relu --runs=5 --reg=spectral --spectral_lambda=1e-3 --name=spectral_1e-3_random_epoch_seed0_250 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=250 --model=MLP --activation=relu --runs=5 --reg=spectral --spectral_lambda=1e-2 --name=spectral_1e-2_random_epoch_seed0_250 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=250 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_250_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=0 --epochs=200 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_500 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=200 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_500_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=0 --epochs=750 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_750 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=750 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_750_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=0 --epochs=1000 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_1k --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=0 --epochs=1000 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral_random_epoch_seed0_1k_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"


    # "python3 implicit_regularization.py --seed=1 --model=MLP --activation=relu --runs=15 --reg=wass --wass_lambda=3e-2 --name=wass_random_epoch_seed1 --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=1 --model=MLP --activation=relu --runs=15 --reg=wass --wass_lambda=3e-2 --name=wass_random_epoch_seed1_reset --exp_name=new --lr=1e-3 --dataset=MNIST --reset_model"
    # tp slides
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=M_relu_reset --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=wadk --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --model=MLP --activation=relu --runs=5 --name=wadk --exp_name=new --epochs=100 --lr=1e-3 --dataset=CIFAR10 "
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=mlp --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=mlp_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --epochs=100 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2_ly --exp_name=new --lr=1e-3 --dataset=MNIST --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --seed=2026 --epochs=100 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2 --exp_name=new --lr=1e-3 --dataset=MNIST" 
    # "python3 implicit_regularization.py --seed=2027 --epochs=100 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2 --exp_name=new --lr=1e-3 --dataset=MNIST" 
    # "python3 implicit_regularization.py --seed=2025 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST --reset_model" 
    # "python3 implicit_regularization.py --model=BatchNormMLP --seed=2025 --activation=relu --runs=5 --name=batchnorm --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "  
    # "python3 implicit_regularization.py --model=BatchNormMLP --seed=2026 --activation=relu --runs=5 --name=batchnorm --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "  
    # "python3 implicit_regularization.py --model=BatchNormMLP --seed=2027 --activation=relu --runs=5 --name=batchnorm --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "  
    # "python3 implicit_regularization.py --model=BatchNormMLP --seed=2025 --activation=relu --runs=5 --name=batchnorm_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST --reset_model"
    # "python3 implicit_regularization.py --seed=2025 --activation=crelu --runs=5 --name=crelu --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST " 
    # "python3 implicit_regularization.py --seed=2026 --activation=crelu --runs=5 --name=crelu --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST " 
    # "python3 implicit_regularization.py --seed=2027 --activation=crelu --runs=5 --name=crelu --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST " 
    # "python3 implicit_regularization.py --seed=2025 --activation=crelu --runs=5 --name=crelu_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST --reset_model" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-3_2 --epochs=100 --ortho_lambda=1e-3"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-1_2 --epochs=100 --ortho_lambda=1e-1"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-2_2 --epochs=100 --ortho_lambda=1e-2"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-4_2 --epochs=100 --ortho_lambda=1e-4"

    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-3_0.5 --epochs=100 --ortho_lambda=1e-3 --ortho_frac=0.5"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_5e-1_0.5 --epochs=100 --ortho_lambda=5e-1 --ortho_frac=0.5"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-1_0.5 --epochs=100 --ortho_lambda=1e-1 --ortho_frac=0.5"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-2_0.5 --epochs=100 --ortho_lambda=1e-2 --ortho_frac=0.5"

    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-3_2 --epochs=100 --ortho_lambda=1e-3 --ortho_frac=2"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_5e-1_2 --epochs=100 --ortho_lambda=5e-1 --ortho_frac=2"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-1_2 --epochs=100 --ortho_lambda=1e-1 --ortho_frac=2"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg ortho --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_1e-2_2 --epochs=100 --ortho_lambda=1e-2 --ortho_frac=2"

    
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_frob --epochs=100"
    # "python3 implicit_regularization.py --seed 2 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name ortho_frob --epochs=100"

    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name orthofrob_1e-1_1 --epochs=100 --ortho_interval=1 --ortho_lambda=1e-1 --ortho_frac=1"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name orthofrob_1e-2_1 --epochs=100 --ortho_interval=1 --ortho_lambda=1e-2 --ortho_frac=1"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name orthofrob_1e-3_1 --epochs=100 --ortho_interval=1 --ortho_lambda=1e-3 --ortho_frac=1"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name orthofrob_1e-4_1 --epochs=100 --ortho_interval=1 --ortho_lambda=1e-4 --ortho_frac=1"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 5 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name orthofrob_1e-5_1 --epochs=100 --ortho_interval=1 --ortho_lambda=1e-5 --ortho_frac=1"

    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 --name=mlp --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=mlp --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=5 --name=mlp --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=mlp_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2026 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2 --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST " 
    # "python3 implicit_regularization.py --seed=2026 --reg=l2 --l2_lambda=1e-4 --activation=relu --runs=5 --name=l2_r --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST  --reset_model"
    # python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --reg ortho --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name blad --ortho_lambda 
    # wasserstein
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_3" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_3" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_3ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_3ly --lr_schedule lyapunov" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_4" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_4" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_4ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_4ly --lr_schedule lyapunov" 

    # relu
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu" 

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2__5e-4_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2__5e-4_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4" 
    
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4" 

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4" 

    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4" 

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3" 
    "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_reset --reset_model" 
    
    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-2 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-2_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-2 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-2_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-2 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-2" 


    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-1 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-1_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-1 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-1_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 200 --seed 2 --model MLP --reg l2 --l2_lambda 1e-1 --activation relu --runs 100 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-1" 

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5" 

    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_ply --lr_schedule pl_lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5" 


    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4_r --reset_model" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 5e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_5e-4_r --reset_model" 
    
    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_r --reset_model" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-4 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_r --reset_model"

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_r --reset_model" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-3 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-3_r --reset_model" 

    # "python3 implicit_regularization.py --epochs 150 --seed 1 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_r --reset_model" 
    # "python3 implicit_regularization.py --epochs 150 --seed 2 --model MLP --reg l2 --l2_lambda 1e-5 --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-5_r --reset_model" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly --lr_schedule lyapunov" 

    # "python3 implicit_regularization.py --seed 1 --model BatchNormMLP  --runs 20 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name bn+l2"
    # "python3 implicit_regularization.py --seed 2 --model BatchNormMLP  --runs 20 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name bn+l2"

    # "python3 implicit_regularization.py --seed 1 --model BatchNormMLP  --runs 20 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name bn+l2_ly --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 2 --model BatchNormMLP  --runs 20 --reg l2 --l2_lambda 1e-2 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name bn+l2_ly --lr_schedule lyapunov"

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly --lr_schedule lyapunov" 
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly --lr_schedule lyapunov" 

    # relu ab
    # "python3 implicit_regularization.py --seed 2 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_ly --epochs=100 --lr_schedule lyapunov"
    
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_safe_0.9 --lr_schedule lyapunov --ly_safety 0.9" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_safe_0.8 --lr_schedule lyapunov --ly_safety 0.7" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_safe_0.7 --lr_schedule lyapunov --ly_safety 0.5" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_warm_1.1 --lr_schedule lyapunov --ly_warm 1.1" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_warm_1.3 --lr_schedule lyapunov --ly_warm 1.3" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_warm_1.5 --lr_schedule lyapunov --ly_warm 1.5" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_cool_0.9 --lr_schedule lyapunov --ly_cool 0.9" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_cool_0.7 --lr_schedule lyapunov --ly_cool 0.7" 
    # "python3 implicit_regularization.py --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_ly_cool_0.5 --lr_schedule lyapunov --ly_cool 0.5" 

    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass --exp_name new"
    # "python3 implicit_regularization.py --seed 2026 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --epochs 300 --dataset MNIST --name wass --exp_name new"
    # "python3 implicit_regularization.py --seed 2027 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --epochs 300 --dataset MNIST --name wass --exp_name new"
    # "python3 implicit_regularization.py --model=BatchNormMLP --seed=2025 --activation=relu --runs=5 --name=batchnorm --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=2025 --activation=relu --runs=5 -me=ala --exp_name=new --epochs=100 --lr=1e-3 --dataset=MNIST  --lr_schedule=skew"
    # "python3 implicit_regularization.py --epochs=100 --seed=2025 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-3 --name=spectral --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --epochs=100 --seed=2026 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-4 --name=spectral --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --epochs=100 --seed=2027 --model=MLP --activation=relu --runs=15 --reg=spectral --spectral_lambda=1e-4 --name=spectral --exp_name=new --lr=1e-3 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=M_relu_reset --exp_name=new --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"

    # "python3 implicit_regularization.py --seed=2026 --activation=relu --runs=5 --name=M_relu --exp_name=new --epochs=200 --lr=1e-3 --dataset=MNIST "

    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=5 --name=M_relu_reset --exp_name=new --epochs=200 --lr=1e-3 --dataset=MNIST  --reset_model"
    # "python3 implicit_regularization.py --seed=2027 --activation=relu --runs=5 --name=M_relu --exp_name=new --epochs=200 --lr=1e-3 --dataset=MNIST "
    # "python3 implicit_regularization.py --seed 2023 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep100_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2024 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep100_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep100_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2023 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2024 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2023 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2024 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 200 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    
    # "python3 implicit_regularization.py --seed 2023 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2024 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025 --exp_name under"
    # "python3 implicit_regularization.py --seed 2023 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2024 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 5 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-2 --dataset MNIST --name sweeps_relu_ep200_seed2025_reset --exp_name under --reset_model"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass --exp_name under"
    # "python3 implicit_regularization.py --seed 2025 --model BatchNormMLP --activation relu --runs 5 --reg l2 --l2_lambda 1e-2 --batch_size 256 --epochs 100 --optimizer adam --lr 1e-3 --dataset MNIST --name bn --exp_name under"
    # "python3 implicit_regularization.py --seed 2025 --model MLP --activation relu --runs 50 --reg wass --wass_lambda 1e-2 --batch_size 256 --epochs 500 --optimizer adam --lr 1e-3 --dataset MNIST --name sweeps_wass_lr1e-3_lam1e-3_seed2025 --exp_name under"
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
