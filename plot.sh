#!/bin/bash

# Define the number of parallel jobs you want to run.
# This should be balanced between your CPU cores and GPU memory.
# Start with a conservative number (e.g., 2-4) and increase if your GPU utilization allows.
MAX_PARALLEL_JOBS=3 # Adjust this based on your RTX 3070 Ti's memory and performance

# List of commands to run
COMMANDS=(
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu_f --exp_name l2 "
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4 --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4 --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_ss1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param ss1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_scv1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param scv1"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_svar1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param svar1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_sqm1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param sqm1"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_ss10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param ss10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_scv10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param scv10"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_svar10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param svar10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_sqm10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param sqm10"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_ss1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param ss1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_scv1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param scv1" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_svar1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param svar1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_sqm1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param sqm1" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_ss10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param ss10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_scv10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param scv10" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_svar10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param svar10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_sqm10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param sqm10" 

    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rs1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rs1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rcv1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rcv1"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rvar1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rvar1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rqm1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rqm1"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rs10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rs10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rcv10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rcv10"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rvar10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rvar10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly_rqm10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov --param rqm10"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rs1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rs1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rcv1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rcv1" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rvar1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rvar1"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rqm1rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rqm1" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rs10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rs10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rcv10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rcv10" 
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rvar10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rvar10"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply_rqm10rr --exp_name crelu_1e-4 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov --param rqm10" 

    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-3 --epochs=100"
    # "python3 implicit_regularization.py --seed 2 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-3 --epochs=100"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-4 --epochs=100"
    # "python3 implicit_regularization.py --seed 2 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-4 --epochs=100"
    # "python3 implicit_regularization.py --seed 1 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-4 --epochs=100"
    # "python3 implicit_regularization.py --seed 2 --model MLP --reg orthofrob --activation relu --runs 20 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --exp_name l2 --name ortho_frob_f_1e-4 --epochs=100"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer clamped_adam --lr 1e-3 --dataset MNIST --name l2_1e-4_ca --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f --exp_name l2 "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"

    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"

    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_s ize 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --epochs 250 --seed 2 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3_250 --exp_name wass_again"
    # "python3 implicit_regularization.py --epochs 250 --seed 3 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3_250 --exp_name wass_again"
    # "python3 implicit_regularization.py --epochs 250 --seed 2 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3_ply_sqm10rrl_250_2 --exp_name wass_again --lr_schedule pl_lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 250 --seed 3 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3_ply_sqm10rrl_250_3 --exp_name wass_again --lr_schedule pl_lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 500 --seed 2 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3ll_500 --exp_name wass_again" 
    # "python3 implicit_regularization.py --epochs 500 --seed 3 --model MLP --activation relu --runs 40 --reg wass --wass_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_f_1e-3ll_500 --exp_name wass_again"
    # "python3 x_implicit_regularization.py --epochs 500 --seed 1 --model MLP --activation crelu --runs 40 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name crelu_f_1e-2_500_reset --exp_name crelu_1e-2 --reset_model"
    # "python3 x_implicit_regularization.py --epochs 500 --seed 2 --model MLP --activation crelu --runs 40 --batch_size 256 --optimizer adam --lr 1e-2 --dataset MNIST --name crelu_f_1e-2_500_reset --exp_name crelu_1e-2 --reset_model"
    "python3 implicit_regularization.py --epochs 250 --seed 2023 --model MLP --activation relu --reg l2 --l2_lambda 1e-4 --runs 20 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_1e-4_250 --exp_name l2_1e-4"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 40 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu --exp_name sweeps"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 40 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu --exp_name sweeps"
    # "python3 implicit_regularization.py --epochs 100 --seed 3 --model MLP --activation relu --runs 40 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name relu --exp_name sweeps"

    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_ss10 --exp_name l2 --lr_schedule pl_lyapunov --param ss10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_scv10 --exp_name l2 --lr_schedule pl_lyapunov --param scv10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_sqm10 --exp_name l2 --lr_schedule pl_lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_svar10 --exp_name l2 --lr_schedule pl_lyapunov --param svar10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_ss1 --exp_name l2 --lr_schedule pl_lyapunov --param ss1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_scv1 --exp_name l2 --lr_schedule pl_lyapunov --param scv1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_sqm1 --exp_name l2 --lr_schedule pl_lyapunov --param sqm1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_svar1 --exp_name l2 --lr_schedule pl_lyapunov --param svar1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_ss10 --exp_name l2 --lr_schedule lyapunov --param ss10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_scv10 --exp_name l2 --lr_schedule lyapunov --param scv10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_sqm10 --exp_name l2 --lr_schedule lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_svar10 --exp_name l2 --lr_schedule lyapunov --param svar10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_ss1 --exp_name l2 --lr_schedule lyapunov --param ss1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_scv1 --exp_name l2 --lr_schedule lyapunov --param scv1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_sqm1 --exp_name l2 --lr_schedule lyapunov --param sqm1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_svar1 --exp_name l2 --lr_schedule lyapunov --param svar1"

    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rs10 --exp_name l2 --lr_schedule pl_lyapunov --param rs10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rcv10 --exp_name l2 --lr_schedule pl_lyapunov --param rcv10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rqm10 --exp_name l2 --lr_schedule pl_lyapunov --param rqm10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rvar10 --exp_name l2 --lr_schedule pl_lyapunov --param rvar10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rs1 --exp_name l2 --lr_schedule pl_lyapunov --param rs1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rcv1 --exp_name l2 --lr_schedule pl_lyapunov --param rcv1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rqm1 --exp_name l2 --lr_schedule pl_lyapunov --param rqm1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rvar1 --exp_name l2 --lr_schedule pl_lyapunov --param rvar1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rs10 --exp_name l2 --lr_schedule lyapunov --param rs10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rcv10 --exp_name l2 --lr_schedule lyapunov --param rcv10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rqm10 --exp_name l2 --lr_schedule lyapunov --param rqm10"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rvar10 --exp_name l2 --lr_schedule lyapunov --param rvar10"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rs1 --exp_name l2 --lr_schedule lyapunov --param rs1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rcv1 --exp_name l2 --lr_schedule lyapunov --param rcv1"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rqm1 --exp_name l2 --lr_schedule lyapunov --param rqm1"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rvar1 --exp_name l2 --lr_schedule lyapunov --param rvar1"


    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_ss10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param ss10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_scv10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param scv10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_sqm10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_svar10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param svar10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_ss1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param ss1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_scv1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param scv1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_sqm1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param sqm1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_svar1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param svar1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_ss10_s0.9 --exp_name l2 --lr_schedule lyapunov --param ss10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_scv10_s0.9 --exp_name l2 --lr_schedule lyapunov --param scv10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_sqm10_s0.9 --exp_name l2 --lr_schedule lyapunov --param sqm10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_svar10_s0.9 --exp_name l2 --lr_schedule lyapunov --param svar10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_ss1_s0.9 --exp_name l2 --lr_schedule lyapunov --param ss1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_scv1_s0.9 --exp_name l2 --lr_schedule lyapunov --param scv1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_sqm1_s0.9 --exp_name l2 --lr_schedule lyapunov --param sqm1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_svar1_s0.9 --exp_name l2 --lr_schedule lyapunov --param svar1"

    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rs10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rs10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rcv10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rcv10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rqm10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rqm10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rvar10_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rvar10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rs1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rs1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rcv1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rcv1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rqm1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rqm1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ply_rvar1_s0.9 --exp_name l2 --lr_schedule pl_lyapunov --param rvar1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rs10_s0.9 --exp_name l2 --lr_schedule lyapunov --param rs10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rcv10_s0.9 --exp_name l2 --lr_schedule lyapunov --param rcv10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rqm10_s0.9 --exp_name l2 --lr_schedule lyapunov --param rqm10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rvar10_s0.9 --exp_name l2 --lr_schedule lyapunov --param rvar10"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rs1_s0.9 --exp_name l2 --lr_schedule lyapunov --param rs1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rcv1_s0.9 --exp_name l2 --lr_schedule lyapunov --param rcv1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rqm1_s0.9 --exp_name l2 --lr_schedule lyapunov --param rqm1"
    # "python3 implicit_regularization.py --epochs 100 --safety 0.9 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name l2_f_1e-4_ly_rvar1_s0.9 --exp_name l2 --lr_schedule lyapunov --param rvar1"


    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg l2 --l2_lambda 1e-3 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name l2_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov"


    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ly --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule lyapunov"
    # "python3 implicit_regularization.py --seed 1 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"
    # "python3 implicit_regularization.py --seed 2 --optimizer adam --activation=crelu --runs 20 --name=crelu_f_1e-4_ply --exp_name l2 --epochs 100 --lr=1e-4 --dataset=MNIST --lr_schedule pl_lyapunov"

    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 5e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ly --exp_name l2 --lr_schedule lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-3 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 1 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "
    # "python3 implicit_regularization.py --epochs 100 --seed 2 --model MLP --activation relu --runs 20 --reg wass --wass_lambda 1e-4 --batch_size 256 --optimizer adam --lr 1e-4 --dataset MNIST --name wass_1e-4_f_1e-4_ply --exp_name l2 --lr_schedule pl_lyapunov "

    # only clamped
    # "python3 plot.py --type task --name wass_1e-4_f"
    # "python3 implicit_regularization.py --seed 2 --epochs 100 --model=MLP --activation=relu --runs=25 --reg=spectral --spectral_lambda=1e-4 --name=spectral_1e-4_f_1e-4_100x25 --exp_name l2 --lr=1e-4 --dataset=MNIST"
    # "python3 plot.py --type task --name wass_1e-4_ply_f"
    # "python3 plot.py --type task --name wass_1e-4_ly_f"
    # "python3 plot.py --type task --name wass_1e-4_f"
    # "python3 plot.py --type task --name relu_f"
    # "python3 plot.py --type task --name l2_1e-4_f"
    # "python3 plot.py --type idx1 --name spectral_1e-4_f"
    # "python3 plot.py --type idx1 --name spectral_1e-4_f"
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
        sleep 20 # Avoid busy-waiting
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
