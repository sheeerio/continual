import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="MLP",
        choices=[
            "MLP",
            "LayerNormMLP",
            "BatchNormMLP",
            "LeakyLayerNormMLP",
            "LeakyKaimingLayerNormMLP",
            "KaimingMLP",
            "LeakyMLP",
            "LinearNet",
            "CNN",
            "BatchNormCNN",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "PermutedMNIST", "Shuffle_CIFAR", "Tiny_ImageNet"],
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=[
            "relu",
            "leaky_relu",
            "tanh",
            "identity",
            "crelu",
            "fourier",
            "adalin",
            "cleaky_relu",
            "softplus",
            "swish",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize_percent", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--project", type=bool, default=False)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l2_lambda", type=float, default=0.0)
    parser.add_argument("--spectral_lambda", type=float, default=1e-4)
    parser.add_argument("--spectral_k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--beta_schedule", action="store_true", help="Use beta schedule for Adam optimizer")
    parser.add_argument("--reset_model", action="store_true", help="Reset model at the start of each run")
    parser.add_argument("--random_length", action="store_true", help="Randomize task lengths")
    parser.add_argument("--reset_optimizer", action="store_true", help="Reset optimizer state at the start of each run")
    parser.add_argument(
        "--ns",
        type=float,
        default=1.0,
        help="Fraction of targets to randomize aka non-stationarity (default: 1.0, full randomization)",
    )
    parser.add_argument(
        "--reg",
        type=str,
        default="l2",
        choices=["l2", "l2_init", "wass", "spectral", "shrink_perturb", "ortho", "orthofrob"],
    )
    parser.add_argument("--wass_lambda", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    # Wsd scheduler hyperparameters
    parser.add_argument(
        "--wsd_warmup_tokens",
        type=int,
        default=0,
        help="Number of tokens over which to linearly warm up",
    )
    parser.add_argument(
        "--wsd_decay_proportion",
        type=float,
        default=0.15,
        help="Fraction of total tokens used for the final decay phase (default: 0.1)",
    )

    # Power scheduler hyperparameters
    parser.add_argument(
        "--power_max_lr",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--power_exponent",
        type=float,
        default=0.5,
        help="Decay exponent p for Power scheduler (see Eq (7))",
    )
    parser.add_argument(
        "--power_warmup_tokens",
        type=int,
        default=1_000,
        help="Tokens over which to warm up before applying pure Power decay (default: 1 B)",
    )
    parser.add_argument(
        "--power_decay_proportion",
        type=float,
        default=0.0,
        help="Fraction of total tokens for final exponential decay stage",
    )
    parser.add_argument(
        "--skew_peak_frac",
        type=float,
        default=0.4,
        help="Fraction of total steps at which LR peaks (for skew schedule)"
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default="kaiming",
        choices=["kaiming", "xavier", "normal", "uniform"],
    )
    parser.add_argument(
        "--normal_mean",
        type=float,
        default=0.0,
        help="Mean for normal initialization (default: 0.0)",
    )
    parser.add_argument(
        "--normal_std",
        type=float,
        default=1.0,
        help="Standard deviation for normal initialization (default: 1.0)",
    )
    parser.add_argument(
        "--uniform_a",
        type=float,
        default=-0.1,
        help="Lower bound for uniform initialization (default: -0.1)",
    )
    parser.add_argument(
        "--uniform_b",
        type=float,
        default=0.1,
        help="Upper bound for uniform initialization (default: 0.1)",
    )
    parser.add_argument(
        "--sp_weight_decay",
        type=float,
        default=0.0,
        help="Shrink factor (lambda) for shrink-and-perturb (weight decay per step)",
    )
    parser.add_argument(
        "--sp_noise_std",
        type=float,
        default=0.0,
        help="Standard deviation (gamma) of Gaussian noise for shrink-and-perturb",
    )
    parser.add_argument(
        "--sam",
        action="store_true",
        help="Use Sharpness-Aware Minimization (SAM) for training",
    )
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.025,
        help="Radius for SAM perturbation",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "step", "linear", "exponential", "polynomial", "cosine", "wsd", "power", "skew", "lyapunov", "pl_lyapunov"],
        help="Type of learning‐rate schedule",
    )
    parser.add_argument(
        "--final_lr",
        type=float,
        default=0.0,
        help="Final learning rate for linear schedule (default: 0.0)",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="(for step schedule) number of epochs between drops"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="(for step & exponential) decay factor"
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="(for polynomial) power degree"
    )
    
    parser.add_argument("--ly_window", type=int, default=30)
    parser.add_argument("--ly_safety", type=float, default=0.8)
    parser.add_argument("--ly_cool",   type=float, default=0.95)
    parser.add_argument("--ly_warm",   type=float, default=1.05)

    parser.add_argument("--ortho_lambda", type=float, default=1e-3)
    parser.add_argument("--ortho_frac", type=float, default=2)
    parser.add_argument("--ortho_interval", type=int, default=1)

    grp = parser.add_argument_group("Per-layer Lyapunov")
    grp.add_argument("--pl_lyap", action="store_true",
                    help="Activate per-layer Lyapunov LR controller")
    # tweakables – exposed so you can play in wandb sweeps
    grp.add_argument("--pl_lyap_safety", type=float, default=0.8)
    grp.add_argument("--pl_lyap_cool",   type=float, default=0.9)
    grp.add_argument("--pl_lyap_warm",   type=float, default=1.05)
    grp.add_argument("--pl_lyap_iters",  type=int,   default=20,
                    help="Power-iter steps for sharpness on each layer")

    return parser
