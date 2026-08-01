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
    parser.add_argument("--spectral_k", type=float, default=2)
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
        choices=["l2", "l2_init", "wass", "spectral", "shrink_perturb", "ortho", "orthofrob", "parseval", "l2_loss", "none"],
    )
    parser.add_argument("--wass_lambda", type=float, default=0.0)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "clamped_adam"])
    parser.add_argument("--param", type=str, default="sqm10")
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
        help="Shrink factor applied once per task boundary (theta <- (1-sp_weight_decay)*theta)",
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
    
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--safety", type=float, default=0.8)
    parser.add_argument("--cool",   type=float, default=0.95)
    parser.add_argument("--warm",   type=float, default=1.005)

    parser.add_argument("--ortho_lambda", type=float, default=1e-3)
    parser.add_argument("--ortho_frac", type=float, default=2)
    parser.add_argument("--ortho_interval", type=int, default=1)

    parser.add_argument("--adaptive_reg", action="store_true", help="Enable 1/tau regularization")
    parser.add_argument("--adaptive_scope", type=str, default="local", choices=["local", "global"], help="Scope for adaptive regularization")
    parser.add_argument("--adaptive_type", type=str, default="l2", choices=["l2", "spectral", "wass", "parseval"], help="Type of adaptive regularization")
    parser.add_argument("--reg_sensitivity", type=float, default=0.001, help="Scaling factor for inverse tau penalty")
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--parseval_lambda", type=float, default=0.0)
    parser.add_argument("--adaptive_scale", type=str, default="inv", choices=["inv","saturating","centered"], help="inv: factor = sensitivity/tau. saturating: sensitivity*(1+kappa*g). centered: sensitivity*(1+kappa*(g-0.5)) -- same g, but mean factor stays ~sensitivity for any kappa, so sat_kappa controls the SPREAD of the coefficient rather than its level (saturating shifts the level by ~(1+kappa/2), which moves the basin instead of reshaping it).")
    parser.add_argument("--sat_kappa", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--track_coherence", action="store_true", help="Track cross-layer tau correlation structure")
    parser.add_argument("--coherence_window", type=int, default=100, help="Rolling window size for cross-layer tau coherence tracking")
    parser.add_argument("--diag_csv", type=str, default=None, help="Path to append per-log-step diagnostics (tau, sharpness, eff_lr, alpha_crit_*, coherence) -- survives WANDB_MODE=disabled, which otherwise discards all of it")
    parser.add_argument("--hessian_tol", type=float, default=1e-2, help="RESIDUAL tolerance for power-iteration stopping: stop when ||Hv - lam*v||/|lam| < tol. Gap-independent, unlike the Rayleigh-quotient-delta rule it replaces (that one stopped early exactly where lambda2/lambda1 -> 1). NOTE the units changed with the criterion, so this default is not comparable to the old 1e-3.")
    parser.add_argument("--hessian_max_iters", type=int, default=50, help="Iteration cap for residual-criterion stopping. 20 capped out on 42-46% of calls at the worst-conditioned checkpoints.")
    parser.add_argument("--sigma2_subsample", type=int, default=0, help="Estimate the per-layer within-batch gradient variance from a random N-sample subset instead of all B. 0 = use all B (default, unchanged). This routine runs one autograd.grad per sample, so at B=256 it dominates the diagnostic block.")
    parser.add_argument("--tau_ref_mode", type=str, default="median", choices=["median", "fixed"], help="How tau_ref is computed for the saturating/centered adaptive forms. median (default, behavior-preserving): rolling median of this layer's own tau. fixed: a constant reference frozen after task 0, which makes g sensitive to drift slower than the median window -- under median mode any uniform rescale of tau cancels exactly, so the mechanism cannot see slow drift. Reference-mode logic ported from 539fd61; that commit's competing tau_ref/(tau+tau_ref) factor form is NOT adopted.")
    parser.add_argument("--tau_ref_fixed", type=float, default=None, help="Explicit tau_ref for --tau_ref_mode fixed. If unset, it is calibrated per layer as the mean tau over the last 20%% of task 0 and then frozen.")
    parser.add_argument("--tau_ref_window", type=int, default=100, help="Rolling-median window for tau_ref. Was hardcoded to 100. A task is ~2100 steps, so 100 sees only local fluctuation.")
    parser.add_argument("--tau_update_interval", type=int, default=1, help="Recompute the per-layer curvature estimate every N steps; hold adaptive_factor constant in between. tau is an EMA over ~30 steps and tau_ref a 100-step median, so both already smooth. Default 1 preserves current behavior.")
    parser.add_argument("--curvature_proxy", type=str, default="lam1", choices=["lam1", "top3"], help="Control signal for tau. lam1: top eigenvalue. top3: mean of the top 3 (corrected deflation) -- lower step-to-step CV at comparable estimator noise, ~3.6x the cost per call.")
    parser.add_argument("--hessian_warm_start", action="store_true", help="Carry the top-eigenvector across steps as the power-iteration init in the adaptive path's per-layer estimate_hessian_topk. Default off (cold random init every call, unchanged behavior).")
    parser.add_argument("--hessian_iters", type=int, default=1, help="Power-iteration steps for the ADAPTIVE path's per-layer lam estimate (the tau control signal). Default 1 = previous behavior. At iters=1 the cold estimator's same-batch CV is 5-6x the batch-to-batch curvature CV and the mean is biased low by a layer-dependent factor.")
    parser.add_argument("--ly_eff_lr_floor", type=float, default=0.12, help="Effective-LR floor below which the Lyapunov controllers (LyapunovScheduler, PerLayerLyapunovScheduler) do nothing. Was hardcoded to 0.12 in both; at that value the controller never fired on the calibrated Adam testbed, leaving scheduler arms byte-identical to their no-scheduler twins. Default preserves the previous behavior exactly.")
    parser.add_argument("--diagnostics", type=str, default="full", choices=["off", "light", "full"], help="Cost lever for the observational diagnostic block. off: skip it entirely (no iters=100 hessian topk, no min_eig, no fisher rank, no grad-variance) EXCEPT whatever the active method needs -- adaptive_reg's per-layer iters=1 tau estimate and, for --lr_schedule pl_lyapunov, the alpha_crit inputs the controller consumes. light: per-task end-of-task snapshots only, expensive estimates on a coarse ~10-per-task grid, no per-step series. full: current behavior, every log_interval.")
    parser.add_argument("--taskdiag_csv", type=str, default=None, help="Path to append PER-TASK end-of-task diagnostic snapshots, one row per (task, layer): mean/final tau, mean/final adaptive_factor, last-logged-step grad-variance sigma2, plus run-level mean/final normalized sharpness and coherence mean_off_diag. Long-format detail sink; the layer-collapsed version of these same series also lands as ;-joined trajectory columns on --results_csv")
    parser.add_argument("--diag_interval", type=int, default=None, help="Step interval for the EXPENSIVE diagnostics only (hessian topk/min_eig at iters=100/20, per-sample grad variance at batch_size backprops). Defaults to log_interval if unset, i.e. unchanged behavior; set larger to decouple these from the cheap per-log-interval logging that dominates runtime otherwise")
    parser.add_argument("--diag_task_interval", type=int, default=1, help="Task interval for the empirical Fisher-rank estimate (max_m single-sample backprops each). Default 1 = every task (unchanged behavior); set larger to skip tasks")

    # Continual Backprop (Dohare et al.) -- architectural intervention, NOT a
    # --reg branch: no loss term, no lambda, not part of the coefficient-axis
    # sweep. Reported as its own traditional baseline (AUC/slope vs vanilla).
    parser.add_argument("--use_cbp", action="store_true", help="Enable Continual Backprop (per-unit utility tracking + low-utility reinit)")
    parser.add_argument("--cbp_replacement_rate", type=float, default=1e-4, help="Fraction of ELIGIBLE (mature) units replaced per step, per layer")
    parser.add_argument("--cbp_maturity_threshold", type=int, default=100, help="Steps a unit must age before it's eligible for replacement")
    parser.add_argument("--cbp_decay_rate", type=float, default=0.99, help="EMA decay rate for the per-unit contribution-utility estimate")
    return parser


REG_LAMBDA_FLAGS = {
    "l2_loss": "l2_lambda",
    "l2_init": "l2_lambda",
    "spectral": "spectral_lambda",
    "wass": "wass_lambda",
    "ortho": "ortho_lambda",
    "orthofrob": "ortho_lambda",
    "parseval": "parseval_lambda",
}


def validate_reg_config(config):
    """Fail loud if --reg needs a coefficient that was left at 0.0.

    Exempt --reg none (not in REG_LAMBDA_FLAGS) and any --adaptive_reg run,
    where the coefficient is carried by --reg_sensitivity on the adaptive
    path and lambda == 0.0 on the static path is expected.
    """
    if config.reg == "shrink_perturb":
        if config.sp_weight_decay == 0.0 and config.sp_noise_std == 0.0:
            raise ValueError(
                "--reg 'shrink_perturb' requires --sp_weight_decay and/or "
                "--sp_noise_std to be set (nonzero); got both == 0.0."
            )
        return

    flag = REG_LAMBDA_FLAGS.get(config.reg)
    if flag is None or getattr(config, "adaptive_reg", False):
        return
    if getattr(config, flag) == 0.0:
        raise ValueError(
            f"--reg {config.reg!r} requires --{flag} to be set (nonzero); "
            f"got --{flag}=0.0 and --adaptive_reg is not set."
        )


CBP_SUPPORTED_MODELS = ("MLP", "BatchNormMLP", "LayerNormMLP")
CBP_UNSUPPORTED_ACTIVATIONS = ("crelu", "fourier", "cleaky_relu")


def validate_cbp_config(config):
    """Fail loud if --use_cbp is set on an architecture/activation combo the
    current implementation can't correctly bookkeep.

    CBP tracks utility per hidden unit and needs a clean 1:1 correspondence
    between a hidden layer's output width and the next layer's input width.
    crelu/fourier/cleaky_relu concatenate [act(x), act(-x)] before the next
    Linear, doubling that width, which breaks the unit<->weight-column
    correspondence CBP's reinit logic relies on. Not implemented here.
    """
    if not getattr(config, "use_cbp", False):
        return
    if config.model not in CBP_SUPPORTED_MODELS:
        raise ValueError(
            f"--use_cbp requires --model in {CBP_SUPPORTED_MODELS}, got {config.model!r}."
        )
    if config.activation in CBP_UNSUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"--use_cbp does not support --activation {config.activation!r} "
            f"(concatenation-based activations break CBP's unit<->weight "
            f"correspondence): unsupported = {CBP_UNSUPPORTED_ACTIVATIONS}."
        )
