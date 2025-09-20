from config import get_parser
import math

config = get_parser().parse_args()

# total_steps  = config.epochs * math.ceil(len(train_dataset) / config.batch_size)
total_steps = config.epochs * (20100 // config.batch_size)
total_tokens = total_steps * config.batch_size


def skew_lambda(step):
    t = step / total_steps
    p = config.skew_peak_frac
    if t < p:
        return t / p
    else:
        return max((1 - t) / (1 - p), 0.0)


def wsd_lambda(step):
    n = step * config.batch_size
    Nw = config.wsd_warmup_tokens
    Nd = int(total_tokens * config.wsd_decay_proportion)
    if n < Nw:
        return n / Nw
    elif n <= total_tokens - Nd:
        return 1.0
    else:
        return max(0.0, (total_tokens - n) / Nd)


def power_lambda(step):
    power_alpha = config.lr * (config.power_warmup_tokens ** (config.power_exponent))
    n = step * config.batch_size
    Nw = config.power_warmup_tokens
    Nd = int(total_tokens * config.power_decay_proportion)
    eta_pw = lambda t: min(
        power_alpha * (t ** (-config.power_exponent)),
        config.lr,
    )
    if n < Nw:
        return (n / Nw) * (eta_pw(Nw) / config.lr)
    else:
        return eta_pw(n) / config.lr
    # else:
    #     return ((total_tokens - n) / Nd) * (eta_pw(total_tokens - Nd) / config.lr)
