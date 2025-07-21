import random
import numpy as np
import torch

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

activation_map = {
    "relu": "relu",
    "leaky_relu": "leaky_relu",
    "tanh": "tanh",
    "identity": "linear",
    "crelu": "relu",
    "adalin": "leaky_relu",
    "softplus": "relu",
    "swish": "relu",
}