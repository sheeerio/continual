import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset
import argparse
import wandb
import matplotlib.pyplot as plt

from collections import deque

W = 10  # Window size for sharpness tracking
K = 20
TRACE_INTERVAL = 200

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
parser.add_argument("--log_interval", type=int, default=400)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--project", type=bool, default=False)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--l2_lambda", type=float, default=0.0)
parser.add_argument("--spectral_lambda", type=float, default=0.0)
parser.add_argument("--spectral_k", type=int, default=2)
parser.add_argument(
    "--reg",
    type=str,
    default="l2",
    choices=["l2", "l2_init", "wass", "spectral", "shrink_perturb"],
)
parser.add_argument("--wass_lambda", type=float, default=0.0)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument(
    "--initialization",
    type=str,
    default="kaiming",
    choices=["kaiming", "xavier", "normal", "uniform"],
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
    "--step_size_schedule",
    type=str,
    default="constant",
    choices=["constant", "linear", "exponential", "polynomial", "cosine"],
)
args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

map = {
    "relu": "relu",
    "leaky_relu": "leaky_relu",
    "tanh": "tanh",
    "identity": "linear",
    "crelu": "relu",
    "adalin": "leaky_relu",
}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)


def randomize_targets(dataset, p):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        subset_indices = dataset.indices
        if isinstance(original_dataset.targets, torch.Tensor):
            targets_list = original_dataset.targets.tolist()
        else:
            targets_list = original_dataset.targets
        n = len(subset_indices)
        k = int(p * n)
        random_subset_idx = torch.randperm(n)[:k]
        for i in random_subset_idx:
            original_idx = subset_indices[i]
            targets_list[original_idx] = random.randint(0, 9)

        if isinstance(original_dataset.targets, torch.Tensor):
            original_dataset.targets = torch.tensor(targets_list)
        else:
            original_dataset.targets = targets_list

        return dataset
    else:
        if isinstance(dataset.targets, torch.Tensor):
            targets_list = dataset.targets.tolist()
        else:
            targets_list = dataset.targets

        n = len(targets_list)
        k = int(p * n)
        idx = torch.randperm(n)[:k]
        for i in idx:
            targets_list[i] = random.randint(0, 9)

        if isinstance(dataset.targets, torch.Tensor):
            dataset.targets = torch.tensor(targets_list)
        else:
            dataset.targets = targets_list
        return dataset


def empirical_fischer_rank(model, dataset, device, thresh=0.99, max_m=100):
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    m = 0
    for x, y in loader:
        if m >= max_m:
            break
        if args.model in ["CNN", "BatchNormCNN"]:
            x = x.to(device)
        else:
            x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        model.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        g = torch.cat([p.grad.view(-1) for p in params])
        grads.append(g)
        m += 1
    G = torch.stack(grads)
    M = G @ G.T
    sig = torch.linalg.svdvals(M)
    cumsum = torch.cumsum(sig, dim=0)
    total = cumsum[-1]
    j = (cumsum / total >= thresh).nonzero()[0].item() + 1
    return j / float(m)


def estimate_hessian_topk(model, loss, params, k=5, iters=10):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
    n = flat_grad.numel()

    def hvp(v):
        g_v = (flat_grad * v).sum()
        hv = torch.autograd.grad(g_v, params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1) for h in hv]).detach()

    eigs = []
    vs = []
    for _ in range(k):
        v = torch.randn(n, device=flat_grad.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(iters):
            w = hvp(v)
            for j, u in enumerate(vs):
                w = w - eigs[j] * (u.dot(w)) * u
            v = w / (w.norm() + 1e-12)
        Hv = hvp(v)
        lam = v.dot(Hv).item()
        eigs.append(lam)
        vs.append(v)
    return eigs


def compute_use_for_activation(h):
    with torch.no_grad():
        pos = (h > 0).float().mean(dim=0)
        eps = 1e-12
        ent = -(pos * (pos + eps).log() + (1 - pos) * (1 - pos + eps).log())
        return ent.mean().item()


def power_iteration(W, iters=1):
    v = torch.randn(W.shape[1], device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        v = W.t() @ (W @ v)
        v = v / (v.norm() + 1e-12)
    return (W @ v).norm()


def hessian_trace(loss, params, n_samples=1):
    trace_est = 0.0
    for _ in range(n_samples):
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        v = torch.randn(flat_grad.size(), device=flat_grad.device)
        Hv = torch.autograd.grad(flat_grad.dot(v), params, retain_graph=True)
        Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv])
        trace_est += (flat_grad * Hv_flat).sum().item()
    return trace_est / n_samples


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        if args.activation in ("crelu", "fourier", "cleaky_relu"):
            self.fc2 = nn.Linear(2 * h, h)
            self.fc3 = nn.Linear(2 * h, h)
            self.fc4 = nn.Linear(2 * h, o)
        else:
            self.fc2 = nn.Linear(h, h)
            self.fc3 = nn.Linear(h, h)
            self.fc4 = nn.Linear(h, o)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if args.initialization == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=args.alpha if args.activation == "adalin" else 0,
                        nonlinearity=map[args.activation],
                    )
                elif args.initialization == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif args.initialization == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif args.initialization == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if args.activation == "relu":
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
        elif args.activation == "leaky_relu":
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
        elif args.activation == "tanh":
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
        elif args.activation == "crelu":
            x1 = self.fc1(x)
            x = torch.cat([F.relu(x1), F.relu(-x1)], 1)
            x2 = self.fc2(x)
            x = torch.cat([F.relu(x2), F.relu(-x2)], 1)
            x3 = self.fc3(x)
            x = torch.cat([F.relu(x3), F.relu(-x3)], 1)
        elif args.activation == "cleaky_relu":
            x = self.fc1(x)
            x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
            x = self.fc2(x)
            x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
            x = self.fc3(x)
            x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
        elif args.activation == "adalin":
            x = F.leaky_relu(self.fc1(x), negative_slope=args.alpha)
            x = F.leaky_relu(self.fc2(x), negative_slope=args.alpha)
            x = F.leaky_relu(self.fc3(x), negative_slope=args.alpha)
        elif args.activation == "fourier":
            x = torch.cat(
                [torch.sin(self.fc1(x * 5.0)), torch.cos(self.fc1(x * 5.0))], 1
            )
            x = torch.cat(
                [torch.sin(self.fc2(x * 5.0)), torch.cos(self.fc2(x * 5.0))], 1
            )
            x = torch.cat(
                [torch.sin(self.fc3(x * 5.0)), torch.cos(self.fc3(x * 5.0))], 1
            )
        elif args.activation == "softplus":
            x = F.softplus(self.fc1(x))
            x = F.softplus(self.fc2(x))
            x = F.softplus(self.fc3(x))
        elif args.activation == "swish":
            x = self.fc1(x) * torch.sigmoid(self.fc1(x))
            x = self.fc2(x) * torch.sigmoid(self.fc2(x))
            x = self.fc3(x) * torch.sigmoid(self.fc3(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        return self.fc4(x)


class LayerNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, o)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
        self.ln3 = nn.LayerNorm(h)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class BatchNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.bn1 = nn.BatchNorm1d(h)
        if args.activation in ("crelu", "fourier", "cleaky_relu"):
            self.fc2 = nn.Linear(2 * h, h)
            self.fc3 = nn.Linear(2 * h, h)
            self.fc4 = nn.Linear(2 * h, h)
            self.fc5 = nn.Linear(2 * h, o)
            self.bn2 = nn.BatchNorm1d(2 * h)
            self.bn3 = nn.BatchNorm1d(2 * h)
            self.bn4 = nn.BatchNorm1d(h)
        else:
            self.fc2 = nn.Linear(h, h)
            self.fc3 = nn.Linear(h, h)
            self.fc4 = nn.Linear(h, h)
            self.fc5 = nn.Linear(h, o)
        self.bn2 = nn.BatchNorm1d(h)
        self.bn3 = nn.BatchNorm1d(h)
        self.bn4 = nn.BatchNorm1d(h)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if args.initialization == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=args.alpha if args.activation == "adalin" else 0,
                        nonlinearity=map[args.activation],
                    )
                elif args.initialization == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif args.initialization == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif args.initialization == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if args.activation == "relu":
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.relu(self.bn4(self.fc4(x)))
        elif args.activation == "leaky_relu":
            x = F.leaky_relu(self.bn1(self.fc1(x)))
            x = F.leaky_relu(self.bn2(self.fc2(x)))
            x = F.leaky_relu(self.bn3(self.fc3(x)))
            x = F.leaky_relu(self.bn4(self.fc4(x)))
        elif args.activation == "adalin":
            x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=args.alpha)
            x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=args.alpha)
            x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=args.alpha)
            x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=args.alpha)
        elif args.activation == "crelu":
            x1 = self.fc1(x)
            x = torch.cat([F.relu(self.bn1(x1)), F.relu(-x1)], 1)
            x2 = self.fc2(x)
            x = torch.cat([F.relu(self.bn2(x2)), F.relu(-x2)], 1)
            x3 = self.fc3(x)
            x = torch.cat([F.relu(self.bn3(x3)), F.relu(-x3)], 1)
            x4 = self.fc4(x)
            x = torch.cat([F.relu(self.bn4(x4)), F.relu(-x4)], 1)
        elif args.activation == "softplus":
            x = F.softplus(self.bn1(self.fc1(x)))
            x = F.softplus(self.bn2(self.fc2(x)))
            x = F.softplus(self.bn3(self.fc3(x)))
            x = F.softplus(self.bn4(self.fc4(x)))
        elif args.activation == "swish":
            x = self.fc1(x) * torch.sigmoid(self.bn1(self.fc1(x)))
            x = self.fc2(x) * torch.sigmoid(self.bn2(self.fc2(x)))
            x = self.fc3(x) * torch.sigmoid(self.bn3(self.fc3(x)))
            x = self.fc4(x) * torch.sigmoid(self.bn4(self.fc4(x)))
        return self.fc5(x)


class LeakyLayerNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, o)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BatchNormCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        return self.fc2(x)


set_seed(args.seed)

if args.dataset == "MNIST":
    tmp = MNIST(root="../data", train=True, download=True)
    DATA_MEAN = (tmp.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (tmp.data / 255.0).std(axis=(0, 1, 2))
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD)]
    )
    full = MNIST(root="../data", train=True, download=True, transform=tf)
    perm = torch.randperm(len(full))[:25600]
    train_dataset = Subset(full, perm)
    test_dataset = MNIST(root="../data", train=False, download=True, transform=tf)
    in_ch = 1
    input_size = 28 * 28
elif args.dataset == "CIFAR10":
    tmp = CIFAR10(root="../data", train=True, download=True)
    DATA_MEAN = (tmp.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (tmp.data / 255.0).std(axis=(0, 1, 2))
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(tuple(DATA_MEAN.tolist()), tuple(DATA_STD.tolist())),
        ]
    )
    full = CIFAR10(root="../data", train=True, download=True, transform=tf)
    perm = torch.randperm(len(full))[:38400]
    train_dataset = Subset(full, perm)
    test_dataset = CIFAR10(root="../data", train=False, download=True, transform=tf)
    in_ch = 3
    input_size = 3 * 32 * 32
elif args.dataset == "PermutedMNIST":
    base_full = MNIST(root="../data", train=True, download=True)
    DATA_MEAN = (base_full.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (base_full.data / 255.0).std(axis=(0, 1, 2))
    torch.manual_seed(args.seed)
    subsample_idx = torch.randperm(len(base_full))[:51200]
    base = Subset(base_full, subsample_idx)
    num_tasks = args.runs
    perms = [torch.randperm(28 * 28) for _ in range(num_tasks)]

    def make_perm_tf(t):
        perm = perms[t]
        return transforms.Lambda(lambda x: x.view(-1)[perm].view(1, 28, 28))

    base.dataset = base_full
    base.indices = subsample_idx
    train_dataset = base
    test_dataset = MNIST(
        root="../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(DATA_MEAN, DATA_STD),
                transforms.Lambda(lambda x: x),
            ]
        ),
    )
    in_ch = 1
    input_size = 28 * 28
elif args.dataset == "Shuffle_CIFAR":
    tmp = CIFAR10(root="../data", train=True, download=True)
    DATA_MEAN = (tmp.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (tmp.data / 255.0).std(axis=(0, 1, 2))
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(tuple(DATA_MEAN.tolist()), tuple(DATA_STD.tolist())),
        ]
    )

    full_train = CIFAR10(root="../data", train=True, download=True, transform=tf)
    subset_indices = torch.randperm(len(full_train))[:5000]
    train_subset = Subset(full_train, subset_indices)
    if isinstance(full_train.targets, torch.Tensor):
        orig_labels = full_train.targets[subset_indices].tolist()
    else:
        orig_labels = [full_train.targets[i] for i in subset_indices]

    test_dataset = CIFAR10(root="../data", train=False, download=True, transform=tf)
    in_ch, input_size = 3, 3 * 32 * 32
elif args.dataset == "Tiny_ImageNet":
    from torchvision.datasets import ImageNet

    DATA_MEAN = (0.485, 0.456, 0.406)
    DATA_STD = (0.229, 0.224, 0.225)
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DATA_MEAN, DATA_STD),
        ]
    )
    full = ImageNet(root="../data", split="train", download=True, transform=tf)
    perm = torch.randperm(len(full))[:128000]
    train_dataset = Subset(full, perm)
    test_dataset = ImageNet(root="../data", split="val", download=True, transform=tf)
    in_ch, input_size = 3, 3 * 224 * 224

args.alpha = 0.01 if args.activation == "leaky_relu" else args.alpha
hidden = 256
if args.model == "MLP":
    model = MLP(input_size, hidden, 10).to(device)
elif args.model == "LayerNormMLP":
    model = LayerNormMLP(input_size, hidden, 10).to(device)
elif args.model == "BatchNormMLP":
    model = BatchNormMLP(input_size, hidden, 10).to(device)
elif args.model == "LeakyLayerNormMLP":
    model = LeakyLayerNormMLP(input_size, hidden, 10).to(device)
elif args.model == "LeakyKaimingLayerNormMLP":
    model = LeakyKaimingLayerNormMLP(input_size, hidden, 10).to(device)
elif args.model == "LinearNet":
    model = nn.Sequential(nn.Flatten(), nn.Identity()).to(device)
elif args.model == "CNN":
    model = CNN(in_ch).to(device)
elif args.model == "BatchNormCNN":
    model = BatchNormCNN(in_ch).to(device)
else:
    model = MLP(input_size, hidden, 10).to(device)

init_params = {
    n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
}
if args.reg == "wass":
    for n, p0 in init_params.items():
        init_params[n] = torch.sort(p0.view(-1))[0].to(device)

activations = {}


def save_activations(name):
    def hook(m, inp, out):
        activations[name] = out

    return hook


if args.model in [
    "MLP",
    "LayerNormMLP",
    "BatchNormMLP",
    "LeakyLayerNormMLP",
    "LeakyKaimingLayerNormMLP",
]:
    model.fc1.register_forward_hook(save_activations("l1"))
    model.fc2.register_forward_hook(save_activations("l2"))
    model.fc3.register_forward_hook(save_activations("l3"))
else:
    model.fc1.register_forward_hook(save_activations("l1"))

criterion = nn.CrossEntropyLoss()
wd = args.l2_lambda if args.reg == "l2" else 0.0
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=wd)

#
run = wandb.init(
    project=f"random_label_{args.dataset}",
    entity="sheerio",
    group=args.exp_name,
    name=args.name,
    config={
        "model": args.model,
        "dataset": args.dataset,
        "activation": args.activation,
        "reg": args.reg,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "random_seed": args.seed,
        "l2_lambda": args.l2_lambda,
        "spectral_lambda": args.spectral_lambda,
        "spectral_k": args.spectral_k,
        "wass_lambda": args.wass_lambda,
    },
)

results = {
    args.activation: {
        "batch_error": [],
        "param_norm": [],
        "update_norm": [],
        "effective_rank": [],
        "dormancy": [],
    }
}
for task in range(10, 10 + args.runs):
    sharp_queue = deque(maxlen=W)
    current_runlen = 0
    step_within_task = 0
    collapse_step_within_task = None

    if args.dataset == "PermutedMNIST":
        perm_tf = make_perm_tf(task - 10)
        train_dataset.dataset.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD), perm_tf]
        )
        loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
    elif args.dataset == "Shuffle_CIFAR":
        mapping = torch.randperm(10).tolist()
        remapped = [mapping[orig] for orig in orig_labels]
        for idx, new_lbl in zip(subset_indices, remapped):
            train_subset.dataset.targets[idx] = new_lbl
        train_dataset = train_subset
        loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
    else:
        train_dataset = randomize_targets(train_dataset, task / 10)
        loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

    total_updates = 0

    model.train()
    sum_up = 0.0
    this_task_acc = 0.0
    this_normalized_sharp = 0.0
    for _ in range(args.epochs):
        if total_updates % args.log_interval == 0:
            hessian_rank = empirical_fischer_rank(model, train_dataset, device)
        for x, y in loader:
            if args.model in ["CNN", "BatchNormCNN"]:
                inputs = x.to(device)
            else:
                inputs = x.view(x.size(0), -1).to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            preds = out.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            this_task_acc += acc
            base = criterion(out, labels)
            reg = torch.tensor(0.0, device=device)
            if args.reg == "l2_init":
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += (p - init_params[n]).pow(2).sum()
                reg *= args.l2_lambda
            elif args.reg == "wass":
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        reg += (torch.sort(p.view(-1))[0] - init_params[n]).pow(2).sum()
                reg *= args.wass_lambda
            elif args.reg == "spectral":
                for n, p in model.named_parameters():
                    if p.requires_grad and p.ndim >= 2:
                        reg += (power_iteration(p, 1).pow(args.spectral_k) - 1.0).pow(2)
                reg *= args.spectral_lambda
            loss = base + reg
            params = [p for p in model.parameters() if p.requires_grad]
            old = [p.data.clone() for p in params]

            if total_updates % TRACE_INTERVAL == 0:
                trace_val = hessian_trace(loss, params, n_samples=10)

            if total_updates % args.log_interval == 0:
                eigs = estimate_hessian_topk(model, loss, params, k=5)
                sharpness = eigs[0]

                v_squares = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        state = optimizer.state[p]
                        if "exp_avg_sq" in state:
                            v_sq = state["exp_avg_sq"].detach()
                            v_squares.append(v_sq.view(-1))
                if len(v_squares) > 0:
                    v_cat = torch.cat(v_squares)
                    rms = torch.sqrt(v_cat.mean() + 1e-16)
                    lr0 = optimizer.param_groups[0]["lr"]
                    alpha_agg = lr0 / (rms + optimizer.param_groups[0]["eps"])
                else:
                    alpha_agg = optimizer.param_groups[0]["lr"]

                norm_sharpness = sharpness * alpha_agg
                this_normalized_sharp += norm_sharpness

                sharp_queue.append(norm_sharpness)

                lam_mean = sum(sharp_queue) / len(sharp_queue)
                lam_std = (
                    sum((x - lam_mean) ** 2 for x in sharp_queue) / len(sharp_queue)
                ) ** 0.5
                frac_above = sum(1 for x in sharp_queue if x > 2.0) / len(sharp_queue)

                if norm_sharpness > 2.0:
                    current_runlen += 1
                else:
                    current_runlen = 0

                if collapse_step_within_task is None and current_runlen >= K:
                    collapse_step_within_task = step_within_task

            loss.backward()
            optimizer.step()
            # shrink perturb
            if args.reg == "shrink_perturb":
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.mul_(1.0 - args.sp_weight_decay)
                        p.data.add_(args.sp_noise_std * torch.randn_like(p.data))

            delta = torch.cat(
                [(p.data - o).view(-1).abs() for p, o in zip(params, old)]
            )
            up_norm = delta.mean().item()
            sum_up += up_norm
            total_updates += 1
            if total_updates % args.log_interval == 0:
                # use_vals = { f"use_{name}": compute_use_for_activation(h) for name, h in activations.items() }
                # avg_use_val = sum(use_vals.values()) / len(use_vals)
                wn = torch.cat([p.data.view(-1).abs() for p in params]).mean().item()
                log = {
                    "acc": acc,
                    "loss": loss.item(),
                    "update_norm": up_norm,
                    "weight_norm": wn,
                    # "average_use_val": avg_use_val,
                    "hessian_rank": hessian_rank
                    if total_updates % args.log_interval == 0
                    else None,
                    "trace_val": trace_val
                    if total_updates % TRACE_INTERVAL == 0
                    else None,
                    "sharpness": sharpness,
                    "norm_sharpness": norm_sharpness,
                    "runlen": current_runlen,  # how many consecutive steps ≥ 2, within this task
                    "frac_above": frac_above,  # fraction of last W steps ≥ 2, within this task
                    "lam_mean": lam_mean,
                    "lam_std": lam_std,
                    # **use_vals,
                    # **hess_avgs,
                }
                # if args.model not in ["CNN","BatchNormCNN"]:
                #     h = activations["l1"]
                #     use1 = compute_use_for_activation(h); log["use_l1"]=use1
                run.log(log)

    model.eval()
    eval_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in eval_loader:
            if args.model in ["CNN", "BatchNormCNN"]:
                inp = x.to(device)
            else:
                inp = x.view(x.size(0), -1).to(device)
            out = model(inp)
            l = criterion(out, y.to(device))
            bs = y.size(0)
            total += l.item() * bs
            count += bs
    J = total / count
    pn = (
        torch.cat(
            [p.data.view(-1).abs() for p in model.parameters() if p.requires_grad]
        )
        .mean()
        .item()
    )
    aun = sum_up / total_updates
    inputs, _ = next(iter(eval_loader))
    inputs = inputs.view(inputs.size(0), -1).to(device)
    h = model(inputs)
    s = torch.linalg.svdvals(h)
    cut = s.sum() * 0.99
    j = (torch.cumsum(s, 0) >= cut).nonzero()[0].item() + 1
    effective_rank = -j / float(h.shape[1])
    run.log(
        {
            "J": J,
            "param_norm": pn,
            "avg_norm_sharp": this_normalized_sharp / (args.epochs * len(loader)),
            "average_update_norm": aun,
            "effective_rank": effective_rank,
            "task_acc": this_task_acc / (args.epochs * len(loader)),
        }
    )
    res = results[args.activation]
    res["batch_error"].append(J)
    res["param_norm"].append(pn)
    res["update_norm"].append(aun)

wandb.finish()

tasks = np.arange(1, len(results[args.activation]["batch_error"]) + 1)
for m in ["batch_error", "param_norm", "update_norm"]:
    plt.figure()
    plt.plot(tasks, results[args.activation][m], label=args.activation)
    plt.xlabel("Task")
    plt.ylabel(m)
    plt.legend()
    plt.show()
