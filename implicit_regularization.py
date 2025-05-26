import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms
import argparse
import wandb
import matplotlib.pyplot as plt

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
    ],
)
parser.add_argument(
    "--activation",
    type=str,
    default="relu",
    choices=["relu", "leaky_relu", "tanh", "identity", "crelu", "fourier", "adalin"],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--randomize_percent", type=float, default=0.0)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--log_interval", type=int, default=400)
parser.add_argument("--project", type=bool, default=False)
parser.add_argument("--name", type=str, default="nameless")
parser.add_argument("--alpha",type=float,default=0.5,
    help="mixing weight for α-linearization: phi(z)=alpha*z + (1-alpha)*ReLU(z)")
args = parser.parse_args()

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

def randomize_targets(dataset, p):
    n = len(dataset.targets)
    k = int(p * n)
    idx = torch.randperm(n)[:k]
    for i in idx:
        dataset.targets[i] = random.randint(0, 9)
    return dataset

P = torch.randn(36, 28*28) / (36**0.5)

def stochastic_project(x):
    x = x.view(-1)
    return P @ x

def empirical_fischer_rank(model, dataset, device, thresh=0.99, max_m=100):
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    m = 0
    for x, y in loader:
        if m >= max_m:
            break
        x, y = x.view(x.size(0), -1).to(device), y.to(device)
        model.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        g = torch.cat([p.grad.view(-1) for p in params])
        grads.append(g)
        m += 1
        torch.cuda.empty_cache()
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

class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        if args.activation == "crelu" or args.activation == "fourier":
            self.fc2 = nn.Linear(2 * h, h)
            self.fc3 = nn.Linear(2 * h, h)
            self.fc4 = nn.Linear(2 * h, o)
        else:
            self.fc2 = nn.Linear(h, h)
            self.fc3 = nn.Linear(h, h)
            self.fc4 = nn.Linear(h, o)
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
            x1_out = self.fc1(x)
            x = torch.cat([F.relu(x1_out), F.relu(-x1_out)], dim=1)
            x2_out = self.fc2(x)
            x = torch.cat([F.relu(x2_out), F.relu(-x2_out)], dim=1)
            x3_out = self.fc3(x)
            x = torch.cat([F.relu(x3_out), F.relu(-x3_out)], dim=1)
        elif args.activation == "adalin":
            x = args.alpha * self.fc1(x) + (1 - args.alpha) * F.relu(self.fc1(x))
            x = args.alpha * self.fc2(x) + (1 - args.alpha) * F.relu(self.fc2(x))
            x = args.alpha * self.fc3(x) + (1 - args.alpha) * F.relu(self.fc3(x))
        elif args.activation == "fourier":
            x = torch.cat([torch.sin(self.fc1(x*5.0)), torch.cos(self.fc1(x*5.0))], dim=1)
            x = torch.cat([torch.sin(self.fc2(x*5.0)), torch.cos(self.fc2(x*5.0))], dim=1)
            x = torch.cat([torch.sin(self.fc3(x*5.0)), torch.cos(self.fc3(x*5.0))], dim=1)
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
        with torch.no_grad():
            self.ln1.bias.fill_(0)
            self.ln2.bias.fill_(0)
            self.ln3.bias.fill_(0)
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

class BatchNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, o)
        self.bn1 = nn.BatchNorm1d(h)
        self.bn2 = nn.BatchNorm1d(h)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

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

class LeakyKaimingLayerNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, o)
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, a=0.01, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.fc2.weight, a=0.01, nonlinearity="leaky_relu")
    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

set_seed(args.seed)
train_dataset = MNIST(root="../data", train=True, download=True)
DATA_MEAN = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)])
input_size = 28*28
h = 512
train_dataset = MNIST(root="../data", train=True, download=True, transform=tf)
test_dataset = MNIST(root="../data", train=False, download=True, transform=tf)

if args.model == "MLP":
    model = MLP(input_size, h, 10).to(device)
elif args.model == "LayerNormMLP":
    model = LayerNormMLP(input_size, h, 10).to(device)
elif args.model == "BatchNormMLP":
    model = BatchNormMLP(input_size, h, 10).to(device)
elif args.model == "LeakyLayerNormMLP":
    model = LeakyLayerNormMLP(input_size, h, 10).to(device)
elif args.model == "LeakyKaimingLayerNormMLP":
    model = LeakyKaimingLayerNormMLP(input_size, h, 10).to(device)
elif args.model == "LinearNet":
    from torch.nn import Identity
    model = nn.Sequential(nn.Flatten(), Identity()).to(device)
else:
    model = MLP(input_size, h, 10).to(device)

activations = {}
def save_activations(name):
    def hook(m, inp, out):
        activations[name] = out
    return hook

model.fc1.register_forward_hook(save_activations("l1"))
model.fc2.register_forward_hook(save_activations("l2"))
model.fc3.register_forward_hook(save_activations("l3"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
run = wandb.init(
    project="random_label_MNIST",
    entity="sheerio",
    group="continual",
    name=args.name,
    config={
        "activation": args.activation,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "random_seed": args.seed,
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
EPOCHS = 200

for i in range(10, 10 + args.runs):
    train_dataset_c = randomize_targets(train_dataset, i / 10)
    train_loader = data.DataLoader(
        train_dataset_c,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(15, os.cpu_count()),
    )
    hessian_rank = empirical_fischer_rank(model, train_dataset_c, device)
    run.log({"Hessian_rank": hessian_rank})
    model.train()
    total_updates = 0
    sum_update_norm = 0.0
    for _ in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            params = [p for p in model.parameters() if p.requires_grad]
            old_params = [p.data.clone() for p in params]
            if total_updates % args.log_interval == 0:
                eigs = estimate_hessian_topk(model, loss, params, k=5)
                hess_avg = sum(eigs) / len(eigs)
                hess_avgs = {"Hessian_avg": hess_avg}
            loss.backward()
            optimizer.step()
            deltas = torch.cat([(p.data - old).view(-1).abs() for p, old in zip(params, old_params)])
            update_norm = deltas.mean().item()
            sum_update_norm += update_norm
            total_updates += 1
            if total_updates % args.log_interval == 0:
                all_w = torch.cat([p.data.view(-1).abs() for p in params])
                weight_norm = all_w.mean().item()
                h_in = inputs
                if args.activation == "relu":
                    h = F.relu(model.fc1(h_in))
                    h = F.relu(model.fc2(h))
                    h = F.relu(model.fc3(h))
                elif args.activation == "leaky_relu":
                    h = F.leaky_relu(model.fc1(h_in))
                    h = F.leaky_relu(model.fc2(h))
                    h = F.leaky_relu(model.fc3(h))
                elif args.activation == "tanh":
                    h = torch.tanh(model.fc1(h_in))
                    h = torch.tanh(model.fc2(h))
                    h = torch.tanh(model.fc3(h))
                elif args.activation == "crelu":
                    h = torch.cat([F.relu(model.fc1(h_in)), F.relu(-model.fc1(h_in))], dim=1)
                    h = torch.cat([F.relu(model.fc2(h)), F.relu(-model.fc2(h))], dim=1)
                    h = torch.cat([F.relu(model.fc3(h)), F.relu(-model.fc3(h))], dim=1)
                elif args.activation == "fourier":
                    h = torch.cat([torch.sin(model.fc1(h_in)), torch.cos(model.fc1(h_in))], dim=1)
                    h = torch.cat([torch.sin(model.fc2(h)), torch.cos(model.fc2(h))], dim=1)
                    h = torch.cat([torch.sin(model.fc3(h)), torch.cos(model.fc3(h))], dim=1)
                elif args.activation == "adalin":
                    h = args.alpha * model.fc1(h_in) + (1 - args.alpha) * F.relu(model.fc1(h_in))
                    h = args.alpha * model.fc2(h) + (1 - args.alpha) * F.relu(model.fc2(h))
                    h = args.alpha * model.fc3(h) + (1 - args.alpha) * F.relu(model.fc3(h))
                else:
                    h = model.fc1(h_in)
                    h = model.fc2(h)
                    h = model.fc3(h)
                s = torch.linalg.svdvals(h)
                cut = s.sum() * 0.99
                j = (torch.cumsum(s, 0) >= cut).nonzero()[0].item() + 1
                rep_norm = j / float(h.shape[1])
                rep_effective_rank = -rep_norm

                use_vals = {f"use_{name}": compute_use_for_activation(h) for name, h in activations.items()}
                run.log({
                    "loss": loss.item(),
                    "update_norm": update_norm,
                    "weight_norm": weight_norm,
                    "rep_effective_rank": rep_effective_rank,
                    **use_vals,
                    **hess_avgs,
                })
    model.eval()
    eval_loader = data.DataLoader(train_dataset_c, batch_size=args.batch_size, shuffle=False, num_workers=1)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
    J = total_loss / total_samples
    params = [p for p in model.parameters() if p.requires_grad]
    all_w = torch.cat([p.data.view(-1).abs() for p in params])
    param_norm = all_w.mean().item()
    average_update_norm = sum_update_norm / total_updates
    inputs, _ = next(iter(eval_loader))
    inputs = inputs.view(inputs.size(0), -1).to(device)
    if args.activation == "relu":
        h = F.relu(model.fc1(inputs))
        h = F.relu(model.fc2(h))
        h = F.relu(model.fc3(h))
    elif args.activation == "leaky_relu":
        h = F.leaky_relu(model.fc1(inputs))
        h = F.leaky_relu(model.fc2(h))
        h = F.leaky_relu(model.fc3(h))
    elif args.activation == "tanh":
        h = torch.tanh(model.fc1(inputs))
        h = torch.tanh(model.fc2(h))
        h = torch.tanh(model.fc3(h))
    elif args.activation == "crelu":
        h = torch.cat([F.relu(model.fc1(inputs)), F.relu(-model.fc1(inputs))], dim=1)
        h = torch.cat([F.relu(model.fc2(h)), F.relu(-model.fc2(h))], dim=1)
        h = torch.cat([F.relu(model.fc3(h)), F.relu(-model.fc3(h))], dim=1)
    elif args.activation == "fourier":
        h = torch.cat([torch.sin(model.fc1(inputs)), torch.cos(model.fc1(inputs))], dim=1)
        h = torch.cat([torch.sin(model.fc2(h)), torch.cos(model.fc2(h))], dim=1)
        h = torch.cat([torch.sin(model.fc3(h)), torch.cos(model.fc3(h))], dim=1)
    elif args.activation == "adalin":
        h = args.alpha * model.fc1(inputs) + (1 - args.alpha) * F.relu(model.fc1(inputs))
        h = args.alpha * model.fc2(h) + (1 - args.alpha) * F.relu(model.fc2(h))
        h = args.alpha * model.fc3(h) + (1 - args.alpha) * F.relu(model.fc3(h))
    else:
        h = model.fc1(inputs)
        h = model.fc2(h)
        h = model.fc3(h)
    s = torch.linalg.svdvals(h)
    cut = s.sum() * 0.99
    j = (torch.cumsum(s, 0) >= cut).nonzero()[0].item() + 1
    effective_rank = -j / float(h.shape[1])
    abs_sum = None
    total_count = 0
    with torch.no_grad():
        for inputs, _ in eval_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            if args.activation == "relu":
                h = F.relu(model.fc1(inputs))
                h = F.relu(model.fc2(h))
                h = F.relu(model.fc3(h))
            elif args.activation == "leaky_relu":
                h = F.leaky_relu(model.fc1(inputs))
                h = F.leaky_relu(model.fc2(h))
                h = F.leaky_relu(model.fc3(h))
            elif args.activation == "tanh":
                h = torch.tanh(model.fc1(inputs))
                h = torch.tanh(model.fc2(h))
                h = torch.tanh(model.fc3(h))
            elif args.activation == "crelu":
                h = torch.cat([F.relu(model.fc1(h_in)), F.relu(-model.fc1(h_in))], dim=1)
                h = torch.cat([F.relu(model.fc2(h)), F.relu(-model.fc2(h))], dim=1)
                h = torch.cat([F.relu(model.fc3(h)), F.relu(-model.fc3(h))], dim=1)
            elif args.activation == "fourier":
                h = torch.cat([torch.sin(model.fc1(h_in)), torch.cos(model.fc1(h_in))], dim=1)
                h = torch.cat([torch.sin(model.fc2(h)), torch.cos(model.fc2(h))], dim=1)
                h = torch.cat([torch.sin(model.fc3(h)), torch.cos(model.fc3(h))], dim=1)
            elif args.activation == "adalin":
                h = args.alpha * model.fc1(inputs) + (1 - args.alpha) * F.relu(model.fc1(inputs))
                h = args.alpha * model.fc2(h) + (1 - args.alpha) * F.relu(model.fc2(h))
                h = args.alpha * model.fc3(h) + (1 - args.alpha) * F.relu(model.fc3(h))
            else:
                h = model.fc1(inputs)
                h = model.fc2(h)
                h = model.fc3(h)
            abs_h = h.abs()
            if abs_sum is None:
                abs_sum = abs_h.sum(dim=0)
            else:
                abs_sum += abs_h.sum(dim=0)
            total_count += h.size(0)
    abs_means = abs_sum / total_count
    eps = 1e-12
    p = abs_means / abs_means.sum()
    entropy = -(p * (p + eps).log()).sum().item()
    dormancy = -entropy
    run.log({
        "J": J,
        "param_norm": param_norm,
        "average_update_norm": average_update_norm,
        "effective_rank": effective_rank,
        "dormancy": dormancy,
    })
    res = results[args.activation]
    res["batch_error"].append(J)
    res["param_norm"].append(param_norm)
    res["update_norm"].append(average_update_norm)
    res["effective_rank"].append(effective_rank)
    res["dormancy"].append(dormancy)

tasks = np.arange(1, len(results[args.activation]["batch_error"]) + 1)
for metric in [
    "batch_error",
    "param_norm",
    "update_norm",
    "effective_rank",
    "dormancy",
]:
    plt.figure()
    plt.plot(tasks, results[args.activation][metric], label=args.activation)
    plt.xlabel("Task")
    plt.ylabel(metric)
    plt.legend()
    plt.show()
