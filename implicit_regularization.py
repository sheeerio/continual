import numpy as np
import random
import os
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import torch.optim as optim
from torchvision.datasets import CIFAR10, MNIST
import argparse

import wandb
wandb.login()

DATASET_PATH = "../data"
device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

args = argparse.ArgumentParser()
args.add_argument("--dataset", type=str, default="MNIST")
args.add_argument("--model", type=str, default="LayerNormMLP", choices=["MLP", "KaimingMLP", "LeakyMLP", "LeakyKaimingMLP", "LayerNormMLP", "LeakyLayerNormMLP", "LeakyKaimingLayerNormMLP"])
args.add_argument("--seed", type=int, default=0)
args.add_argument("--weight_decay", type=float, default=0.0)
args.add_argument("--randomize_percent", type=float, default=0.0)
args.add_argument("--epochs", type=int, default=10)
args.add_argument("--batch_size", type=int, default=512)
args.add_argument("--dropout", type=float, default=0.0)
args.add_argument("--initialization", type=str, default="none")
args.add_argument("--log_interval", type=int, default=100)
args = args.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def randomize_targets(dataset, percent):
    num_samples = len(dataset.targets)
    num_to_randomize = int(percent * num_samples)
    indices = torch.randperm(num_samples)[:num_to_randomize]
    for idx in indices:
        dataset.targets[idx] = random.randint(0, 9) 
    dataset.targets = dataset.targets
    return dataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        return x


class KaimingMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KaimingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class LeakyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LeakyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x

class LeakyKaimingMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LeakyKaimingMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0.01, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x

class LayerNormMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LayerNormMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = self.fc3(x)
        return x

class LeakyLayerNormMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LeakyLayerNormMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.ln1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.ln2(x))
        x = self.fc3(x)
        return x

class LeakyKaimingLayerNormMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LeakyKaimingLayerNormMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data, a=0.01, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data, a=0.01, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.ln1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.ln2(x))
        x = self.fc3(x)
        return x

def train(run, model, train_loader, criterion, optimizer, target_loss=0.01):
    model.train()
    # epoch = 0
    steps = 0
    optimizer.param_groups[0]['lr'] = 1e-3
    log_interval = 100  # steps

    while steps < 7500:
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % args.log_interval == 0:
                log_dict = {
                    "loss": loss.item(),
                    # "step": steps,
                    # "lr": optimizer.param_groups[0]['lr']
                }
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        log_dict[f"param_norm/{name}"] = param.data.norm().item()
                        if param.grad is not None:
                            log_dict[f"grad_norm/{name}"] = param.grad.norm().item()

                with torch.no_grad():
                    hidden_activations = F.relu(model.fc1(inputs))
                    dead_units = (hidden_activations.sum(dim=0) == 0).float().mean().item()
                    entropy = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)).item()
                    logit_norm = outputs.norm().item()
                    feature_norm = hidden_activations.norm().item()
                    log_dict["feature_norm"] = feature_norm
                    log_dict["logit_norm"] = logit_norm
                    log_dict["entropy"] = entropy
                    log_dict["dead_unit_fraction"] = dead_units

                run.log(log_dict)

            steps += 1


        # if avg_loss <= target_loss:
        #     print("Target loss reached. Stopping training.")
        #     break
        # epoch += 1

if __name__ == "__main__":
    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    train_dataset = MNIST(root=DATASET_PATH, train=True, download=True)
    DATA_MEAN = (train_dataset.data/255.).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data/255.).std(axis=(0,1,2))
    # print("Data mean", DATA_MEAN)
    # print("DATA std", DATA_STD)

    train_transform = transforms.Compose([
    # transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ])

    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)
    # test_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=test_transform)
    train_dataset = MNIST(root=DATASET_PATH, train=True, download=True, transform=train_transform)
    test_dataset = MNIST(root=DATASET_PATH, train=False, download=True, transform=test_transform)
    set_seed(args.seed)

    # model = MLP(28*28*3, 512, 10)
    if args.model == "MLP":
        model = MLP(28*28, 512, 10).to(device)
    elif args.model == "LeakyKaimingMLP":
        model = LeakyKaimingMLP(28*28, 512, 10).to(device)
    elif args.model == "LayerNormMLP":
        model = LayerNormMLP(28*28, 512, 10).to(device)
    elif args.model == "LeakyLayerNormMLP":
        model = LeakyLayerNormMLP(28*28, 512, 10).to(device)
    elif args.model == "LeakyKaimingLayerNormMLP":
        model = LeakyKaimingLayerNormMLP(28*28, 512, 10).to(device)
    elif args.model == "KaimingMLP":
        model = KaimingMLP(28*28, 512, 10).to(device)
    elif args.model == "LeakyMLP":
        model = LeakyMLP(28*28, 512, 10).to(device)
    else:
        raise ValueError("Invalid model type. Choose from ['MLP', 'LeakyKaimingMLP', 'LayerNormMLP', 'LeakyLayerNormMLP']")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-5, betas=(0.9, 0.85))
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    run = wandb.init(
        project="implicit_regularization",
        entity="sheerio",
        group="continual",
        name=f"{args.model}",
        config={
            "randomize_percent": 1.0,
            "epochs": 10,
            "batch_size": 512,
            "dataset": "MNIST",
            "model": "MLP",
            "random_seed": args.seed,
            "dropout": 0.0,
            "initialization": "none",
            "weight_decay": args.weight_decay,
        },
        # reinit=True
    )
    for i in range(10, 10+args.epochs, 1):
        print(f"Randomizing {min(i+1, 10)}0% of the dataset")
        train_dataset_c = randomize_targets(train_dataset, i/10)
        train_set, val_set = torch.utils.data.random_split(train_dataset_c, [60000, 0])
        print("Train set size", len(train_set))
        print("Validation set size", len(val_set))
        print("Test set size", len(test_dataset))

        num_workers = min(os.cpu_count(), 15)
        train_loader = data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=num_workers)
        val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=1)
        test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=1)

        train(run, model, train_loader, criterion, optimizer)
        # if i > 20:
        #     run.finish()