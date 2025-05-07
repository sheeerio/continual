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

import wandb
wandb.login()

DATASET_PATH = "../data"
device = torch.device("cuda:0") if torch.cuda.is_available else torch.device("cpu")

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
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_uniform__(self.fc1.weight.data, nonlinearity='relu')
        torch.nn.init.kaiming_uniform__(self.fc2.weight.data, nonlinearity='relu')
        # f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        # torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        # torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        # f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        # torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        # torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(run, model, train_loader, criterion, optimizer, target_loss=0.2):
    model.train()
    epoch = 0
    steps = 0
    optimizer.param_groups[0]['lr'] = 0.001
    while steps < 12000:
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.view(inputs.size(0), -1), labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            run.log({"loss": loss.item()})
            steps += 1
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}")

        optimizer.param_groups[0]['lr'] *= 0.995
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss <= target_loss:
            print("Target loss reached. Stopping training.")
            break
        epoch += 1

if __name__ == "__main__":
    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    train_dataset = MNIST(root=DATASET_PATH, train=True, download=True)
    # DATA_MEAN = (train_dataset.data/255.).mean(axis=(0,1,2))
    # DATA_STD = (train_dataset.data/255.).std(axis=(0,1,2))
    # print("Data mean", DATA_MEAN)
    # print("DATA std", DATA_STD)

    train_transform = transforms.Compose([
    # transforms.CenterCrop(28),
    transforms.ToTensor(),
    # transforms.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=DATA_MEAN, std=DATA_STD),
    ])

    # train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)
    # test_dataset = CIFAR10(root=DATASET_PATH, train=False, download=True, transform=test_transform)
    train_dataset = MNIST(root=DATASET_PATH, train=True, download=True, transform=train_transform)
    test_dataset = MNIST(root=DATASET_PATH, train=False, download=True, transform=test_transform)
    set_seed(42)

    # model = MLP(28*28*3, 512, 10)
    model = MLP(28*28, 512, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    run = wandb.init(
        project="implicit_regularization",
        entity="sheerio",
        group="continual",
        config={
            "randomize_percent": 1.0,
            "epochs": 10,
            "batch_size": 128,
            "dataset": "MNIST",
            "model": "MLP",
            "random_seed": 42
        },
        # reinit=True
    )
    for i in range(10, 14, 1):
        # run = wandb.init(
        #     project="implicit_regularization",
        #     entity="sheerio",
        #     group="continual",
        #     config={
        #         "randomize_percent": min(i/10, 1),
        #         "epochs": 10,
        #         "batch_size": 128,
        #         "dataset": "MNIST",
        #         "model": "MLP",
        #         "random_seed": 42
        #     },
        #     # reinit=True
        # )

        print(f"Randomizing {min(i+1, 10)}0% of the dataset")
        train_dataset_c = randomize_targets(train_dataset, i/10)
        train_set, val_set = torch.utils.data.random_split(train_dataset_c, [60000, 0])
        print("Train set size", len(train_set))
        print("Validation set size", len(val_set))
        print("Test set size", len(test_dataset))
        train_loader = data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=1)
        val_loader = data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=1)
        test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=1)

        train(run, model, train_loader, criterion, optimizer)
        # if i > 20:
        #     run.finish()