import torch
import torch.utils.data as data
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset

def get_dataset(config):
    if config.dataset == "MNIST":
        tmp = MNIST(root="../data", train=True, download=True)
        DATA_MEAN = (tmp.data / 255.0).mean(axis=(0, 1, 2))
        DATA_STD = (tmp.data / 255.0).std(axis=(0, 1, 2))
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD)]
        )
        full = MNIST(root="../data", train=True, download=True, transform=tf)
        perm = torch.randperm(len(full))#[:25600]
        train_dataset = Subset(full, perm)
        test_dataset = MNIST(root="../data", train=False, download=True, transform=tf)
        in_ch = 1
        input_size = 28 * 28
    elif config.dataset == "CIFAR10":
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
        perm = torch.randperm(len(full))[:20400]
        train_dataset = Subset(full, perm)
        test_dataset = CIFAR10(root="../data", train=False, download=True, transform=tf)
        in_ch = 3
        input_size = 3 * 32 * 32
    elif config.dataset == "PermutedMNIST":
        base_full = MNIST(root="../data", train=True, download=True)
        DATA_MEAN = (base_full.data / 255.0).mean(axis=(0, 1, 2))
        DATA_STD = (base_full.data / 255.0).std(axis=(0, 1, 2))
        torch.manual_seed(config.seed)
        subsample_idx = torch.randperm(len(base_full))[:51200]
        base = Subset(base_full, subsample_idx)
        num_tasks = config.runs
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
    elif config.dataset == "Shuffle_CIFAR":
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
    elif config.dataset == "Tiny_ImageNet":
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

    return train_dataset, test_dataset, in_ch, input_size, DATA_MEAN, DATA_STD