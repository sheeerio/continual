# How much RNG does DataLoader(shuffle=False) consume? The eval block creates
# TWO iterators per task (the J loop, then next(iter(...)) for effective_rank).
import torch
from torch.utils import data
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
base = datasets.MNIST("/home/gbaveja/data", train=True, download=False, transform=tf)
sub = data.Subset(base, list(range(2000)))
dl = data.DataLoader(sub, batch_size=256, shuffle=False)


def draws(fn):
    torch.manual_seed(0)
    fn()
    after = torch.randn(1).item()
    # how many draws to burn from a fresh seed to reach the same value
    for k in range(0, 8):
        torch.manual_seed(0)
        for _ in range(k):
            torch.empty((), dtype=torch.int64).random_()
        if abs(torch.randn(1).item() - after) < 1e-12:
            return k
    return -1


print("full iteration        :", draws(lambda: [None for _ in dl]))
print("next(iter(dl))        :", draws(lambda: next(iter(dl))))
print("old eval block (both) :", draws(lambda: ([None for _ in dl], next(iter(dl)))))
