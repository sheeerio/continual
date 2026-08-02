# Seconds-long CPU check: the sequential GPUBatches eval path must emit exactly
# the same batch sequence as DataLoader(shuffle=False), and must consume no RNG.
# This is what makes J / effective_rank provably unchanged by fix (b), without
# needing a GPU run to compare a downstream scalar.
import torch, random
from torch.utils import data
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
base = datasets.MNIST("/home/gbaveja/data", train=True, download=False, transform=tf)
sub = data.Subset(base, list(range(2000)))

X = torch.stack([sub[i][0].view(-1) for i in range(len(sub))])
tg = base.targets
Y = tg[torch.as_tensor(list(sub.indices))]


def seq_batches(bs):
    for i in range(0, X.shape[0], bs):
        yield X[i:i + bs], Y[i:i + bs]


bs = 256
dl = data.DataLoader(sub, batch_size=bs, shuffle=False)

st = torch.random.get_rng_state()
nb, ok_x, ok_y = 0, True, True
for (xa, ya), (xb, yb) in zip(dl, seq_batches(bs)):
    nb += 1
    ok_x &= torch.equal(xa.view(xa.size(0), -1), xb)
    ok_y &= torch.equal(ya, yb)
rng_same = torch.equal(st, torch.random.get_rng_state())

print(f"batches={nb} (expected {(len(sub)+bs-1)//bs})")
print(f"images identical : {ok_x}")
print(f"labels identical : {ok_y}")
print(f"rng untouched    : {rng_same}")
print("VERDICT:", "PASS" if (ok_x and ok_y and rng_same and nb == (len(sub)+bs-1)//bs) else "FAIL")
