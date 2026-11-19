"""Train an S4D model on sequential CIFAR-10, as the example.py of the original S4 does.

    uv run --with torchvision example.py

An image is a sequence of its 1024 pixels in row order, and a pixel has 3 features. The model is a
linear encoder, residual S4D blocks with layer norms, a mean over the sequence and a linear decoder.
"""

import argparse
import time

import torch
import torchvision
from torch import nn

from s4ever import S4D, S4Block, param_groups

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--weight_decay", type=float, default=0.01)
p.add_argument("--d_model", type=int, default=128)
p.add_argument("--n_layers", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.1)
args = p.parse_args()
device = torch.accelerator.current_accelerator() or torch.device("cpu")
torch.manual_seed(0)


def load(train):
    """Return the images as (N, 1024, 3) pixel sequences and their labels, both on the device."""
    data = torchvision.datasets.CIFAR10("data", train=train, download=True)
    x = torch.tensor(data.data, device=device) / 255  # (N, 32, 32, 3)
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
    std = torch.tensor((0.2023, 0.1994, 0.2010), device=device)
    return ((x - mean) / std).flatten(1, 2), torch.tensor(data.targets, device=device)


class Model(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.encoder = nn.Linear(d_input, args.d_model)
        # An S4Block is an S4D layer followed by a GELU, dropout and a linear layer.
        self.blocks = nn.ModuleList(
            S4Block(S4D(args.d_model), dropout=args.dropout, tie_dropout=True)
            for _ in range(args.n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(args.d_model) for _ in range(args.n_layers))
        self.dropout = nn.Dropout(args.dropout)
        self.decoder = nn.Linear(args.d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  # (B, L, d_model)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            y, _ = block(x)  # the second value is the final state, which this model does not use
            x = norm(x + self.dropout(y))
        return self.decoder(x.mean(1))


@torch.no_grad()
def accuracy(x, labels):
    model.eval()
    batches = zip(x.split(1000), labels.split(1000), strict=True)
    correct = sum((model(xb).argmax(1) == yb).sum() for xb, yb in batches)
    return 100 * correct.item() / len(labels)


x, labels = load(train=True)
x_train, labels_train = x[:45000], labels[:45000]
val = x[45000:], labels[45000:]
test = load(train=False)
model = Model(d_input=3, d_output=10).to(device)
# The S4D layers train dt, A and B with a learning rate of 0.001 and without weight decay.
# `param_groups` puts these parameters in optimizer groups with those settings.
optimizer = torch.optim.AdamW(param_groups(model), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
for epoch in range(args.epochs):
    model.train()
    start = time.perf_counter()
    losses = []
    for i in torch.randperm(len(x_train), device=device).split(args.batch_size):
        loss = nn.functional.cross_entropy(model(x_train[i]), labels_train[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    scheduler.step()
    loss = torch.stack(losses).mean().item()  # waits for the device, so the time below is complete
    seconds = time.perf_counter() - start
    print(
        f"epoch {epoch + 1}: loss {loss:.3f}, val {accuracy(*val):.1f}%, "
        f"test {accuracy(*test):.1f}%, {seconds:.1f} s"
    )
