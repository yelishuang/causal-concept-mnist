"""Stage 1 — 训练两段式分类器 f = Phi2 ∘ Phi1。

Phi1 输出 (B, 64, 7, 7) 中间表示，作为后续概念关联与因果中介分析的对象。
Phi2 输出 10 类 logits。

输出
----
checkpoints/cnn.pt

跑法
----
    python stage1_cnn.py
    python stage1_cnn.py --epochs 3 --batch-size 256
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Net
from utils import header, kv, saved, table_header, table_row


def get_loaders(data_dir: str, batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out", default="checkpoints/cnn.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Stage 1 :: CNN training")
    kv("device", device)
    kv("epochs", args.epochs)
    kv("batch_size", args.batch_size)
    kv("lr", args.lr)
    kv("seed", args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    cols = [
        ("epoch", 5, ">"),
        ("train_loss", 10, ">"),
        ("test_loss", 10, ">"),
        ("test_acc", 9, ">"),
    ]
    table_header(cols)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)
        test_loss, test_acc = evaluate(model, test_loader, device)
        table_row([ep, train_loss, test_loss, test_acc], cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    saved(out)


if __name__ == "__main__":
    main()
