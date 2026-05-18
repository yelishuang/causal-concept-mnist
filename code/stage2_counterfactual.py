"""Stage 2 — VAE 训练 + 反事实生成。

子命令
------
    train          训练 MNIST conv-VAE，保存权重 -> vae.pt
    counterfactual 用训好的 VAE + CNN 生成反事实 (x, x') 配对

反事实定义
----------
    在 VAE 潜空间从 mu(x) 出发，用梯度下降最小化 f(decode(z))[pred(x)]，
    直到 pred(x) 类的概率降到 target_p（默认 0.1）以下。decode(z) 即 x'。

输出
----
    checkpoints/vae.pt
    checkpoints/counterfactuals_train.pt   含 x, x', y, pred_x, pred_xprime, p
    checkpoints/counterfactuals_test.pt
    checkpoints/cf_samples_*/grid.png      可视化

跑法
----
    python stage2_counterfactual.py train
    python stage2_counterfactual.py counterfactual --split train \\
        --out checkpoints/counterfactuals_train.pt
    python stage2_counterfactual.py counterfactual --split test  \\
        --out checkpoints/counterfactuals_test.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import VAE, vae_loss
from utils import (
    classify, to_normalized,
    pick_random, pick_one_per_class,
    get_mnist, load_models,
    header, section, kv, saved, table_header, table_row,
)


# ---------------------------------------------------------------------------
# 子命令 1：训练 VAE
# ---------------------------------------------------------------------------

def cmd_train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Stage 2 :: VAE training")
    kv("device", device)
    kv("epochs", args.epochs)
    kv("batch_size", args.batch_size)
    kv("latent_dim", args.latent_dim)
    kv("lr", args.lr)
    kv("seed", args.seed)

    train_set = get_mnist(args.data_dir, train=True)
    test_set = get_mnist(args.data_dir, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = VAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    samples, _ = pick_one_per_class(test_set)

    cols = [
        ("epoch", 5, ">"),
        ("loss", 10, ">"),
        ("bce", 10, ">"),
        ("kld", 10, ">"),
    ]
    table_header(cols)
    for ep in range(1, args.epochs + 1):
        model.train()
        n = 0
        loss_sum = bce_sum = kld_sum = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, bce, kld = vae_loss(x_hat, x, mu, logvar)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            bce_sum += bce.item()
            kld_sum += kld.item()
            n += x.size(0)
        table_row([ep, loss_sum / n, bce_sum / n, kld_sum / n], cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    saved(out)

    # reconstruction grid
    model.eval()
    with torch.no_grad():
        x = samples.to(device)
        x_hat, _, _ = model(x)
    grid = torch.cat([x.cpu(), x_hat.cpu()], dim=0)
    recon_path = Path(args.samples_dir) / "recon.png"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, recon_path, nrow=10)
    saved(recon_path)


# ---------------------------------------------------------------------------
# 子命令 2：反事实生成
# ---------------------------------------------------------------------------

def generate_counterfactual_batch(
    net, vae, x,
    target_p: float = 0.1,
    max_iter: int = 300,
    lr: float = 0.3,
    lambda_reg: float = 0.0,
):
    """潜空间梯度下降，最小化 f(decode(z))[pred(x)]，直到 < target_p。"""
    net.eval()
    vae.eval()
    device = x.device

    with torch.no_grad():
        probs_x, pred_x = classify(net, x)
        mu, _ = vae.encode(x)
    z_init = mu.detach().clone()
    z = z_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    idx = torch.arange(x.size(0), device=device)
    for it in range(max_iter):
        x_hat = vae.decode(z)
        logits = net(to_normalized(x_hat))
        probs = F.softmax(logits, dim=1)
        p_orig = probs[idx, pred_x]
        # 正则化，要求z最小化改动
        reg = ((z - z_init) ** 2).sum(dim=1)
        loss = (p_orig + lambda_reg * reg).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (p_orig.detach() < target_p).all():
            break

    with torch.no_grad():
        x_prime = vae.decode(z).clamp(0, 1)
        probs_xp, pred_xp = classify(net, x_prime)

    info = {
        "pred_x": pred_x.cpu(),
        "pred_xprime": pred_xp.cpu(),
        "p_orig_x": probs_x[idx, pred_x].cpu(),
        "p_orig_xprime": probs_xp[idx, pred_x].cpu(),
        "n_iters": it + 1,
    }
    return x_prime.detach().cpu(), info


def plot_pairs(x, x_prime, info, y, out_path: Path, title=""):
    n = x.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.2))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        flipped = info["pred_x"][i] != info["pred_xprime"][i]
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"y={y[i].item()}  pred={info['pred_x'][i].item()}\n"
            f"p={info['p_orig_x'][i]:.2f}",
            fontsize=8,
        )
        axes[1, i].imshow(x_prime[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        mark = " *" if flipped else ""
        axes[1, i].set_title(
            f"pred'={info['pred_xprime'][i].item()}{mark}\n"
            f"p'={info['p_orig_xprime'][i]:.2f}",
            fontsize=8,
        )
    flip_rate = (info["pred_x"] != info["pred_xprime"]).float().mean().item()
    fig.suptitle(
        f"{title}   flip_rate={flip_rate:.0%}  "
        f"avg p drop={info['p_orig_x'].mean()-info['p_orig_xprime'].mean():.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved(out_path)


def cmd_counterfactual(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header(f"Stage 2 :: counterfactual generation [{args.split}]")
    kv("device", device)
    kv("split", args.split)
    kv("target_p", args.target_p)
    kv("max_iter", args.max_iter)
    kv("lr", args.lr)
    kv("lambda_reg", args.lambda_reg)
    kv("batch_size", args.batch_size)

    dataset = get_mnist(args.data_dir, train=(args.split == "train"))
    net, vae = load_models(args.cnn, args.vae, args.latent_dim, device)

    section("preview samples")
    samples_x, samples_y = pick_random(dataset, 10, args.seed)
    samples_x = samples_x.to(device)
    sx_prime, sx_info = generate_counterfactual_batch(
        net, vae, samples_x,
        target_p=args.target_p, max_iter=args.max_iter,
        lr=args.lr, lambda_reg=args.lambda_reg,
    )
    plot_pairs(samples_x.cpu(), sx_prime, sx_info, samples_y,
               Path(args.samples_dir) / "grid.png",
               title="random 10 samples")

    perclass_x, perclass_y = pick_one_per_class(dataset)
    perclass_x = perclass_x.to(device)
    pc_prime, pc_info = generate_counterfactual_batch(
        net, vae, perclass_x,
        target_p=args.target_p, max_iter=args.max_iter,
        lr=args.lr, lambda_reg=args.lambda_reg,
    )
    plot_pairs(perclass_x.cpu(), pc_prime, pc_info, perclass_y,
               Path(args.samples_dir) / "per_class.png",
               title="one per class")

    section(f"full {args.split} set")
    if args.max_samples and args.max_samples > 0:
        n_total = min(args.max_samples, len(dataset))
        subset = torch.utils.data.Subset(dataset, list(range(n_total)))
    else:
        n_total = len(dataset)
        subset = dataset
    kv("n_total", n_total)

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    all_x = torch.empty(n_total, 1, 28, 28)
    all_xp = torch.empty(n_total, 1, 28, 28)
    all_y = torch.empty(n_total, dtype=torch.long)
    all_pred_x = torch.empty(n_total, dtype=torch.long)
    all_pred_xp = torch.empty(n_total, dtype=torch.long)
    all_p_x = torch.empty(n_total)
    all_p_xp = torch.empty(n_total)

    pos = 0
    flips = 0
    cols = [
        ("processed", 12, ">"),
        ("flip_rate", 10, ">"),
    ]
    table_header(cols)
    for bi, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        xp, info = generate_counterfactual_batch(
            net, vae, xb,
            target_p=args.target_p, max_iter=args.max_iter,
            lr=args.lr, lambda_reg=args.lambda_reg,
        )
        b = xb.size(0)
        all_x[pos:pos + b] = xb.cpu()
        all_xp[pos:pos + b] = xp
        all_y[pos:pos + b] = yb
        all_pred_x[pos:pos + b] = info["pred_x"]
        all_pred_xp[pos:pos + b] = info["pred_xprime"]
        all_p_x[pos:pos + b] = info["p_orig_x"]
        all_p_xp[pos:pos + b] = info["p_orig_xprime"]
        flips += (info["pred_x"] != info["pred_xprime"]).sum().item()
        pos += b
        if (bi + 1) % 20 == 0 or pos == n_total:
            table_row([f"{pos}/{n_total}", flips / pos], cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "x": all_x, "x_prime": all_xp, "y": all_y,
        "pred_x": all_pred_x, "pred_xprime": all_pred_xp,
        "p_orig_x": all_p_x, "p_orig_xprime": all_p_xp,
        "split": args.split,
    }, out)
    saved(out)

    section("summary")
    kv("flip_rate", f"{flips / pos:.4f}")
    kv("avg p drop (≈ ATE)", f"{(all_p_x - all_p_xp).mean():.4f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train VAE")
    pt.add_argument("--data-dir", default="./data")
    pt.add_argument("--out", default="checkpoints/vae.pt")
    pt.add_argument("--samples-dir", default="checkpoints/vae_samples")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=128)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--latent-dim", type=int, default=16)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    pc = sub.add_parser("counterfactual", help="generate counterfactuals")
    pc.add_argument("--data-dir", default="./data")
    pc.add_argument("--cnn", default="checkpoints/cnn.pt")
    pc.add_argument("--vae", default="checkpoints/vae.pt")
    pc.add_argument("--out", default="checkpoints/counterfactuals.pt")
    pc.add_argument("--samples-dir", default="checkpoints/cf_samples")
    pc.add_argument("--split", choices=["train", "test"], default="train")
    pc.add_argument("--max-samples", type=int, default=0,
                    help="0 = full set; >0 for quick check")
    pc.add_argument("--batch-size", type=int, default=128)
    pc.add_argument("--target-p", type=float, default=0.1)
    pc.add_argument("--max-iter", type=int, default=300)
    pc.add_argument("--lr", type=float, default=0.3)
    pc.add_argument("--lambda-reg", type=float, default=0.0)
    pc.add_argument("--latent-dim", type=int, default=16)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=cmd_counterfactual)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
