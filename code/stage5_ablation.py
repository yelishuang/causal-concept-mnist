"""Stage 5b — V_k 随机化消融（论文图 5 的 sanity check）。

把每个概念的 V_k 替换为在像素空间随机抽 |V_k| 个位置生成的掩码，重算 IE。
预期：lasso V_k 的 IE 显著高于随机 V_k 的 IE。

输入
----
    checkpoints/cnn.pt
    checkpoints/counterfactuals_train.pt
    checkpoints/concept_units.pt   提供 V_k 大小用作随机抽样
    checkpoints/causal_ranking.pt  提供 lasso 真实 IE 用作对比

输出
----
    checkpoints/ablation_random_Vk.pt

跑法
----
    python stage5_ablation.py
    python stage5_ablation.py --n-trials 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import Net
from utils import (
    to_normalized,
    header, section, kv, saved, table_header, table_row,
)


@torch.no_grad()
def compute_ie_de_for_masks(net, loader, masks, device, min_conf=0.5):
    """masks: (M, C, H, W) long, 0/1。返回 ATE, IE (M,), DE (M,), n_kept。"""
    M, C, H, W = masks.shape
    masks_f = masks.to(device).float()

    sum_ate = torch.zeros((), dtype=torch.float64)
    sum_ie = torch.zeros(M, dtype=torch.float64)
    sum_de = torch.zeros(M, dtype=torch.float64)
    n_kept = 0

    for xb, xpb in loader:
        xb = xb.to(device)
        xpb = xpb.to(device)

        z_x = net.phi1(to_normalized(xb))
        z_xp = net.phi1(to_normalized(xpb))
        probs_x = F.softmax(net.phi2(z_x), dim=1)
        probs_xp = F.softmax(net.phi2(z_xp), dim=1)
        pred_x = probs_x.argmax(dim=1)
        idx = torch.arange(xb.size(0), device=device)
        p_x = probs_x[idx, pred_x]
        p_xp = probs_xp[idx, pred_x]

        keep = p_x >= min_conf
        if keep.sum() == 0:
            continue
        z_x_k = z_x[keep]
        z_xp_k = z_xp[keep]
        p_x_k = p_x[keep]
        p_xp_k = p_xp[keep]
        pred_k = pred_x[keep]
        b = z_x_k.size(0)
        idx_k = torch.arange(b, device=device)

        sum_ate += ((p_x_k - p_xp_k) / p_x_k).double().sum().cpu()
        n_kept += b

        for m in range(M):
            mask = masks_f[m].unsqueeze(0)
            z_ie = z_x_k * (1 - mask) + z_xp_k * mask
            z_de = z_x_k * mask + z_xp_k * (1 - mask)
            p_ie = F.softmax(net.phi2(z_ie), dim=1)[idx_k, pred_k]
            p_de = F.softmax(net.phi2(z_de), dim=1)[idx_k, pred_k]
            sum_ie[m] += ((p_x_k - p_ie) / p_x_k).double().sum().cpu()
            sum_de[m] += ((p_x_k - p_de) / p_x_k).double().sum().cpu()

    if n_kept == 0:
        raise RuntimeError("no samples passed min_conf filter")
    ate = float(sum_ate / n_kept)
    ie = (sum_ie / n_kept).float()
    de = (sum_de / n_kept).float()
    return ate, ie, de, n_kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn", default="checkpoints/cnn.pt")
    p.add_argument("--cf", default="checkpoints/counterfactuals_train.pt")
    p.add_argument("--units", default="checkpoints/concept_units.pt")
    p.add_argument("--ranking", default="checkpoints/causal_ranking.pt")
    p.add_argument("--out", default="checkpoints/ablation_random_Vk.pt")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    header("Stage 5b :: random-V_k ablation")
    kv("device", device)
    kv("n_trials", args.n_trials)
    kv("min_conf", args.min_conf)
    kv("seed", args.seed)

    # 1. 模型
    net = Net().to(device)
    net.load_state_dict(torch.load(args.cnn, map_location=device))
    net.eval()

    # 2. 反事实
    cf = torch.load(args.cf, weights_only=False)
    x_all = cf["x"]
    xp_all = cf["x_prime"]
    loader = DataLoader(TensorDataset(x_all, xp_all),
                        batch_size=args.batch_size, shuffle=False)
    kv("counterfactual pairs", x_all.size(0))

    # 3. 真 V_k 与现有排序
    units = torch.load(args.units, weights_only=False)
    Vk_real = units["Vk_pixel"]
    names = units["names"]
    K, C, H, W = Vk_real.shape
    D = C * H * W
    sizes = Vk_real.view(K, D).sum(dim=1).tolist()
    kv("K (concepts)", K)
    kv("feature shape", f"({C}, {H}, {W}) -> D = {D}")

    rank = torch.load(args.ranking, weights_only=False)
    ie_real = rank["IE"]

    # 4. 生成 K * n_trials 个随机像素掩码
    section(f"random masks: {args.n_trials} trials per concept")
    n_trials = args.n_trials
    M = K * n_trials
    rand_flat = torch.zeros(M, D, dtype=torch.long)
    for k in range(K):
        size_k = sizes[k]
        for t in range(n_trials):
            perm = torch.randperm(D)[:size_k]
            rand_flat[k * n_trials + t, perm] = 1
    rand_masks = rand_flat.view(M, C, H, W)
    kv("total masks", f"{n_trials} × {K} = {M}")

    # 5. 扫数据
    section("scanning counterfactual pairs")
    ate, ie_all, de_all, n_kept = compute_ie_de_for_masks(
        net, loader, rand_masks, device, min_conf=args.min_conf,
    )
    ie_per = ie_all.view(K, n_trials)
    de_per = de_all.view(K, n_trials)
    ie_mean = ie_per.mean(dim=1)
    ie_std = ie_per.std(dim=1)

    section("results")
    kv("N_used", n_kept)
    kv("ATE", f"{ate:.4f}")

    cols = [
        ("concept", 22, "<"),
        ("|V_k|", 6, ">"),
        ("IE_lasso", 9, ">"),
        ("IE_rand_mean", 13, ">"),
        ("IE_rand_std", 12, ">"),
        ("lift", 8, ">"),
    ]
    table_header(cols)
    diffs = []
    for k in range(K):
        lift = float(ie_real[k]) - float(ie_mean[k])
        diffs.append(lift)
        table_row([
            names[k], int(sizes[k]),
            float(ie_real[k]), float(ie_mean[k]), float(ie_std[k]), lift,
        ], cols)

    section("summary")
    kv("avg IE_lasso", f"{ie_real.mean().item():.4f}")
    kv("avg IE_rand_mean", f"{ie_mean.mean().item():.4f}")
    kv("avg lift (lasso - rand)", f"{sum(diffs) / len(diffs):.4f}")
    kv("concepts with lasso > rand", f"{sum(1 for d in diffs if d > 0)} / {K}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": names,
        "Vk_sizes": torch.tensor(sizes),
        "IE_lasso": ie_real,
        "IE_random_per_trial": ie_per,
        "DE_random_per_trial": de_per,
        "IE_random_mean": ie_mean,
        "IE_random_std": ie_std,
        "ATE_random": ate,
        "n_trials": n_trials,
        "n_used": n_kept,
    }, out)
    saved(out)


if __name__ == "__main__":
    main()
