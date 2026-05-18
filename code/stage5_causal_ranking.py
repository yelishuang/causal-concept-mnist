"""Stage 5 — 因果概念排序（Causal Concept Ranking），论文公式 (2)(3)(4)。

混合前向传播：x 与 x' 的 Phi1 激活按像素级 V_k 二值掩码混合后送入 Phi2。

    z_x  = Phi1(x)              (B, 64, 7, 7)
    z_xp = Phi1(x')             (B, 64, 7, 7)
    mask = V_k.unsqueeze(0)     (1, 64, 7, 7)

    do(V_k = V_k(x'))    -> z_ie = z_x*(1-mask) + z_xp*mask    "via concept k"
    do(Vbar_k = Vbar(x'))-> z_de = z_x*mask     + z_xp*(1-mask) "bypass concept k"

按预测类提取概率：
    ATE = E[ (p_x - p_xp) / p_x ]
    IE_k= E[ (p_x - p_ie) / p_x ]
    DE_k= E[ (p_x - p_de) / p_x ]

按预测类过滤 p_x < min_conf 的样本。

输入
----
    checkpoints/cnn.pt
    checkpoints/counterfactuals_train.pt
    checkpoints/concept_units.pt        Vk_pixel (K, 64, 7, 7)

输出
----
    checkpoints/causal_ranking.pt   含 ATE / IE / DE / rank_by_IE

跑法
----
    python stage5_causal_ranking.py
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn", default="checkpoints/cnn.pt")
    p.add_argument("--cf", default="checkpoints/counterfactuals_train.pt")
    p.add_argument("--units", default="checkpoints/concept_units.pt")
    p.add_argument("--out", default="checkpoints/causal_ranking.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="filter p_pred(x) < this; required by relative-difference IE/DE")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Stage 5 :: causal concept ranking  (eq. 2/3/4)")
    kv("device", device)
    kv("min_conf", args.min_conf)
    kv("batch_size", args.batch_size)

    # 1. 加载模型
    net = Net().to(device)
    net.load_state_dict(torch.load(args.cnn, map_location=device))
    net.eval()

    # 2. 反事实配对
    cf = torch.load(args.cf, weights_only=False)
    x_all = cf["x"]
    xp_all = cf["x_prime"]
    kv("counterfactual pairs", f"{x_all.size(0)} (from {args.cf})")

    # 3. V_k （像素级）
    units = torch.load(args.units, weights_only=False)
    Vk = units["Vk_pixel"].to(device).float()
    names = units["names"]
    K, C, H, W = Vk.shape
    kv("K (concepts)", K)
    kv("feature shape", f"({C}, {H}, {W})")

    # 4. 累加器
    section("scanning counterfactual pairs")
    loader = DataLoader(
        TensorDataset(x_all, xp_all),
        batch_size=args.batch_size, shuffle=False,
    )
    sum_ate = torch.zeros((), dtype=torch.float64)
    sum_ie = torch.zeros(K, dtype=torch.float64)
    sum_de = torch.zeros(K, dtype=torch.float64)
    n_kept = 0

    for bi, (xb, xpb) in enumerate(loader):
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

        keep = p_x >= args.min_conf
        if keep.sum() == 0:
            continue
        z_x_k = z_x[keep]
        z_xp_k = z_xp[keep]
        p_x_k = p_x[keep]
        p_xp_k = p_xp[keep]
        pred_k = pred_x[keep]
        idx_k = torch.arange(keep.sum().item(), device=device)

        sum_ate += ((p_x_k - p_xp_k) / p_x_k).double().sum().cpu()
        n_kept += int(keep.sum().item())

        for k in range(K):
            mask = Vk[k].unsqueeze(0)
            # 中介效应计算
            z_ie = z_x_k * (1 - mask) + z_xp_k * mask
            z_de = z_x_k * mask + z_xp_k * (1 - mask)
            p_ie = F.softmax(net.phi2(z_ie), dim=1)[idx_k, pred_k]
            p_de = F.softmax(net.phi2(z_de), dim=1)[idx_k, pred_k]
            sum_ie[k] += ((p_x_k - p_ie) / p_x_k).double().sum().cpu()
            sum_de[k] += ((p_x_k - p_de) / p_x_k).double().sum().cpu()

        if (bi + 1) % 20 == 0:
            print(f"  batch {bi+1:>4d}/{len(loader):<4d}  kept = {n_kept}")

    if n_kept == 0:
        raise RuntimeError("no samples passed min_conf filter; check CNN training")

    ate = float(sum_ate / n_kept)
    ie = (sum_ie / n_kept).float()
    de = (sum_de / n_kept).float()
    rank = torch.argsort(ie, descending=True)

    section("results")
    kv("N_used", f"{n_kept} / {x_all.size(0)}  (filter p_x < {args.min_conf})")
    kv("ATE", f"{ate:.4f}")

    cols = [
        ("rank", 4, ">"),
        ("concept", 22, "<"),
        ("IE", 8, ">"),
        ("DE", 8, ">"),
        ("IE/ATE", 8, ">"),
    ]
    table_header(cols)
    for r, k in enumerate(rank.tolist()):
        ratio_to_ate = (ie[k] / ate) if ate != 0 else 0.0
        table_row([
            r + 1, names[k], float(ie[k]), float(de[k]), float(ratio_to_ate),
        ], cols)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": names,
        "ATE": float(ate),
        "IE": ie,
        "DE": de,
        "rank_by_IE": rank,
        "n_used": n_kept,
        "n_total": x_all.size(0),
        "min_conf": args.min_conf,
    }, out)
    saved(out)


if __name__ == "__main__":
    main()
