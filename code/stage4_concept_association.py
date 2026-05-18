"""Stage 4 — 概念关联，对应论文公式 (1)。

对每个概念 c_k 在 Phi1 输出（3136 = 64*7*7 维像素级特征）上独立训练
L1-logistic 回归。系数非零的位置即概念单元 V_k。

输出
----
checkpoints/concept_units.pt
    Vk_pixel       : (K, 64, 7, 7) long, 0/1
    coef_pixel     : (K, 64, 7, 7) float
    intercept      : (K,) float
    feature_mean   : (3136,) 用于标准化
    feature_std    : (3136,)
    train_auc      : (K,)
    test_auc       : (K,)
    n_nonzero_*    : 每个概念的稀疏统计
    feature_shape  : (64, 7, 7)
    lam, hard_threshold

跑法
----
    python stage4_concept_association.py
    python stage4_concept_association.py --lam 0.1 --hard-threshold 0.02   # 更稀疏
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model import Net
from utils import (
    get_mnist, to_normalized,
    header, section, kv, saved, table_header, table_row,
)


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_phi1(net: Net, dataset, batch_size: int, device) -> torch.Tensor:
    """前向到 Phi1，返回 (N, 64, 7, 7) 激活（GPU）。"""
    net.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    for x, _ in loader:
        x = x.to(device)
        feats.append(net.phi1(to_normalized(x)))
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# L1 逻辑回归（近端梯度法 + 训练后硬阈值清扫）
# ---------------------------------------------------------------------------

def fit_l1_logreg(
    X: torch.Tensor, y: torch.Tensor,
    lam: float, lr: float, n_epochs: int, batch_size: int,
    pos_weight: float, device,
    hard_threshold: float = 1e-2,
):
    """带近端梯度的 L1 二分类逻辑回归，最后再做一次硬阈值清扫。

    损失:    BCE(sigmoid(X @ w + b), y) + lam * ||w||_1
    近端项:   每步 Adam 后做软阈值（soft-thresholding）
    hard_threshold: 训练结束后再清扫 |w| < hard_threshold 的小残差，
                    对抗 mini-batch 噪声反复把已被推零的小值推回非零的现象。
    """
    N, D = X.shape
    w = torch.zeros(D, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)

    pw = torch.tensor([pos_weight], device=device)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = X[idx] @ w + b
            loss = F.binary_cross_entropy_with_logits(
                logits, y[idx], pos_weight=pw,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                w.data = torch.sign(w.data) * (w.data.abs() - lr * lam).clamp_min(0)

    with torch.no_grad():
        if hard_threshold > 0:
            w.data[w.data.abs() < hard_threshold] = 0
    return w.detach(), b.detach()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--cnn", default="checkpoints/cnn.pt")
    p.add_argument("--concepts-train", default="checkpoints/concepts_train.pt")
    p.add_argument("--concepts-test", default="checkpoints/concepts_test.pt")
    p.add_argument("--out", default="checkpoints/concept_units.pt")
    p.add_argument("--lam", type=float, default=0.05,
                   help="L1 正则强度（越大越稀疏）")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--feat-batch-size", type=int, default=256)
    p.add_argument("--hard-threshold", type=float, default=0.01)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Stage 4 :: concept association  (eq. 1)")
    kv("device", device)
    kv("lam", args.lam)
    kv("lr", args.lr)
    kv("epochs", args.epochs)
    kv("batch_size", args.batch_size)
    kv("hard_threshold", args.hard_threshold)

    # 1. 加载 CNN
    net = Net().to(device)
    net.load_state_dict(torch.load(args.cnn, map_location=device))

    # 2. 提取 Phi1 特征
    section("extract Phi1 pixel-level features")
    train_set = get_mnist(args.data_dir, train=True)
    test_set = get_mnist(args.data_dir, train=False)
    feats_tr_chw = extract_phi1(net, train_set, args.feat_batch_size, device)
    feats_te_chw = extract_phi1(net, test_set, args.feat_batch_size, device)
    N_tr, C, H, W = feats_tr_chw.shape
    D = C * H * W
    kv("train shape", f"{tuple(feats_tr_chw.shape)} -> flat {D}")
    kv("test  shape", f"{tuple(feats_te_chw.shape)}")

    X_tr = feats_tr_chw.reshape(N_tr, D)
    X_te = feats_te_chw.reshape(feats_te_chw.size(0), D)
    mu = X_tr.mean(dim=0)
    sd = X_tr.std(dim=0).clamp_min(1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    kv("feature mem (X_tr)", f"{X_tr.element_size() * X_tr.numel() / 1e6:.1f} MB")
    if device.type == "cuda":
        kv("GPU allocated", f"{torch.cuda.memory_allocated() / 1e6:.1f} MB")
    del feats_tr_chw, feats_te_chw

    # 3. 加载概念标签
    ct = torch.load(args.concepts_train, weights_only=False)
    cv = torch.load(args.concepts_test, weights_only=False)
    Y_tr_all = ct["concepts"].float().to(device)
    Y_te_all = cv["concepts"].float().to(device)
    names = ct["names"]
    K = len(names)

    # 4. 对每个概念独立训练 L1 逻辑回归
    section(f"fit L1 logreg per concept  (K = {K})")
    Vk_chw = torch.zeros(K, C, H, W, dtype=torch.long)
    coef_chw = torch.zeros(K, C, H, W, dtype=torch.float32)
    intercept = torch.zeros(K)
    train_auc = torch.zeros(K)
    test_auc = torch.zeros(K)
    n_nz_pixel = torch.zeros(K, dtype=torch.long)
    n_nz_chan = torch.zeros(K, dtype=torch.long)

    cols = [
        ("concept", 22, "<"),
        ("train_auc", 9, ">"),
        ("test_auc", 9, ">"),
        ("#pixels", 8, ">"),
        ("#chan", 6, ">"),
    ]
    table_header(cols)
    for k in range(K):
        y_tr = Y_tr_all[:, k]
        n_pos = float(y_tr.sum().item())
        n_neg = float(N_tr - n_pos)
        pos_weight = max(n_neg / max(n_pos, 1.0), 1.0)

        w_k, b_k = fit_l1_logreg(
            X_tr, y_tr, lam=args.lam, lr=args.lr,
            n_epochs=args.epochs, batch_size=args.batch_size,
            pos_weight=pos_weight, device=device,
            hard_threshold=args.hard_threshold,
        )

        with torch.no_grad():
            prob_tr = torch.sigmoid(X_tr @ w_k + b_k).cpu().numpy()
            prob_te = torch.sigmoid(X_te @ w_k + b_k).cpu().numpy()
        train_auc[k] = roc_auc_score(Y_tr_all[:, k].cpu().numpy(), prob_tr)
        test_auc[k] = roc_auc_score(Y_te_all[:, k].cpu().numpy(), prob_te)

        coef_chw[k] = w_k.view(C, H, W).cpu()
        intercept[k] = b_k.cpu().item()
        Vk_chw[k] = (coef_chw[k] != 0).long()
        n_nz_pixel[k] = int(Vk_chw[k].sum().item())
        n_nz_chan[k] = int((Vk_chw[k].any(dim=(1, 2))).sum().item())

        table_row([
            names[k], float(train_auc[k]), float(test_auc[k]),
            int(n_nz_pixel[k]), int(n_nz_chan[k]),
        ], cols)

    # 5. 保存
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": names,
        "Vk_pixel": Vk_chw,
        "coef_pixel": coef_chw,
        "intercept": intercept,
        "feature_mean": mu.cpu(),
        "feature_std": sd.cpu(),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "n_nonzero_pixel": n_nz_pixel,
        "n_nonzero_channel": n_nz_chan,
        "lam": args.lam,
        "hard_threshold": args.hard_threshold,
        "feature_shape": (C, H, W),
    }, out)
    saved(out)

    # 6. 像素级稀疏统计
    section("sparsity summary")
    pct_pixel = n_nz_pixel.float() / D * 100
    kv("avg test_auc",
       f"{test_auc.mean():.4f}   min = {test_auc.min():.4f} ({names[int(test_auc.argmin())]})")
    kv("avg #pixels",
       f"{n_nz_pixel.float().mean():.1f} / {D}  ({pct_pixel.mean():.1f}%)")
    kv("min #pixels",
       f"{int(n_nz_pixel.min())}  ({names[int(n_nz_pixel.argmin())]})")
    kv("max #pixels",
       f"{int(n_nz_pixel.max())}  ({names[int(n_nz_pixel.argmax())]})")
    kv("avg #channels involved", f"{n_nz_chan.float().mean():.1f} / {C}")

    # 像素复用分布
    section("pixel reuse distribution")
    Vk_flat = Vk_chw.view(K, D)
    pix_overlap = Vk_flat.sum(dim=0)
    n_zero = int((pix_overlap == 0).sum())
    n_unique = int((pix_overlap == 1).sum())
    n_shared = int((pix_overlap >= 2).sum())
    pix_used = D - n_zero
    kv("never selected", f"{n_zero:>5d}   ({n_zero/D*100:.1f}%)")
    kv("exclusive (1 concept)", f"{n_unique:>5d}   ({n_unique/D*100:.1f}%)")
    kv("shared (>=2 concepts)", f"{n_shared:>5d}   ({n_shared/D*100:.1f}%)")
    if pix_used > 0:
        avg_share = float(pix_overlap[pix_overlap > 0].float().mean())
        kv("avg sharing among used", f"{avg_share:.2f} concepts/pixel")

    # 概念两两 Jaccard 相似度
    section("pairwise Jaccard between V_k")
    inter = Vk_flat @ Vk_flat.T
    union = n_nz_pixel.unsqueeze(0) + n_nz_pixel.unsqueeze(1) - inter
    jaccard = inter.float() / union.clamp_min(1).float()
    iu = torch.triu_indices(K, K, offset=1)
    jvals = jaccard[iu[0], iu[1]]
    kv("mean", f"{jvals.mean():.3f}")
    kv("median", f"{jvals.median():.3f}")
    kv("max", f"{jvals.max():.3f}")


if __name__ == "__main__":
    main()
