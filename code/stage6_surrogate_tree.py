"""Stage 6 — 替代解释函数（Surrogate Explanation Function），论文公式 (5)(6)。

公式 (5): w_n[k] = logit(P(c_n[k]=1 | x_n))
    用 stage 4 的像素级 lasso 系数 + 截距，把 3136 维像素特征映射到 K 维概念 logit。

公式 (6): g(w_n) ≈ argmax f(x_n)
    用 (w_n, CNN 预测类) 训浅决策树，目标是用概念 logit 还原 CNN 决策。

三段评估
--------
    1. 全部 K 个概念 -> 决策树召回率（性能上限）
    2. 按 IE 排名累积加入 top-x 个概念 -> 召回率随 x 的变化（论文 Fig.5 趋势）
    3. 决策树本身的结构（feature_importance、深度、每类还原率）

输入
----
    checkpoints/cnn.pt
    checkpoints/concept_units.pt        coef_pixel + intercept + feature_mean/std
    checkpoints/causal_ranking.pt       rank_by_IE

输出
----
    checkpoints/surrogate_tree.pt

跑法
----
    python stage6_surrogate_tree.py
    python stage6_surrogate_tree.py --max-depth 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from model import Net
from utils import (
    get_mnist, to_normalized,
    header, section, kv, saved, table_header, table_row,
)


@torch.no_grad()
def extract_pixel_features(net, dataset, batch_size, device):
    """前向到 Phi1，返回 (N, 3136) 像素级特征 + (N,) CNN 预测类。"""
    net.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats = []
    preds = []
    for x, _ in loader:
        x = x.to(device)
        z = net.phi1(to_normalized(x))
        feats.append(z.flatten(1).cpu())
        preds.append(net.phi2(z).argmax(dim=1).cpu())
    return torch.cat(feats, dim=0), torch.cat(preds, dim=0)


def compute_concept_logits(feats, mu, sd, coef, intercept):
    """feats (N, D) -> 概念 logit 向量 (N, K)。"""
    feats_norm = (feats - mu) / sd
    return feats_norm @ coef.T + intercept


def fit_and_eval_tree(W_train, pred_train, W_test, pred_test, max_depth, seed=0):
    # 决策树
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(W_train, pred_train)
    acc_tr = accuracy_score(pred_train, tree.predict(W_train))
    acc_te = accuracy_score(pred_test, tree.predict(W_test))
    return acc_tr, acc_te, tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--cnn", default="checkpoints/cnn.pt")
    p.add_argument("--units", default="checkpoints/concept_units.pt")
    p.add_argument("--ranking", default="checkpoints/causal_ranking.pt")
    p.add_argument("--out", default="checkpoints/surrogate_tree.pt")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header("Stage 6 :: surrogate decision tree  (eq. 5/6)")
    kv("device", device)
    kv("max_depth", args.max_depth)
    kv("seed", args.seed)

    # 1. 模型
    net = Net().to(device)
    net.load_state_dict(torch.load(args.cnn, map_location=device))
    net.eval()

    # 2. lasso 系数（像素级）
    units = torch.load(args.units, weights_only=False)
    names = units["names"]
    coef_chw = units["coef_pixel"]
    intercept = units["intercept"]
    mu = units["feature_mean"]
    sd = units["feature_std"]
    K = coef_chw.size(0)
    C, H, W = coef_chw.shape[1:]
    D = C * H * W
    coef = coef_chw.reshape(K, D)
    kv("K (concepts)", K)
    kv("feature shape", f"({C}, {H}, {W}) -> D = {D}")

    # 3. IE 排序
    rank = torch.load(args.ranking, weights_only=False)
    rank_by_IE = rank["rank_by_IE"]
    rank_lookup = {int(idx): r + 1 for r, idx in enumerate(rank_by_IE.tolist())}
    ie_np = rank["IE"].numpy()
    kv("top-5 by IE", ", ".join(names[i] for i in rank_by_IE[:5].tolist()))

    # 4. 提取特征 + CNN 预测
    section("extract features")
    train_set = get_mnist(args.data_dir, train=True)
    test_set = get_mnist(args.data_dir, train=False)
    feats_tr, pred_tr = extract_pixel_features(net, train_set, args.batch_size, device)
    feats_te, pred_te = extract_pixel_features(net, test_set, args.batch_size, device)
    kv("train feats", tuple(feats_tr.shape))
    kv("test  feats", tuple(feats_te.shape))

    # 5. 概念 logit 向量 (公式 5)
    W_tr = compute_concept_logits(feats_tr, mu, sd, coef, intercept).numpy()
    W_te = compute_concept_logits(feats_te, mu, sd, coef, intercept).numpy()
    pred_tr_np = pred_tr.numpy()
    pred_te_np = pred_te.numpy()
    kv("W_train", W_tr.shape)
    kv("W_test ", W_te.shape)

    # 6. 评估 1：上限召回
    section(f"eval 1 :: full {K}-concept tree (upper bound)")
    acc_tr_full, acc_te_full, tree_full = fit_and_eval_tree(
        W_tr, pred_tr_np, W_te, pred_te_np, args.max_depth, args.seed,
    )
    kv("acc_train", f"{acc_tr_full:.4f}")
    kv("acc_test", f"{acc_te_full:.4f}")

    # 7. 评估 2：按 IE 累积加入
    section("eval 2 :: cumulative top-x by IE rank")
    rank_idx = rank_by_IE.tolist()
    recall_by_ie = []
    cols = [
        ("rank", 4, ">"),
        ("+concept", 22, "<"),
        ("acc_test", 9, ">"),
        ("delta", 8, ">"),
        ("recover", 8, ">"),
    ]
    table_header(cols)
    prev = 0.0
    for x in range(1, K + 1):
        chosen = rank_idx[:x]
        acc_tr, acc_te, _ = fit_and_eval_tree(
            W_tr[:, chosen], pred_tr_np,
            W_te[:, chosen], pred_te_np,
            args.max_depth, args.seed,
        )
        recall_by_ie.append((x, acc_tr, acc_te, list(chosen)))
        delta = acc_te - prev
        recover = acc_te / acc_te_full if acc_te_full > 0 else 0.0
        table_row([x, names[chosen[-1]], acc_te, delta, recover], cols)
        prev = acc_te

    # 8. 评估 3：决策树结构
    section(f"eval 3 :: tree structure (full {K}-concept model)")
    n_nodes = tree_full.tree_.node_count
    n_leaves = tree_full.get_n_leaves()
    depth = tree_full.get_depth()
    kv("nodes", n_nodes)
    kv("leaves", n_leaves)
    kv("depth", f"{depth}  (max_depth limit = {args.max_depth})")

    fi = tree_full.feature_importances_
    fi_order = np.argsort(-fi)
    section("feature importance vs IE rank")
    cols = [
        ("rank", 4, ">"),
        ("concept", 22, "<"),
        ("importance", 11, ">"),
        ("IE_rank", 8, ">"),
        ("IE", 8, ">"),
    ]
    table_header(cols)
    for r, k in enumerate(fi_order.tolist()):
        table_row([
            r + 1, names[k], float(fi[k]), rank_lookup[k], float(ie_np[k]),
        ], cols)

    section("per-class recovery (full model)")
    test_pred_tree = tree_full.predict(W_te)
    cols = [
        ("class", 5, ">"),
        ("#samples", 9, ">"),
        ("tree_acc", 9, ">"),
    ]
    table_header(cols)
    for c in range(10):
        mask = pred_te_np == c
        if mask.sum() == 0:
            continue
        acc_c = (test_pred_tree[mask] == c).mean()
        table_row([c, int(mask.sum()), float(acc_c)], cols)

    # 9. summary
    section("summary")
    target = 0.95 * acc_te_full
    ie_arr = np.array([r[2] for r in recall_by_ie])
    first_x_ie = next((x + 1 for x, a in enumerate(ie_arr) if a >= target), None)
    kv(f"upper bound (all {K})", f"{acc_te_full:.4f}")
    kv("top-1 (IE)", f"{recall_by_ie[0][2]:.4f}")
    kv("top-3 (IE)", f"{recall_by_ie[2][2]:.4f}")
    kv("top-6 (IE)", f"{recall_by_ie[5][2]:.4f}")
    if first_x_ie is not None:
        kv("first reach 95% upper bound", f"top-{first_x_ie}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "names": names,
        "max_depth": args.max_depth,
        "acc_train_full": acc_tr_full,
        "acc_test_full": acc_te_full,
        "recall_by_IE": [(x, atr, ate, c) for x, atr, ate, c in recall_by_ie],
        "rank_by_IE": rank_by_IE,
        "tree_feature_importance": fi,
        "tree_n_nodes": n_nodes,
        "tree_n_leaves": n_leaves,
        "tree_depth": depth,
    }, out)
    saved(out)


if __name__ == "__main__":
    main()
