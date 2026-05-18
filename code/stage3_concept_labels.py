"""Stage 3 — 为 MNIST 每张图打 12 维概念二值标签 c_n ∈ {0,1}^12。

无需训练：MNIST 数字 → 概念是硬编码映射。论文里这一步对应"从放射学
报告 NLP 抽出的影像学征象"，在 MNIST 简化为查表。

输出
----
    checkpoints/concepts_train.pt   dict(concepts (60000,12), y, names, table)
    checkpoints/concepts_test.pt    dict(concepts (10000,12), y, names, table)

跑法
----
    python stage3_concept_labels.py
"""

import argparse
from pathlib import Path

import torch

from utils import (
    get_mnist,
    header, section, kv, saved, table_header, table_row,
)


# 12 个概念，顺序固定，下游全部按此索引引用
CONCEPT_NAMES = [
    "closed_loop",
    "two_loops",
    "loop_top",
    "loop_bottom",
    "vertical_stroke",
    "diagonal_stroke",
    "horizontal_top",
    "horizontal_middle",
    "horizontal_bottom",
    "left_facing_curve",
    "right_facing_curve",
    "oval_body",
]

# 每个数字的 12 维概念向量（与 CONCEPT_NAMES 同序）
DIGIT_TO_CONCEPTS = {
    0: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    1: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    6: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    7: [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    8: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    9: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def build_lookup_table() -> torch.Tensor:
    """返回 (10, K) 的 LongTensor，行 i = 数字 i 的概念向量。"""
    K = len(CONCEPT_NAMES)
    table = torch.zeros(10, K, dtype=torch.long)
    for d, vec in DIGIT_TO_CONCEPTS.items():
        assert len(vec) == K, f"digit {d} 概念向量长度 {len(vec)} != {K}"
        table[d] = torch.tensor(vec, dtype=torch.long)
    return table


def label_dataset(dataset, table: torch.Tensor):
    """从 dataset 取所有标签，查表得到 (N, K) 概念矩阵和 (N,) 数字标签。"""
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))], dtype=torch.long)
    concepts = table[y]
    return concepts, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="checkpoints")
    args = p.parse_args()

    table = build_lookup_table()
    K = len(CONCEPT_NAMES)

    header("Stage 3 :: concept labels")
    kv("K (concepts)", K)
    kv("|digits|", 10)

    section("digit → concept lookup table")
    cols = [("digit", 5, ">")]
    for n in CONCEPT_NAMES:
        cols.append((n[:6], 6, ">"))
    table_header(cols)
    for d in range(10):
        row = [d] + [int(table[d, k]) for k in range(K)]
        table_row(row, cols)

    rows = {tuple(table[d].tolist()) for d in range(10)}
    assert len(rows) == 10, "存在两个数字概念向量相同！"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        ds = get_mnist(args.data_dir, train=(split == "train"))
        concepts, y = label_dataset(ds, table)
        out = out_dir / f"concepts_{split}.pt"
        torch.save({
            "concepts": concepts,
            "y": y,
            "names": CONCEPT_NAMES,
            "table": table,
        }, out)
        n = concepts.size(0)
        pos_rate = concepts.float().mean(dim=0)

        section(f"split = {split}")
        kv("N", n)
        cols = [
            ("concept", 22, "<"),
            ("pos_rate", 8, ">"),
        ]
        table_header(cols)
        for k in range(K):
            table_row([CONCEPT_NAMES[k], float(pos_rate[k])], cols)
        saved(out)


if __name__ == "__main__":
    main()
