# Causal-DL on MNIST — 复现交接文档

## 1. 目标

最小化复现 Singla et al. *Using Causal Analysis for Conceptual Deep Learning Explanation* (MICCAI 2021) 的核心理念：用因果中介分析（IE）对概念进行因果排序，区分"真正驱动决策的概念"与"仅相关的概念"。

**不复现**：cGAN 反事实生成、MIMIC-CXR、DenseNet-121。
**用替代**：MNIST + 简单 CNN + VAE 潜空间反事实。

---

## 2. 数据与概念

数据集：MNIST（10 类手写数字）。
概念标注：硬编码 `digit -> 12-dim binary vector` 查表（见 [stage3_concept_labels.py](../stage3_concept_labels.py) 中的 `DIGIT_TO_CONCEPTS`）。论文里这一步对应"NLP 抽影像学征象"，MNIST 上简化为查表。

12 个概念固定顺序：
`closed_loop, two_loops, loop_top, loop_bottom, vertical_stroke, diagonal_stroke, horizontal_top, horizontal_middle, horizontal_bottom, left_facing_curve, right_facing_curve, oval_body`

---

## 3. 模型

两段式 CNN（[model.py](../model.py)）：

- `Phi1` : x (B,1,28,28) -> z (B,64,7,7)。后续因果分析的对象。
- `Phi2` : z -> logits (B,10)。
- 决策由 `f = Phi2 ∘ Phi1` 给出。

VAE：标准 conv VAE，潜空间 16 维。仅用于生成反事实 x'。

---

## 4. 反事实 x'

在 VAE 潜空间从 `mu(x)` 出发梯度下降，最小化 `f(decode(z))[pred(x)]`，直到 `pred(x)` 类的概率掉到 `target_p=0.1` 以下。`decode(z)` 即 x'。
默认参数 `lr=0.3, max_iter=300, lambda_reg=0.0, target_p=0.1` 在 MNIST 上 flip rate ≈ 100%, ATE ≈ 0.999。

---

## 5. Pipeline 概览

| Stage | 文件 | 论文公式 | 输入 | 输出 |
|-------|------|----------|------|------|
| 1 | [stage1_cnn.py](../stage1_cnn.py) | — | MNIST | `cnn.pt` |
| 2-train | [stage2_counterfactual.py](../stage2_counterfactual.py) `train` | — | MNIST | `vae.pt` |
| 2-cf | `stage2_counterfactual.py counterfactual` | — | `cnn.pt`, `vae.pt` | `counterfactuals_{train,test}.pt` |
| 3 | [stage3_concept_labels.py](../stage3_concept_labels.py) | — | MNIST labels | `concepts_{train,test}.pt` |
| 4 | [stage4_concept_association.py](../stage4_concept_association.py) | (1) | `cnn.pt`, concepts | `concept_units.pt` |
| 5 | [stage5_causal_ranking.py](../stage5_causal_ranking.py) | (2)(3)(4) | `cnn.pt`, cf, units | `causal_ranking.pt` |
| 5b | [stage5_ablation.py](../stage5_ablation.py) | sanity check | 同上 + ranking | `ablation_random_Vk.pt` |
| 6 | [stage6_surrogate_tree.py](../stage6_surrogate_tree.py) | (5)(6) | `cnn.pt`, units, ranking | `surrogate_tree.pt` |

---

## 6. 全流程命令（首次复现）

所有脚本从 `code/` 目录下运行，默认 `--out` 写到 `checkpoints/`。

```bash
# 0. 环境检查
python check_env.py

# 1. 训练 CNN
python stage1_cnn.py

# 2. 训练 VAE + 生成反事实
python stage2_counterfactual.py train
python stage2_counterfactual.py counterfactual --split train \
    --out checkpoints/counterfactuals_train.pt \
    --samples-dir checkpoints/cf_samples_train
python stage2_counterfactual.py counterfactual --split test \
    --out checkpoints/counterfactuals_test.pt \
    --samples-dir checkpoints/cf_samples_test

# 3. 概念标签（查表）
python stage3_concept_labels.py

# 4. 概念关联（lasso, 像素级 V_k）
python stage4_concept_association.py
# 默认 lam=0.05, hard_threshold=0.01。要更稀疏: --lam 0.1 --hard-threshold 0.02

# 5. 因果排序（IE/DE/ATE）
python stage5_causal_ranking.py

# 5b. 随机 V_k 消融（论文 sanity check）
python stage5_ablation.py
# 可选: --n-trials 10

# 6. 决策树替代解释
python stage6_surrogate_tree.py
```

---

## 7. 关键实现细节

- **像素级 V_k**：V_k ∈ {0,1}^(64,7,7)，按 (channel, position) 精确替换，而不是整通道。lasso 在 3136 维上稀疏。
- **L1 实现**：自写 GPU proximal gradient + 训练后 hard-threshold 清扫小残差。sklearn liblinear 在 60000×3136 上 OOM。
- **IE/DE 数值过滤**：相对版分母为 `p_x`，需剔除 `p_x < min_conf`（默认 0.5）的样本。
- **统计前提**：`IE + DE ≈ ATE` 应大致成立（不严格相等，相对差形式会偏大）。`ATE ≈ 0.999` 表示 cf 几乎肯定翻转决策。

---

## 8. 已知局限

1. MNIST 12 个概念几乎一一对应 10 类数字，概念间冗余度比胸片高。论文里"按 IE 排序逐个加入概念，召回率先升后降"的现象在 MNIST 上不出现（单调上升到上限）。
2. 反事实由 VAE 生成，不是论文的 cGAN。语义保真度有限，但 flip rate ≈ 100% 满足消融实验要求。
3. 决策树替代主要受概念分类器质量上限约束（每概念 AUC ≈ 0.997），上限召回率 ≈ 0.97。
