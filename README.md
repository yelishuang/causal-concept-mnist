# causal-concept-mnist

**MICCAI 2021 论文 _Using Causal Analysis for Conceptual Deep Learning Explanation_ 的 MNIST 最小复现沙箱。**

> Singla, S., Wallace, S., Triantafillou, S., Batmanghelich, K.
> *Using Causal Analysis for Conceptual Deep Learning Explanation.* MICCAI 2021.

把原论文在胸部 X 光 (MIMIC-CXR + DenseNet-121) 上的"概念关联 → 因果中介排序 → 决策树替代"三步 pipeline，搬到 MNIST 上做最小可运行复现，方便理解方法本身。

---

## 范围声明

| 论文环节 | 本仓库做法 |
|---|---|
| 数据集 | MNIST（替代 MIMIC-CXR） |
| 黑盒分类器 | 简单两段式 CNN `Φ₂ ∘ Φ₁`（替代 DenseNet-121） |
| 概念标注 | 数字 → 12 维二值向量查表（替代 NLP 抽影像学征象） |
| 反事实 x' | VAE 潜空间梯度优化（**替代**论文 cGAN，Singla 2019 ICLR） |
| 概念关联 (公式 1) | ✅ Lasso，像素级 V_k |
| 因果中介 ATE/DE/IE (公式 2-4) | ✅ |
| 决策树替代 (公式 5-6) | ✅ |
| Sanity check：随机 V_k 消融 | ✅ |

**没复现**：cGAN 反事实、MIMIC-CXR、DenseNet-121。原因见 [code/docs/HANDOVER.md](code/docs/HANDOVER.md) §1。

---

## 仓库结构

```
.
├── code/
│   ├── stage1_cnn.py                  # 训练黑盒 CNN
│   ├── stage2_counterfactual.py       # VAE + 生成反事实
│   ├── stage3_concept_labels.py       # 数字 → 12 概念查表
│   ├── stage4_concept_association.py  # 公式 (1) Lasso 找概念单元 V_k
│   ├── stage5_causal_ranking.py       # 公式 (2)(3)(4) ATE/DE/IE
│   ├── stage5_ablation.py             # 随机 V_k 消融对照
│   ├── stage6_surrogate_tree.py       # 公式 (5)(6) 决策树替代
│   ├── model.py / utils.py / check_env.py
│   └── docs/HANDOVER.md               # 完整复现文档（强烈建议先读这份）
├── markdown/
│   ├── paper.md                       # 论文中文译文（个人学习用途，见下方声明）
│   └── images/                        # 论文 6 张图
└── README.md
```

> 注：`code/checkpoints/`（模型权重 + 反事实 .pt 文件，约 430 MB）和 `code/data/`（MNIST 原始数据）未入库，按下方"快速开始"运行脚本会自动重新生成 / 下载。

---

## 快速开始

```bash
cd code

# 0. 环境检查（torch, sklearn 等）
python check_env.py

# 1. 训练 CNN
python stage1_cnn.py

# 2. 训练 VAE + 生成反事实
python stage2_counterfactual.py train
python stage2_counterfactual.py counterfactual --split train \
    --out checkpoints/counterfactuals_train.pt
python stage2_counterfactual.py counterfactual --split test \
    --out checkpoints/counterfactuals_test.pt

# 3-6. 概念标签 → 关联 → 因果排序 → 决策树
python stage3_concept_labels.py
python stage4_concept_association.py
python stage5_causal_ranking.py
python stage5_ablation.py          # 可选：sanity check
python stage6_surrogate_tree.py
```

各 stage 的输入输出、参数、关键实现细节见 [code/docs/HANDOVER.md](code/docs/HANDOVER.md)。

---

## 已知局限

1. MNIST 12 个概念几乎和 10 类数字一一对应，概念冗余度比胸片低很多——论文中"按 IE 排序逐个加入概念，召回率先升后降"的现象在 MNIST 上不出现（单调上升）。
2. 反事实由 VAE 生成而非 cGAN，语义保真度有限，但 flip rate ≈ 100%、ATE ≈ 0.999，足以支撑消融。
3. 决策树替代受概念分类器精度上限约束（每概念 AUC ≈ 0.997），上限召回率 ≈ 0.97。

---

## 关于译文版权

`markdown/paper.md` 是论文的**个人学习用中文译文 + 整理批注**，并非原文逐字搬运：
- 公式编号、图片引用沿用原文
- 用于辅助本人理解，不作商业 / 公开传播用途
- 原文版权归原作者及 MICCAI 2021 所有

如有版权方异议，请提 issue，我会立即移除该文件。

---

## 引用

如使用本仓库的复现思路，请同时引用原论文：

```bibtex
@inproceedings{singla2021causal,
  title     = {Using Causal Analysis for Conceptual Deep Learning Explanation},
  author    = {Singla, Sumedha and Wallace, Stephen and Triantafillou, Sofia and Batmanghelich, Kayhan},
  booktitle = {MICCAI},
  year      = {2021}
}
```
