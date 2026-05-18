"""通用工具：CNN/VAE 调用、归一化、数据集抽样、日志格式化。

日志辅助函数 (header/section/kv/table_header/table_row) 为整个 pipeline
提供统一的科研感输出风格：双横线分隔的章节、对齐的键值对、表格化的结果。
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net, VAE


# ---------------------------------------------------------------------------
# MNIST 归一化常量
# ---------------------------------------------------------------------------

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

LINE_WIDTH = 78


def to_normalized(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] 像素 -> CNN 训练用的归一化空间。"""
    return (x01 - MNIST_MEAN) / MNIST_STD


# ---------------------------------------------------------------------------
# 模型 / 数据辅助
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify(net: Net, x: torch.Tensor, normalize: bool = True):
    """x 默认 [0,1] 像素；返回 (probs (B,10), preds (B,))。"""
    net.eval()
    inp = to_normalized(x) if normalize else x
    logits = net(inp)
    probs = F.softmax(logits, dim=1)
    return probs, probs.argmax(1)


def pick_random(dataset, n: int, seed: int):
    """从数据集里按 seed 随机抽 n 张，返回 (xs, ys)。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    xs, ys = zip(*[dataset[i] for i in idx])
    return torch.stack(xs, dim=0), torch.tensor(ys)


def pick_one_per_class(dataset, num_classes: int = 10):
    """每个类别取第 1 张，返回 (xs (10,1,28,28), ys (10,))。"""
    picked = {}
    for img, y in dataset:
        if y not in picked:
            picked[y] = img
        if len(picked) == num_classes:
            break
    xs = [picked[i] for i in range(num_classes)]
    ys = torch.tensor(list(range(num_classes)))
    return torch.stack(xs, dim=0), ys


def get_mnist(data_dir: str, train: bool):
    """返回 [0,1] 像素的 MNIST dataset（不做 Normalize）。"""
    return datasets.MNIST(data_dir, train=train, download=True,
                          transform=transforms.ToTensor())


def load_models(cnn_path: str, vae_path: str, latent_dim: int, device):
    net = Net().to(device)
    net.load_state_dict(torch.load(cnn_path, map_location=device))
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    return net, vae


# ---------------------------------------------------------------------------
# 日志格式化
# ---------------------------------------------------------------------------

def header(title: str, char: str = "=") -> None:
    """主标题：双横线包裹。"""
    print()
    print(char * LINE_WIDTH)
    print(title)
    print(char * LINE_WIDTH)


def section(title: str) -> None:
    """子章节：单破折线分隔。"""
    print()
    print(f"-- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 4))


def kv(key: str, value, width: int = 24) -> None:
    """打印一对键值。"""
    print(f"  {key:<{width}s} {value}")


def table_header(columns: list) -> None:
    """打印表头：[(name, width, align), ...]，align ∈ {'<', '>', '^'}。"""
    parts = []
    for name, w, align in columns:
        parts.append(f"{name:{align}{w}s}")
    line = "  ".join(parts)
    print()
    print(line)
    print("-" * len(line))


def table_row(values: list, columns: list) -> None:
    """配合 table_header，按列宽对齐打印一行。"""
    parts = []
    for v, (_, w, align) in zip(values, columns):
        if isinstance(v, str):
            parts.append(f"{v:{align}{w}s}")
        elif isinstance(v, int):
            parts.append(f"{v:{align}{w}d}")
        elif isinstance(v, float):
            parts.append(f"{v:{align}{w}.4f}")
        else:
            parts.append(f"{str(v):{align}{w}s}")
    print("  ".join(parts))


def saved(path) -> None:
    """统一格式打印保存路径。"""
    from pathlib import Path
    print(f"\n[saved] {Path(path).resolve()}")
