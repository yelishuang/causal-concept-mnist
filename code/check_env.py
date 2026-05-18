"""Environment / GPU sanity check."""

import platform
import sys

import torch

from utils import header, section, kv


def main():
    header("Environment")
    kv("Python", sys.version.split()[0])
    kv("Platform", platform.platform())
    kv("PyTorch", torch.__version__)
    kv("CUDA build", str(torch.version.cuda))

    header("CUDA detection")
    avail = torch.cuda.is_available()
    kv("torch.cuda.is_available()", avail)
    if not avail:
        section("hint")
        print("  - PyTorch may be CPU-only; check `nvidia-smi` and reinstall a CUDA build.")
        return

    n = torch.cuda.device_count()
    kv("device count", n)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        section(f"cuda:{i}")
        kv("name", torch.cuda.get_device_name(i))
        kv("compute capability", f"{props.major}.{props.minor}")
        kv("total memory", f"{props.total_memory / 1e9:.2f} GB")
        kv("multi-processors", props.multi_processor_count)

    header("Quick benchmark on cuda:0")
    device = torch.device("cuda:0")
    x = torch.randn(60000, 3136, device=device)
    kv("60000x3136 fp32 tensor", f"{x.element_size() * x.numel() / 1e6:.1f} MB")
    kv("memory reserved", f"{torch.cuda.memory_reserved() / 1e6:.1f} MB")
    kv("memory allocated", f"{torch.cuda.memory_allocated() / 1e6:.1f} MB")

    import time
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    kv("20 x (2048x2048 matmul)", f"{(time.time() - t0) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
