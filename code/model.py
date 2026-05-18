"""
两段式 CNN，仿原论文 DenseNet-121 的切分。

Phi1（特征层）：x -> z = Phi1(x)，shape (B, 64, 7, 7)
    64 个通道即后续因果分析的 "unit"
    在 z 上做 lasso 概念关联（Step 1）和 do-intervention（Step 2）

Phi2（决策层）：z -> logits，shape (B, 10)

VAE：用于生成反事实 x'，潜空间 16 维
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phi1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x


class Phi2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, z):
        z = F.relu(self.conv3(z))
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)


class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.phi1 = Phi1()
        self.phi2 = Phi2(num_classes=num_classes)

    def forward(self, x):
        return self.phi2(self.phi1(x))


class VAE(nn.Module):
    """
    标准 conv VAE，潜空间 16 维。
    输入 x ∈ [0,1]，shape (B, 1, 28, 28)。
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)   # 28 -> 14
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 14 -> 7
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2,
                                            padding=1, output_padding=1)  # 7 -> 14
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2,
                                            padding=1, output_padding=1)  # 14 -> 28

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec(z)).view(-1, 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """BCE 重建 + KL 散度，按 batch 求和（与论文常用形式一致）。"""
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld
