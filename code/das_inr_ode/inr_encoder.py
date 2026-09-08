"""隐式神经表示 (INR) 编码器。"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class RandomFourierEncoding(nn.Module):
    """随机傅里叶特征映射 (RFF)。"""

    def __init__(self, in_dim: int, n_features: int, scale: float = 10.0):
        super().__init__()
        B = torch.randn(n_features, in_dim) * scale
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * coords @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, n_freqs: int = 10):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim * (2 * n_freqs + 1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        parts = [coords]
        for f in self.freqs:
            parts.append(torch.sin(math.pi * f * coords))
            parts.append(torch.cos(math.pi * f * coords))
        return torch.cat(parts, dim=-1)


class SirenLayer(nn.Linear):
    def __init__(self, in_f: int, out_f: int, omega: float = 30.0, is_first: bool = False):
        super().__init__(in_f, out_f)
        self.omega = omega
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        w_bound = 1 / self.in_features if self.is_first else math.sqrt(6 / self.in_features) / self.omega
        nn.init.uniform_(self.weight, -w_bound, w_bound)
        nn.init.uniform_(self.bias, -w_bound, w_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * super().forward(x))


class LatentINREncoder(nn.Module):
    """
    条件隐式神经表示编码器 f(x,t; z)。
    潜向量 z 经线性层注入各隐藏层 (FiLM 风格缩放)。
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_layers: int = 5,
        n_units: int = 256,
        activation: Literal["siren", "relu"] = "siren",
        coord_encoding: Literal["rff", "positional"] = "rff",
        rff_features: int = 128,
        rff_scale: float = 10.0,
        siren_omega: float = 30.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.activation_type = activation

        if coord_encoding == "rff":
            self.coord_enc = RandomFourierEncoding(2, rff_features, rff_scale)
            enc_dim = self.coord_enc.out_dim
        else:
            self.coord_enc = PositionalEncoding(2, n_freqs=10)
            enc_dim = self.coord_enc.out_dim

        self.latent_proj = nn.Linear(latent_dim, n_units)
        self.film = nn.Linear(latent_dim, n_units * 2)

        layers: list[nn.Module] = []
        in_dim = enc_dim
        for i in range(n_layers):
            if activation == "siren":
                layers.append(SirenLayer(in_dim if i == 0 else n_units, n_units, siren_omega, is_first=(i == 0)))
            else:
                layers.append(nn.Linear(in_dim if i == 0 else n_units, n_units))
                layers.append(nn.ReLU(inplace=True))
            in_dim = n_units
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(n_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, 2) 归一化 [x, t]
        z: (B, latent_dim) 或 (latent_dim,) 广播
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(coords.shape[0], -1)

        h = self.coord_enc(coords)
        gamma_beta = self.film(z)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        for i, layer in enumerate(self.layers):
            if self.activation_type == "siren":
                h = layer(h)
            else:
                h = layer(h) if isinstance(layer, nn.Linear) else layer(h)
            if isinstance(layer, SirenLayer) or (self.activation_type == "relu" and isinstance(layer, nn.ReLU)):
                h = h * (1 + gamma) + beta

        out = self.output(h)
        return 2 * self.sigmoid(out) - 1

    def encode_latent(
        self,
        coords: torch.Tensor,
        targets: torch.Tensor,
        steps: int = 200,
        lr: float = 1e-2,
        z_init: torch.Tensor | None = None,
        n_channels: int | None = None,
        n_samples: int | None = None,
        sample_rate_hz: float = 50.0,
        use_band_loss: bool = False,
        seismic_weight: float = 8.0,
        band_mix: float = 0.15,
    ) -> torch.Tensor:
        """固定网络权重，优化潜向量 z (压缩阶段)。支持 z 热启动与分频段损失。"""
        from das_inr_ode.losses import band_weighted_mse

        was_training = self.training
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        if z_init is not None:
            z0 = z_init.detach().reshape(1, self.latent_dim).to(coords.device).clone()
        else:
            z0 = torch.zeros(1, self.latent_dim, device=coords.device)
        z = torch.nn.Parameter(z0, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        use_grid_loss = use_band_loss and n_channels and n_samples
        for _ in range(steps):
            pred = self.forward(coords, z.squeeze(0))
            if use_grid_loss:
                pred_grid = pred.reshape(n_channels, n_samples)
                target_grid = targets.reshape(n_channels, n_samples)
                spatial = nn.functional.mse_loss(pred_grid, target_grid)
                bw, _ = band_weighted_mse(
                    pred_grid,
                    target_grid,
                    sample_rate_hz=sample_rate_hz,
                    seismic_weight=seismic_weight,
                )
                loss = (1.0 - band_mix) * spatial + band_mix * bw
            else:
                loss = nn.functional.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()

        for p in self.parameters():
            p.requires_grad_(True)
        if was_training:
            self.train()
        return z.detach().squeeze(0)


class INRTrainer:
    """INR 自监督训练器。"""

    def __init__(self, model: LatentINREncoder, lr: float = 1e-4, reg_lambda: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.reg_lambda = reg_lambda

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for coords, targets, z_batch in loader:
            pred = self.model(coords, z_batch)
            mse = nn.functional.mse_loss(pred, targets)
            reg = self.reg_lambda * z_batch.pow(2).mean()
            loss = mse + reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)


def build_coordinate_grid(n_channels: int, n_samples: int, device: torch.device):
    """构建归一化时空坐标网格。"""
    x = torch.linspace(0, 1, n_channels)
    t = torch.linspace(0, 1, n_samples)
    X, T = torch.meshgrid(x, t, indexing="ij")
    coords = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1).to(device)
    return coords


def prepare_training_batch(
    data: np.ndarray,
    latent_dim: int,
    device: torch.device,
    subsample: int | None = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 (C,T) 数据构建训练批次。"""
    nc, ns = data.shape
    coords = build_coordinate_grid(nc, ns, device)
    values = torch.tensor(data, dtype=torch.float32, device=device).reshape(-1, 1)
    z = torch.randn(1, latent_dim, device=device) * 0.01
    z_batch = z.expand(coords.shape[0], -1)

    if subsample and coords.shape[0] > subsample:
        idx = torch.randperm(coords.shape[0], device=device)[:subsample]
        coords, values, z_batch = coords[idx], values[idx], z_batch[idx]

    return coords, values, z_batch
