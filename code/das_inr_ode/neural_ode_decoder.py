"""Neural ODE 解码器与 PINN 物理约束。"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    odeint = None
    odeint_adjoint = None


class SpectralLinear(nn.Module):
    """谱归一化线性层，约束 Lipschitz 常数。"""

    def __init__(self, in_f: int, out_f: int, lipschitz_max: float = 10.0):
        super().__init__()
        self.lipschitz_max = lipschitz_max
        self.linear = spectral_norm(nn.Linear(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) / max(1.0, self.lipschitz_max / 10.0)


class ODEFunc(nn.Module):
    """动力学函数 dh/dt = f(h, t, z)。"""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int = 3,
        lipschitz_max: float = 10.0,
    ):
        super().__init__()
        in_dim = hidden_dim + latent_dim + 1
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(n_layers - 1):
            layers += [SpectralLinear(d, hidden_dim, lipschitz_max), nn.Tanh()]
            d = hidden_dim
        layers.append(SpectralLinear(d, hidden_dim, lipschitz_max))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._z: torch.Tensor | None = None

    def set_condition(self, z: torch.Tensor):
        self._z = z

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self._z is None:
            raise RuntimeError("Call set_condition(z) before ODE integration.")
        squeeze = h.dim() == 1
        if squeeze:
            h = h.unsqueeze(0)
        batch = h.shape[0]
        t_val = t.detach().item() if isinstance(t, torch.Tensor) else float(t)
        t_expand = torch.full((batch, 1), t_val, device=h.device, dtype=h.dtype)
        z = self._z.unsqueeze(0) if self._z.dim() == 1 else self._z
        z = z.expand(batch, -1)
        inp = torch.cat([h, z, t_expand], dim=-1)
        out = self.net(inp)
        return out.squeeze(0) if squeeze else out


class InitialStateMapper(nn.Module):
    """潜向量 -> ODE 初始隐藏状态 (≤10K 参数)。"""

    def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        dims = [latent_dim] + [64] * (n_layers - 1) + [hidden_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z)


class SpatioTemporalDecoderHead(nn.Module):
    """
    融合 h(t), z, x, t 的时空解码头。
    解决旧版 signal_head(h) 仅产生秩-1 时空分离的问题。
    """

    def __init__(self, hidden_dim: int, latent_dim: int, n_units: int = 128):
        super().__init__()
        in_dim = hidden_dim + latent_dim + 4  # x, t, sin(2pi t), cos(2pi t)
        self.net = nn.Sequential(
            nn.Linear(in_dim, n_units),
            nn.ReLU(inplace=True),
            nn.Linear(n_units, n_units),
            nn.ReLU(inplace=True),
            nn.Linear(n_units, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        h: (N, hidden_dim)
        z: (N, latent_dim) or (latent_dim,) broadcast
        coords: (N, 2) 归一化 [x, t]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(h.shape[0], -1)
        t = coords[:, 1:2]
        feat = torch.cat([h, z, coords, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1)
        return self.net(feat).squeeze(-1)


class SignalDecoderHead(nn.Module):
    """隐藏状态 -> 空间通道信号 (时间切片重建)。"""

    def __init__(self, hidden_dim: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_channels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NeuralODEDecoder(nn.Module):
    """Neural ODE 连续时间解码器。"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        n_channels: int = 256,
        map_layers: int = 2,
        dynamics_layers: int = 3,
        lipschitz_max: float = 10.0,
        solver: Literal["dopri5", "euler"] = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        use_spatiotemporal_head: bool = True,
        head_units: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_spatiotemporal_head = use_spatiotemporal_head

        self.mapper = InitialStateMapper(latent_dim, hidden_dim, map_layers)
        self.ode_func = ODEFunc(hidden_dim, latent_dim, dynamics_layers, lipschitz_max)
        if use_spatiotemporal_head:
            self.signal_head = SpatioTemporalDecoderHead(hidden_dim, latent_dim, head_units)
        else:
            self.signal_head = SignalDecoderHead(hidden_dim, n_channels)

    def integrate(
        self,
        z: torch.Tensor,
        t_eval: torch.Tensor,
        method: str | None = None,
        euler_step: float | None = None,
    ) -> torch.Tensor:
        """沿 t_eval 积分，返回 (T, hidden_dim)。"""
        h0 = self.mapper(z)
        if h0.dim() == 1:
            h0 = h0.unsqueeze(0)
        self.ode_func.set_condition(z if z.dim() == 1 else z.squeeze(0))

        method = method or self.solver
        if method == "euler" or odeint is None:
            return self._euler_integrate(h0.squeeze(0), t_eval, step=euler_step or 0.02)

        integrator = odeint_adjoint if z.requires_grad else odeint
        out = integrator(
            self.ode_func,
            h0.squeeze(0),
            t_eval,
            method="dopri5" if method == "dopri5" else method,
            rtol=self.rtol,
            atol=self.atol,
        )
        return out

    def _euler_integrate(self, h0: torch.Tensor, t_eval: torch.Tensor, step: float = 0.02) -> torch.Tensor:
        states = [h0]
        h = h0
        for i in range(1, len(t_eval)):
            dt = float(t_eval[i] - t_eval[i - 1])
            n_sub = max(1, int(abs(dt) / step))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                h = h + sub_dt * self.ode_func(t_eval[i - 1], h)
            states.append(h)
        return torch.stack(states, dim=0)

    def reconstruct(
        self,
        z: torch.Tensor,
        n_samples: int,
        n_channels: int | None = None,
        batch_size: int = 8192,
        **kwargs,
    ) -> torch.Tensor:
        """重建 (n_channels, n_samples) 时空场。"""
        nc = n_channels or self.n_channels
        device = z.device
        t_eval = torch.linspace(0, 1, n_samples, device=device)
        traj = self.integrate(z, t_eval, **kwargs)  # (T, H)

        if not self.use_spatiotemporal_head:
            signals = self.signal_head(traj)
            return signals.T

        x_norm = torch.linspace(0, 1, nc, device=device)
        z_vec = z if z.dim() == 1 else z.squeeze(0)
        X, T = torch.meshgrid(x_norm, t_eval, indexing="ij")
        coords = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
        # traj[ti] 对应每个 (c, ti)
        h = traj.unsqueeze(0).expand(nc, -1, -1).reshape(-1, self.hidden_dim)
        z_batch = z_vec.unsqueeze(0).expand(nc * n_samples, -1)
        out = self.signal_head(h, z_batch, coords)
        return out.reshape(nc, n_samples)


def wave_equation_residual(u: torch.Tensor, dx: float = 1.0, dt: float = 1.0, c: float = 1500.0) -> torch.Tensor:
    """离散弹性波方程残差 ∂²u/∂t² - c²∂²u/∂x²。"""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    u_tt = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dt ** 2)
    u_xx = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / (dx ** 2)
    min_t, min_x = min(u_tt.shape[1], u_xx.shape[1]), min(u_tt.shape[2], u_xx.shape[2])
    return u_tt[:, :min_t, :min_x] - (c ** 2) * u_xx[:, :min_t, :min_x]


def heat_diffusion_residual(u: torch.Tensor, dx: float = 1.0, dt: float = 1.0, kappa: float = 1e-5) -> torch.Tensor:
    """热扩散方程残差 ∂u/∂t - κ∂²u/∂x²。"""
    if u.dim() == 2:
        u = u.unsqueeze(0)
    u_t = (u[:, :, 1:] - u[:, :, :-1]) / dt
    u_xx = (u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]) / (dx ** 2)
    min_x = min(u_t.shape[-1], u_xx.shape[-1])
    return u_t[..., :min_x] - kappa * u_xx[..., :min_x]


def pinn_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    pde_type: str = "wave",
    weight: float = 0.5,
    wave_speed: float = 1500.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """数据拟合 + PDE 残差联合损失。"""
    data_loss = nn.functional.mse_loss(recon, target)
    pde_loss = torch.tensor(0.0, device=recon.device)
    if weight > 0 and recon.shape[-1] > 4 and recon.shape[0] > 4:
        if pde_type in ("wave", "dual"):
            res = wave_equation_residual(recon.unsqueeze(0), c=wave_speed)
            pde_loss = pde_loss + res.pow(2).mean()
        if pde_type in ("heat_diffusion", "dual"):
            res_h = heat_diffusion_residual(recon.unsqueeze(0))
            pde_loss = pde_loss + res_h.pow(2).mean()
    total = data_loss + weight * pde_loss
    return total, {"data": float(data_loss.item()), "pde": float(pde_loss.item())}
