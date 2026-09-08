"""损失函数：分频段加权、联合训练。"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rfft_freqs(n: int, fs: float, device: torch.device) -> torch.Tensor:
    return torch.fft.rfftfreq(n, d=1.0 / fs).to(device)


def band_weighted_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    sample_rate_hz: float = 50.0,
    seismic_band: tuple[float, float] = (3.0, 10.0),
    seismic_weight: float = 5.0,
    bg_weight: float = 1.0,
    noise_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    空域 MSE + 频域分频段加权。
    recon/target: (C, T)
    """
    spatial = nn.functional.mse_loss(recon, target)

    # 沿时间维 FFT，按频段加权
    R = torch.fft.rfft(recon, dim=-1)
    T = torch.fft.rfft(target, dim=-1)
    freqs = _rfft_freqs(recon.shape[-1], sample_rate_hz, recon.device)
    err = (R - T).abs() ** 2

    lo, hi = seismic_band
    seismic_mask = (freqs >= lo) & (freqs <= hi)
    bg_mask = (freqs >= 0.1) & (freqs < lo)
    noise_mask = freqs > hi

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return err[..., mask].mean()
        return torch.tensor(0.0, device=recon.device)

    freq_loss = (
        seismic_weight * masked_mean(seismic_mask)
        + bg_weight * masked_mean(bg_mask)
        + noise_weight * masked_mean(noise_mask)
    )
    total = 0.5 * spatial + 0.5 * freq_loss
    return total, {
        "spatial": float(spatial.item()),
        "freq": float(freq_loss.item()),
    }


def combined_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    sample_rate_hz: float = 50.0,
    pinn_weight: float = 0.1,
    pde_type: str = "wave",
    wave_speed: float = 1500.0,
    seismic_weight: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    from das_inr_ode.neural_ode_decoder import pinn_loss

    bw, parts = band_weighted_mse(
        recon, target, sample_rate_hz=sample_rate_hz, seismic_weight=seismic_weight
    )
    if pinn_weight > 0:
        _, pinn_parts = pinn_loss(recon, target, pde_type=pde_type, weight=pinn_weight, wave_speed=wave_speed)
        total = bw + pinn_weight * torch.tensor(pinn_parts["pde"], device=recon.device)
        parts["pde"] = pinn_parts["pde"]
    else:
        total = bw
    parts["total"] = float(total.item())
    return total, parts
