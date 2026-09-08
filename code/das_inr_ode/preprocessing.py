"""物理预处理: 均值剥离、DWT/DCT、LPC、受控量化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pywt
from scipy.fftpack import dct, idct
from scipy.signal import lfilter


@dataclass
class PreprocessMetadata:
    channel_means: np.ndarray
    channel_max: np.ndarray
    lpc_coeffs: list[np.ndarray] = field(default_factory=list)
    quant_scales: dict[str, float] = field(default_factory=dict)


def mean_removal(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐通道均值剥离。"""
    means = data.mean(axis=-1, keepdims=True)
    return data - means, means.squeeze(-1)


def dwt_dct_fusion(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> tuple[np.ndarray, list]:
    """单通道 DWT + 低频 DCT 融合。"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx = coeffs[0]
    approx_dct = dct(approx, norm="ortho")
    coeffs_dct = [approx_dct] + list(coeffs[1:])
    return np.concatenate([c.ravel() for c in coeffs_dct]), coeffs_dct


def inverse_dwt_dct_fusion(coeffs_dct: list, wavelet: str = "db4") -> np.ndarray:
    approx = idct(coeffs_dct[0], norm="ortho")
    restored = [approx] + [np.array(c) for c in coeffs_dct[1:]]
    return pywt.waverec(restored, wavelet)


def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin 算法求 LPC 系数。"""
    r = np.asarray(r, dtype=np.float64)
    if r[0] == 0:
        return np.zeros(order)
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(1, i))
        k = -(r[i] + acc) / (e + 1e-12)
        a_old = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_old[j] + k * a_old[i - j]
        e *= 1 - k * k
    return a[1:]


def linear_predictive_coding(
    channel_coeffs: np.ndarray,
    order: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """利用相邻通道空间相干性做线性预测，返回残差与预测系数。"""
    nc = channel_coeffs.shape[0]
    residuals = np.zeros_like(channel_coeffs)
    lpc_list: list[np.ndarray] = []
    for i in range(nc):
        if i == 0:
            residuals[i] = channel_coeffs[i]
            lpc_list.append(np.zeros(order))
            continue
        ref = channel_coeffs[max(0, i - order) : i].mean(axis=0)
        r = np.correlate(ref, ref, mode="full")
        r = r[r.size // 2 : r.size // 2 + order + 1]
        a = levinson_durbin(r, order)
        pred = np.zeros_like(channel_coeffs[i])
        if i >= order:
            pred[order:] = lfilter([0] + (-a).tolist(), [1], ref[order - 1 : -1])
        residuals[i] = channel_coeffs[i] - pred
        lpc_list.append(a)
    return residuals, np.stack(lpc_list, axis=0)


def controlled_quantize(
    data: np.ndarray,
    epsilon: float,
    original_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """受控量化: 步长由劣化阈值约束。"""
    peak = np.max(np.abs(data)) + 1e-12
    scale = max(peak * epsilon, 1e-8)
    q = np.round(data / scale) * scale
    if original_scale is not None:
        recon_err = np.max(np.abs(data - q))
        if recon_err > original_scale * epsilon:
            scale *= 0.5
            q = np.round(data / scale) * scale
    return q.astype(np.float32), float(scale)


def preprocess_spatiotemporal(
    data: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, PreprocessMetadata]:
    """
    完整预处理流水线。
    data: (n_channels, n_samples)
    返回压缩友好的系数矩阵与元数据。
    """
    pp = cfg.get("preprocessing", {})
    wavelet = pp.get("dwt_wavelet", "db4")
    level = int(pp.get("dwt_level", 4))
    lpc_order = int(pp.get("lpc_order", 10))
    thresholds = pp.get("degradation_thresholds", {})

    centered, means = mean_removal(data.astype(np.float64))
    peak = np.max(np.abs(centered), axis=-1, keepdims=True)
    peak[peak == 0] = 1.0
    normalized = centered / peak

    channel_coeffs = []
    coeff_shapes = []
    for ch in range(normalized.shape[0]):
        flat, parts = dwt_dct_fusion(normalized[ch], wavelet, level)
        channel_coeffs.append(flat)
        coeff_shapes.append([p.shape for p in parts])

    coeff_matrix = np.stack(channel_coeffs, axis=0)
    residuals, lpc = linear_predictive_coding(coeff_matrix, lpc_order)

    eps_noise = float(thresholds.get("noise", 1e-3))
    q_data, q_scale = controlled_quantize(residuals, eps_noise)

    meta = PreprocessMetadata(
        channel_means=means.astype(np.float32),
        channel_max=peak.squeeze(-1).astype(np.float32),
        lpc_coeffs=[lpc[i] for i in range(lpc.shape[0])],
        quant_scales={"noise": q_scale},
    )
    meta.coeff_shapes = coeff_shapes  # type: ignore[attr-defined]
    meta.wavelet = wavelet  # type: ignore[attr-defined]
    meta.dwt_level = level  # type: ignore[attr-defined]
    return q_data.astype(np.float32), meta


def inverse_preprocess(
    q_data: np.ndarray,
    meta: PreprocessMetadata,
    n_samples: int,
) -> np.ndarray:
    """逆预处理 (近似重建)。"""
    wavelet = getattr(meta, "wavelet", "db4")
    level = getattr(meta, "dwt_level", 4)
    coeff_shapes = getattr(meta, "coeff_shapes", None)

    restored_channels = []
    for ch in range(q_data.shape[0]):
        flat = q_data[ch]
        if coeff_shapes is not None:
            parts = []
            idx = 0
            for shape in coeff_shapes[ch]:
                size = int(np.prod(shape))
                parts.append(flat[idx : idx + size].reshape(shape))
                idx += size
            sig = inverse_dwt_dct_fusion(parts, wavelet)
            sig = sig[:n_samples]
        else:
            sig = flat[:n_samples]
        restored_channels.append(sig)

    norm = np.stack(restored_channels, axis=0)
    data = norm * meta.channel_max[:, None] + meta.channel_means[:, None]
    return data.astype(np.float32)
