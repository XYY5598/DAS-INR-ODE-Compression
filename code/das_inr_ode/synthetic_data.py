"""合成 DAS 应变数据生成。"""

from __future__ import annotations

import numpy as np


def generate_synthetic_das(
    n_channels: int = 256,
    n_samples: int = 500,
    sample_rate_hz: float = 50.0,
    scenario: str = "海洋",
    seed: int = 42,
) -> np.ndarray:
    """
    生成合成时空应变场 (n_channels, n_samples)。
    scenario 对应 {{location_theme}} 预设。
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_channels)
    t = np.arange(n_samples) / sample_rate_hz

    data = np.zeros((n_channels, n_samples), dtype=np.float32)

    # 宏观背景 (潮汐/重力波)
    tide = 0.3 * np.sin(2 * np.pi * 0.05 * t)
    data += tide[np.newaxis, :]

    if scenario in ("海洋", "地质活动监测"):
        # 地震波事件
        event_t = n_samples // 3
        for ch in range(n_channels):
            delay = int(ch * 0.02 * sample_rate_hz)
            idx = min(event_t + delay, n_samples - 1)
            width = int(0.5 * sample_rate_hz)
            pulse = np.exp(-((np.arange(n_samples) - idx) ** 2) / (2 * (width / 3) ** 2))
            data[ch] += 0.8 * pulse * np.sin(2 * np.pi * 5 * (t - t[idx]))

    if scenario == "陆地基础设施":
        # 准静态温度应变
        temp = 0.2 * np.sin(2 * np.pi * x)[:, np.newaxis]
        creep = 0.05 * t / t.max()
        data += temp + creep

    if scenario == "地质活动监测":
        # 微震叠加
        for _ in range(3):
            ch0 = rng.integers(0, n_channels - 20)
            t0 = rng.integers(n_samples // 4, 3 * n_samples // 4)
            data[ch0 : ch0 + 20, t0 : t0 + 10] += rng.normal(0, 0.15, (20, min(10, n_samples - t0)))

    # 船舶/环境噪声
    noise_level = 0.02 if scenario == "陆地基础设施" else 0.05
    data += rng.normal(0, noise_level, data.shape).astype(np.float32)

    # 通道基线漂移
    drift = 0.1 * x[:, np.newaxis]
    data += drift.astype(np.float32)
    return data.astype(np.float32)
