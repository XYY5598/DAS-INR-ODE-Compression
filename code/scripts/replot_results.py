#!/usr/bin/env python
"""从已保存的 .npy 重新绘制全部结果图像（含扩展分析图）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.visualization import (
    plot_absolute_error,
    plot_comparison,
    plot_error_histogram,
    plot_fft_spectrum_comparison,
    plot_metrics_summary,
    plot_per_channel_rmse,
    plot_per_time_rmse,
    plot_scatter_original_vs_recon,
    plot_short_time_energy,
    plot_spectrogram_comparison,
    plot_strain_field,
    plot_time_slice_wiggle,
    plot_waveform_channel,
)

from das_inr_ode.paths import OUTPUTS


def main():
    out_dir = OUTPUTS / "real_data_validation"
    data = np.load(out_dir / "original.npy")
    recon = np.load(out_dir / "reconstructed.npy")
    title = "Cook Inlet KKFLS — 海洋"
    nc, ns = data.shape
    ch = nc // 2
    t_event = ns // 3  # 地震事件大致区域

    metrics = {}
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = raw.get("metrics", raw)

    plots = [
        ("01_original.png", lambda: plot_strain_field(data, f"原始 — {title}", out_dir / "01_original.png")),
        ("02_reconstructed.png", lambda: plot_strain_field(recon, f"重建 — {title}", out_dir / "02_reconstructed.png")),
        ("03_comparison.png", lambda: plot_comparison(data, recon, title, out_dir / "03_comparison.png")),
        ("04_waveform_channel.png", lambda: plot_waveform_channel(data, recon, ch, title, out_dir / "04_waveform_channel.png")),
        ("05_absolute_error.png", lambda: plot_absolute_error(data, recon, title, out_dir / "05_absolute_error.png")),
        ("06_error_histogram.png", lambda: plot_error_histogram(data, recon, title, out_dir / "06_error_histogram.png")),
        ("07_scatter.png", lambda: plot_scatter_original_vs_recon(data, recon, title, out_dir / "07_scatter.png")),
        ("08_per_channel_rmse.png", lambda: plot_per_channel_rmse(data, recon, title, out_dir / "08_per_channel_rmse.png")),
        ("09_per_time_rmse.png", lambda: plot_per_time_rmse(data, recon, title, out_dir / "09_per_time_rmse.png")),
        ("10_spectrogram.png", lambda: plot_spectrogram_comparison(data, recon, ch, title, out_dir / "10_spectrogram.png")),
        ("11_fft_spectrum.png", lambda: plot_fft_spectrum_comparison(data, recon, ch, title, out_dir / "11_fft_spectrum.png")),
        ("12_time_slice_wiggle.png", lambda: plot_time_slice_wiggle(data, recon, t_event, title, out_dir / "12_time_slice_wiggle.png")),
        ("13_short_time_energy.png", lambda: plot_short_time_energy(data, recon, title, out_dir / "13_short_time_energy.png")),
    ]
    if metrics:
        plots.append(
            ("14_metrics_summary.png", lambda: plot_metrics_summary(metrics, title, out_dir / "14_metrics_summary.png"))
        )

    for name, fn in plots:
        fn()
        print(f"  OK {name}")

    print(f"\n共生成 {len(plots)} 张图像: {out_dir}")


if __name__ == "__main__":
    main()
