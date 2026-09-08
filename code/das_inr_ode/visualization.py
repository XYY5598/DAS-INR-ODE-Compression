"""可视化与报告生成（中文宋体标注）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_FONT_CONFIGURED = False
_RESOLVED_FONT_PATH: str | None = None
CHINESE_FONT = "SimSun"  # 宋体


def setup_chinese_font() -> str:
    """注册并启用宋体（SimSun / Noto Serif CJK SC / AR PL UMing CN）。"""
    global _FONT_CONFIGURED, _RESOLVED_FONT_PATH
    if _FONT_CONFIGURED:
        return CHINESE_FONT

    font_paths = [
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttf"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),  # AR PL UMing CN，宋体风格
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    font_name = CHINESE_FONT
    for fp in font_paths:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            _RESOLVED_FONT_PATH = str(fp)
            font_name = font_manager.FontProperties(fname=str(fp)).get_name()
            break

    plt.rcParams.update({
        "font.family": font_name,
        "font.serif": [font_name, "Noto Serif CJK SC", "AR PL UMing CN", "SimSun", "STSong", "宋体"],
        "axes.unicode_minus": False,
        "font.size": 11,
    })
    _FONT_CONFIGURED = True
    return font_name


def _font_props(size: float | None = None):
    setup_chinese_font()
    if _RESOLVED_FONT_PATH:
        return font_manager.FontProperties(fname=_RESOLVED_FONT_PATH, size=size)
    return font_manager.FontProperties(family="serif", size=size)


def plot_strain_field(
    data: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """绘制时空应变场，title 使用 {{title_text}}。"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    fp = _font_props()
    ax.set_xlabel("时间采样点", fontproperties=fp)
    ax.set_ylabel("空间通道", fontproperties=fp)
    ax.set_title(title_text, loc="right", fontproperties=_font_props(12))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("应变 (归一化)", fontproperties=fp)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
):
    setup_chinese_font()
    err = original - reconstructed
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fp = _font_props(10)
    for ax, arr, title in zip(
        axes,
        [original, reconstructed, err],
        ["原始", "重建", "误差"],
    ):
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="seismic")
        ax.set_title(f"{title} — {title_text}", fontproperties=fp)
        ax.set_xlabel("时间采样点", fontproperties=_font_props(9))
        ax.set_ylabel("空间通道", fontproperties=_font_props(9))
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_channel(
    original: np.ndarray,
    reconstructed: np.ndarray,
    channel: int,
    title_text: str,
    save_path: str | Path | None = None,
):
    """单通道波形对比图。"""
    setup_chinese_font()
    ns = original.shape[-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    fp = _font_props()
    t = np.arange(ns)
    ax.plot(t, original[channel], label="原始", alpha=0.85, linewidth=0.8)
    ax.plot(t, reconstructed[channel], label="重建", alpha=0.85, linewidth=0.8)
    ax.set_xlabel("采样点", fontproperties=fp)
    ax.set_ylabel("归一化应变", fontproperties=fp)
    ax.set_title(f"通道 {channel} 波形对比 — {title_text}", fontproperties=_font_props(12))
    ax.legend(prop=_font_props(10))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
):
    """绝对误差时空分布图。"""
    setup_chinese_font()
    err = np.abs(original - reconstructed)
    fig, ax = plt.subplots(figsize=(10, 5))
    fp = _font_props()
    im = ax.imshow(err, aspect="auto", origin="lower", cmap="hot")
    ax.set_xlabel("时间采样点", fontproperties=fp)
    ax.set_ylabel("空间通道", fontproperties=fp)
    ax.set_title(f"绝对误差分布 — {title_text}", fontproperties=_font_props(12))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|原始 - 重建|", fontproperties=fp)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
):
    """重建误差直方图。"""
    setup_chinese_font()
    err = (original - reconstructed).ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    fp = _font_props()
    ax.hist(err, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1, label="零误差")
    ax.set_xlabel("重建误差", fontproperties=fp)
    ax.set_ylabel("频数", fontproperties=fp)
    ax.set_title(f"误差分布直方图 — {title_text}", fontproperties=_font_props(12))
    ax.legend(prop=_font_props(10))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_original_vs_recon(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
    max_points: int = 8000,
):
    """原始值 vs 重建值散点图（含 y=x 参考线）。"""
    setup_chinese_font()
    o = original.ravel()
    r = reconstructed.ravel()
    if o.size > max_points:
        idx = np.random.default_rng(0).choice(o.size, max_points, replace=False)
        o, r = o[idx], r[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    fp = _font_props()
    ax.scatter(o, r, s=4, alpha=0.25, c="teal", edgecolors="none")
    lim = max(np.abs(o).max(), np.abs(r).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="理想 y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("原始值", fontproperties=fp)
    ax.set_ylabel("重建值", fontproperties=fp)
    ax.set_title(f"原始–重建散点图 — {title_text}", fontproperties=_font_props(12))
    ax.set_aspect("equal")
    ax.legend(prop=_font_props(10))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_channel_rmse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
):
    """各空间通道 RMSE 曲线。"""
    setup_chinese_font()
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    fp = _font_props()
    ax.plot(np.arange(len(rmse)), rmse, color="darkorange", linewidth=1.2)
    ax.fill_between(np.arange(len(rmse)), rmse, alpha=0.25, color="darkorange")
    ax.set_xlabel("空间通道索引", fontproperties=fp)
    ax.set_ylabel("RMSE", fontproperties=fp)
    ax.set_title(f"逐通道重建 RMSE — {title_text}", fontproperties=_font_props(12))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_time_rmse(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
):
    """各时刻全场 RMSE 曲线。"""
    setup_chinese_font()
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2, axis=0))
    fig, ax = plt.subplots(figsize=(10, 4))
    fp = _font_props()
    ax.plot(np.arange(len(rmse)), rmse, color="purple", linewidth=1.2)
    ax.fill_between(np.arange(len(rmse)), rmse, alpha=0.2, color="purple")
    ax.set_xlabel("时间采样点", fontproperties=fp)
    ax.set_ylabel("RMSE", fontproperties=fp)
    ax.set_title(f"逐时刻重建 RMSE — {title_text}", fontproperties=_font_props(12))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectrogram_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    channel: int,
    title_text: str,
    save_path: str | Path | None = None,
    sample_rate_hz: float = 50.0,
):
    """单通道时频谱（原始 vs 重建）对比。"""
    setup_chinese_font()
    from matplotlib import mlab

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fp = _font_props(10)
    for ax, sig, label in zip(axes, [original[channel], reconstructed[channel]], ["原始", "重建"]):
        Pxx, freqs, bins = mlab.specgram(sig, NFFT=64, Fs=sample_rate_hz, noverlap=32)
        im = ax.pcolormesh(bins, freqs, 10 * np.log10(Pxx + 1e-12), shading="auto", cmap="viridis")
        ax.set_ylabel("频率 (Hz)", fontproperties=fp)
        ax.set_title(f"{label} 时频谱（通道 {channel}）", fontproperties=fp)
        plt.colorbar(im, ax=ax, label="功率 (dB)")
    axes[-1].set_xlabel("时间 (s)", fontproperties=fp)
    fig.suptitle(f"时频域对比 — {title_text}", fontproperties=_font_props(12), y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fft_spectrum_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    channel: int,
    title_text: str,
    save_path: str | Path | None = None,
    sample_rate_hz: float = 50.0,
):
    """单通道平均频谱对比。"""
    setup_chinese_font()
    ns = original.shape[1]
    freqs = np.fft.rfftfreq(ns, d=1.0 / sample_rate_hz)
    spec_o = np.abs(np.fft.rfft(original[channel]))
    spec_r = np.abs(np.fft.rfft(reconstructed[channel]))
    fig, ax = plt.subplots(figsize=(10, 4))
    fp = _font_props()
    ax.semilogy(freqs, spec_o + 1e-12, label="原始", alpha=0.85, linewidth=1)
    ax.semilogy(freqs, spec_r + 1e-12, label="重建", alpha=0.85, linewidth=1, linestyle="--")
    ax.set_xlabel("频率 (Hz)", fontproperties=fp)
    ax.set_ylabel("幅值（对数）", fontproperties=fp)
    ax.set_title(f"频谱对比（通道 {channel}）— {title_text}", fontproperties=_font_props(12))
    ax.legend(prop=_font_props(10))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_slice_wiggle(
    original: np.ndarray,
    reconstructed: np.ndarray,
    time_idx: int,
    title_text: str,
    save_path: str | Path | None = None,
    skip: int = 2,
):
    """某一时刻沿光纤的空间波形（wiggle 叠加）对比。"""
    setup_chinese_font()
    nc = original.shape[0]
    channels = np.arange(0, nc, skip)
    o_slice = original[channels, time_idx]
    r_slice = reconstructed[channels, time_idx]
    scale = 0.8
    fig, ax = plt.subplots(figsize=(10, 8))
    fp = _font_props()
    for i, ch in enumerate(channels):
        ax.plot(o_slice[i] * scale + ch, ch, "b-", linewidth=0.6, alpha=0.7)
        ax.plot(r_slice[i] * scale + ch, ch, "r-", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("归一化应变（偏移叠加）", fontproperties=fp)
    ax.set_ylabel("空间通道", fontproperties=fp)
    ax.set_title(f"时刻 {time_idx} 空间波形 — {title_text}", fontproperties=_font_props(12))
    ax.plot([], [], "b-", label="原始")
    ax.plot([], [], "r-", label="重建")
    ax.legend(prop=_font_props(10), loc="upper right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_short_time_energy(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title_text: str,
    save_path: str | Path | None = None,
    window: int = 25,
):
    """短时能量随时间变化对比（全场平均）。"""
    setup_chinese_font()

    def rolling_energy(x: np.ndarray, w: int) -> np.ndarray:
        e = np.mean(x ** 2, axis=0)
        kernel = np.ones(w) / w
        return np.convolve(e, kernel, mode="same")

    e_o = rolling_energy(original, window)
    e_r = rolling_energy(reconstructed, window)
    fig, ax = plt.subplots(figsize=(10, 4))
    fp = _font_props()
    t = np.arange(len(e_o))
    ax.plot(t, e_o, label="原始", color="steelblue", linewidth=1.2)
    ax.plot(t, e_r, label="重建", color="coral", linewidth=1.2, linestyle="--")
    ax.set_xlabel("时间采样点", fontproperties=fp)
    ax.set_ylabel("短时能量（归一化）", fontproperties=fp)
    ax.set_title(f"短时能量对比 — {title_text}", fontproperties=_font_props(12))
    ax.legend(prop=_font_props(10))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(
    metrics: dict,
    title_text: str,
    save_path: str | Path | None = None,
):
    """压缩比 / PSNR / RMSE 等指标汇总条形图。"""
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fp = _font_props(10)

    items = [
        ("PSNR (dB)", metrics.get("psnr_db", 0), "steelblue"),
        ("压缩比 (×)", metrics.get("compression_ratio", 0), "seagreen"),
        ("RMSE", metrics.get("rmse", 0), "darkorange"),
    ]
    for ax, (label, val, color) in zip(axes, items):
        ax.bar([label], [val], color=color, width=0.5)
        ax.set_title(label, fontproperties=fp)
        ax.text(0, val, f"{val:.2f}", ha="center", va="bottom", fontproperties=_font_props(11))
    fig.suptitle(f"重建性能指标 — {title_text}", fontproperties=_font_props(12))
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
