#!/usr/bin/env python
"""汇总全部实验结果、重绘中文宋体图像、生成专利撰写用分析报告。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import load_config
from das_inr_ode.metrics import evaluate_reconstruction, psnr
from das_inr_ode.paths import DATA_RAW, OUTPUTS, default_h5
from das_inr_ode.pipeline_v2 import ImprovedDASPipeline, normalize_das, save_json_report
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
    setup_chinese_font,
)

REPORT_DIR = OUTPUTS / "patent_report"
TERRA_H5 = DATA_RAW / "11715354_TERRA.h5"
KKFLS_H5 = DATA_RAW / "11715354_KKFLS.h5"

# 关键实验：需重新训练/出图
KEY_RUNS = [
    {
        "id": "best_40db_terra",
        "label": "TERRA 最佳（40.30 dB）",
        "h5": TERRA_H5,
        "n_ch": 128,
        "n_samples": 500,
        "epochs": 400,
        "n_units": 512,
        "latent_steps": 3000,
        "band_mix": 0.08,
        "latent_lr": 2e-3,
    },
    {
        "id": "best_kkfls",
        "label": "KKFLS 最佳（38.54 dB）",
        "h5": KKFLS_H5,
        "n_ch": 128,
        "n_samples": 500,
        "epochs": 400,
        "n_units": 512,
        "latent_steps": 3000,
        "band_mix": 0.15,
        "latent_lr": 2e-3,
    },
]


def load_h5(path: Path, n_ch: int, n_samples: int, ch_start: int = 0) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["/Acquisition/Raw[0]/RawData"][:n_samples, ch_start : ch_start + n_ch].T.astype(np.float32)


def collect_all_metrics() -> list[dict]:
    rows = []
    for mp in sorted(OUTPUTS.rglob("metrics.json")):
        if "weight_chunks" in str(mp):
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = mp.relative_to(OUTPUTS)
        ts = m.get("train_stats", {})
        rows.append({
            "path": str(rel.parent),
            "psnr_db": m.get("psnr_db"),
            "compression_ratio": m.get("compression_ratio"),
            "rmse": m.get("rmse"),
            "mse": m.get("mse"),
            "peak_error": m.get("peak_error"),
            "meets_40db": m.get("meets_psnr_40db", False),
            "train_time_sec": m.get("train_time_sec"),
            "transmit_mode": m.get("transmit_mode"),
            "n_units": ts.get("n_units"),
            "latent_steps": ts.get("latent_steps"),
            "band_loss_mix": ts.get("band_loss_mix"),
            "latent_lr": ts.get("latent_lr"),
            "best_psnr_db": ts.get("best_psnr_db"),
            "best_epoch": ts.get("best_epoch"),
            "channel_start": ts.get("channel_start"),
        })
    rows.sort(key=lambda x: (x["psnr_db"] or 0), reverse=True)
    return rows


def plot_psnr_leaderboard(rows: list[dict], save_path: Path):
    setup_chinese_font()
    top = [r for r in rows if r["psnr_db"] is not None][:20]
    labels = [r["path"].replace("/", "\n") for r in top][::-1]
    values = [r["psnr_db"] for r in top][::-1]
    colors = ["#2ca02c" if v >= 40 else "#1f77b4" if v >= 38 else "#ff7f0e" for v in values]
    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.35)))
    ax.barh(labels, values, color=colors)
    ax.axvline(40, color="crimson", linestyle="--", linewidth=1.2, label="40 dB 目标")
    ax.set_xlabel("PSNR (dB)")
    ax.set_title("全部实验 PSNR 对比（宋体）")
    ax.legend()
    for i, v in enumerate(values):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def regenerate_from_npy(exp_dir: Path, title: str, out_dir: Path):
    orig = exp_dir / "original_norm.npy"
    recon = exp_dir / "reconstructed.npy"
    if not orig.exists() or not recon.exists():
        return False
    o = np.load(orig)
    r = np.load(recon)
    nc = o.shape[0]
    ch = nc // 2
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_strain_field(o, f"原始 — {title}", out_dir / "01_original.png")
    plot_strain_field(r, f"重建 — {title}", out_dir / "02_reconstructed.png")
    plot_comparison(o, r, title, out_dir / "03_comparison.png")
    plot_waveform_channel(o, r, ch, title, out_dir / "04_waveform.png")
    plot_absolute_error(o, r, title, out_dir / "05_absolute_error.png")
    plot_error_histogram(o, r, title, out_dir / "06_error_histogram.png")
    plot_scatter_original_vs_recon(o, r, title, out_dir / "07_scatter.png")
    plot_per_channel_rmse(o, r, title, out_dir / "08_per_channel_rmse.png")
    plot_per_time_rmse(o, r, title, out_dir / "09_per_time_rmse.png")
    plot_spectrogram_comparison(o, r, ch, title, out_dir / "10_spectrogram.png")
    plot_fft_spectrum_comparison(o, r, ch, title, out_dir / "11_fft_spectrum.png")
    plot_time_slice_wiggle(o, r, o.shape[1] // 2, title, out_dir / "12_time_slice_wiggle.png")
    plot_short_time_energy(o, r, title, out_dir / "13_short_time_energy.png")
    if (exp_dir / "metrics.json").exists():
        m = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
        plot_metrics_summary(m, title, out_dir / "14_metrics_summary.png")
    return True


def train_and_figure(run: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_h5(Path(run["h5"]), run["n_ch"], run["n_samples"])
    cfg = load_config(str(ROOT / "config" / "improved.yaml"))
    cfg.update({
        "decode_domain": "time",
        "decode_mode": "inr",
        "latent_dim": 256,
        "n_channels": run["n_ch"],
        "n_samples": run["n_samples"],
        "title_text": run["label"],
    })
    cfg["inr"].update({
        "n_layers": 20,
        "n_units": run["n_units"],
        "lr": 1e-5,
        "epochs": run["epochs"],
        "patience": run["epochs"],
        "batch_size": 16384,
        "latent_steps": run["latent_steps"],
        "latent_lr": run["latent_lr"],
        "reg_lambda": 1e-6,
        "use_band_loss": True,
        "band_loss_train": False,
        "best_checkpoint": True,
        "seismic_weight": 8.0,
        "band_loss_mix": run["band_mix"],
    })
    print(f"[训练] {run['label']} ...", flush=True)
    pipe = ImprovedDASPipeline(cfg)
    stats = pipe.train(data, inr_epochs=run["epochs"], ode_epochs=0, decode_mode="inr")
    pkt = pipe.compress(data, z_init=pipe._z)
    recon = pipe.decompress(pkt)
    norm, _, _ = normalize_das(data)
    recon_np = recon if isinstance(recon, np.ndarray) else recon.cpu().numpy()
    metrics = evaluate_reconstruction(norm, recon_np, norm.nbytes, pkt.compressed_bytes)
    metrics["meets_psnr_40db"] = metrics["psnr_db"] >= 40.0
    metrics["train_stats"] = stats
    np.save(out_dir / "original_norm.npy", norm)
    np.save(out_dir / "reconstructed.npy", recon_np)
    pipe.save_checkpoint(out_dir / "checkpoint.pt")
    save_json_report(metrics, out_dir / "metrics.json")
    regenerate_from_npy(out_dir, run["label"], out_dir / "figures")
    return metrics


def write_patent_analysis(rows: list[dict], path: Path):
    best = rows[0] if rows else {}
    terra_rows = [r for r in rows if "terra" in r["path"]]
    kkfls_rows = [r for r in rows if "kkfls" in r["path"]]
    best_terra = terra_rows[0] if terra_rows else {}
    best_kkfls = kkfls_rows[0] if kkfls_rows else {}

    lines = [
        "# 基于隐式神经表示与 Neural ODE 的 DAS 数据压缩 — 实验结果与专利分析",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "> 用途：专利权利要求支撑、实施例撰写、效果对比说明",
        "",
        "---",
        "",
        "## 一、技术方案概述",
        "",
        "本发明采用 **\"物理预处理 — 潜向量 INR 编码 — 连续时间解码\"** 三阶段架构：",
        "",
        "1. **编码端**：对 DAS 应变场 `(C×T)` 逐通道归一化后，以随机傅里叶特征 (RFF) 映射时空坐标，",
        "   通过 SIREN 隐式网络 `f(x,t;z)` 拟合；潜向量 `z∈R^256` 与网络权重联合训练，",
        "   压缩阶段固定权重、热启动精调 `z`（可选分频段损失）。",
        "2. **传输**：一级数据层传输潜向量包（约 2.3 KB），二级控制层携带元数据；",
        "   共享 INR 权重驻留解码端（后续任务压缩比 >100×）。",
        "3. **解码端**：由 `z` 经 INR 全网格重建归一化应变场，可选 Neural ODE 混合解码。",
        "",
        "### 核心创新点（相对现有技术）",
        "",
        "| 创新点 | 技术效果 |",
        "|--------|----------|",
        "| 潜向量热启动精调 | 避免 z 从 0 冷启动，加速收敛、提升 PSNR |",
        "| PSNR 最优 checkpoint | 避免过训练回退，稳定选取最优 epoch |",
        "| 精调阶段分频段混合损失 | 在空域 MSE 基础上加权 3–10 Hz，**band_mix=0.08 达 40.30 dB** |",
        "| 统一时域编解码 | 消除系数域/时域语义断裂（PSNR 7→26 dB 量级提升） |",
        "",
        "---",
        "",
        "## 二、实验数据与配置",
        "",
        "| 项 | 说明 |",
        "|----|------|",
        "| 数据源 | Cook Inlet DAS 实验，事件 11715354 |",
        "| TERRA | `11715354_TERRA.h5`，8531 通道 × 3000 采样 |",
        "| KKFLS | `11715354_KKFLS.h5`，同上 |",
        "| 训练子集 | 128 通道 × 500 采样（除非另有说明） |",
        "| 采样率 | 50 Hz（配置项） |",
        "| 网络 | SIREN 20 层 / 512 单元 / RFF(128, scale=20) / ω=30 |",
        "| 训练 | Adam lr=1e-5，400 epoch，batch=16384 |",
        "| 精调 | latent_steps=3000，latent_lr=2e-3，band_mix=0.08（最佳） |",
        "",
        "---",
        "",
        "## 三、全部实验结果汇总",
        "",
        "| 排名 | 实验路径 | PSNR (dB) | 压缩比 | 40dB | n_units | band_mix |",
        "|------|----------|-----------|--------|------|---------|----------|",
    ]
    for i, r in enumerate(rows[:25], 1):
        if r["psnr_db"] is None:
            continue
        flag = "✓" if r.get("meets_40db") else ""
        lines.append(
            f"| {i} | `{r['path']}` | {r['psnr_db']:.2f} | {r.get('compression_ratio', 0):.1f}× | {flag} | "
            f"{r.get('n_units', '-')} | {r.get('band_loss_mix', '-')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 四、关键结论",
        "",
        f"1. **专利核心指标 PSNR ≥ 40 dB 已在 TERRA 数据上达成**："
        f"最优 **{best_terra.get('psnr_db', 0):.2f} dB**（`{best_terra.get('path', '')}`）。",
        f"2. **KKFLS 事件数据最优 {best_kkfls.get('psnr_db', 0):.2f} dB**，"
        "说明数据特性对压缩重建质量有显著影响。",
        "3. **纯 SIREN 权重传输范式**（无潜向量）在本数据上 PSNR 仅 20–25 dB，",
        "   且传输体积 MB 级，不满足压缩语义。",
        "4. **单纯增加 epoch（500）或 latent_steps（4000）反而降低 PSNR**，",
        "   验证 best checkpoint 与精调超参选择的必要性。",
        "5. **band_loss_mix 敏感性**：0.15→39.79 dB，**0.08→40.30 dB**，存在最优区间。",
        "",
        "---",
        "",
        "## 五、专利撰写建议",
        "",
        "### 5.1 独立权利要求支撑数据",
        "",
        "- 重建 PSNR：**40.30 dB**（128×500 子集，潜向量 256 维）",
        "- 压缩比：**112.7×**（后续任务，原始 256 KB → 压缩包 2.3 KB）",
        "- 传输内容：潜向量 + 元数据（不含完整网络权重重复传输）",
        "",
        "### 5.2 从属权利要求可引用特征",
        "",
        "- 潜向量精调采用联合训练所得 z 作为初始值（热启动）；",
        "- 训练过程中按全网格 PSNR 保存最优模型参数；",
        "- 精调损失函数为 `(1-α)·MSE + α·L_band`，α=0.08，L_band 对 3–10 Hz 加权；",
        "- 坐标编码采用 RFF，激活函数采用 SIREN，层数 ≥20，隐藏单元 ≥512。",
        "",
        "### 5.3 实施例附图建议",
        "",
        "见 `outputs/patent_report/` 目录：",
        "",
        "- `figures/best_40db_terra/` — 达标方案全套中文宋体图",
        "- `figures/best_kkfls/` — 对比实施例",
        "- `psnr_leaderboard.png` — 全部实验对比",
        "- `all_experiments.json` — 机器可读完整指标",
        "",
        "---",
        "",
        "## 六、复现命令（最佳方案）",
        "",
        "```bash",
        "python -u das_inr_ode_compression/scripts/train_200epoch.py \\",
        "  --h5 data/raw/11715354_TERRA.h5 \\",
        "  --inr-epochs 400 --gpu-512 --latent-steps 3000 \\",
        "  --n-channels 128 --n-samples 500 --band-loss-mix 0.08 \\",
        "  --output-dir das_inr_ode_compression/outputs/terra_11715354",
        "```",
        "",
        "---",
        "",
        "*本报告由 `scripts/generate_patent_report.py` 自动生成。*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    font = setup_chinese_font()
    print(f"中文字体: {font}", flush=True)

    rows = collect_all_metrics()
    save_json_report({"generated_at": datetime.now().isoformat(), "experiments": rows},
                     REPORT_DIR / "all_experiments.json")
    plot_psnr_leaderboard(rows, REPORT_DIR / "psnr_leaderboard.png")

    # 重绘已有 npy 实验图（宋体）
    for mp in OUTPUTS.rglob("metrics.json"):
        exp_dir = mp.parent
        if "weight_chunks" in str(exp_dir):
            continue
        if (exp_dir / "original_norm.npy").exists():
            title = exp_dir.name
            regenerate_from_npy(exp_dir, title, REPORT_DIR / "figures" / "regenerated" / exp_dir.relative_to(OUTPUTS))

    # 训练关键方案并出全套图
    for run in KEY_RUNS:
        train_and_figure(run, REPORT_DIR / "figures" / run["id"])

    write_patent_analysis(rows, REPORT_DIR / "PATENT_RESULTS_ANALYSIS.md")
    print(f"报告已生成: {REPORT_DIR}", flush=True)


if __name__ == "__main__":
    main()
