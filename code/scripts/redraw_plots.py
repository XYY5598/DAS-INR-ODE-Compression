#!/usr/bin/env python
"""从已有 npy 结果重绘可视化图（中文宋体）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.visualization import (
    plot_comparison,
    plot_strain_field,
    plot_waveform_channel,
    setup_chinese_font,
)


def main():
    p = argparse.ArgumentParser(description="重绘 DAS INR 可视化图")
    p.add_argument("--dir", required=True, help="含 original_norm.npy / reconstructed.npy 的目录")
    p.add_argument("--title", default="DAS INR 重建结果")
    args = p.parse_args()

    out_dir = Path(args.dir)
    norm = np.load(out_dir / "original_norm.npy")
    recon = np.load(out_dir / "reconstructed.npy")
    font = setup_chinese_font()
    print(f"使用字体: {font}", flush=True)

    title = args.title
    plot_comparison(norm, recon, title, out_dir / "comparison.png")
    plot_strain_field(recon, f"重建 — {title}", out_dir / "reconstructed.png")
    plot_waveform_channel(norm, recon, norm.shape[0] // 2, title, out_dir / "waveform.png")
    print(f"已保存至 {out_dir}", flush=True)


if __name__ == "__main__":
    main()
