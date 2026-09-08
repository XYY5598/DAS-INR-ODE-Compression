#!/usr/bin/env python
"""验证: 压缩比、PSNR、多场景 {{location_theme}} 对比。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import get_location_theme, get_title_text, load_config
from das_inr_ode.paths import OUTPUTS
from das_inr_ode.pipeline import DASCompressionPipeline, save_report
from das_inr_ode.synthetic_data import generate_synthetic_das
from das_inr_ode.visualization import plot_comparison


def run_validation(cfg, theme: str, title: str, epochs: int = 15) -> dict:
    cfg = load_config(
        location_theme=theme,
        title_text=title,
        overrides={
            "n_channels": cfg.get("n_channels", 128),
            "n_samples": cfg.get("n_samples", 300),
            "inr": {"epochs": epochs},
        },
    )
    data = generate_synthetic_das(
        int(cfg["n_channels"]),
        int(cfg["n_samples"]),
        float(cfg["sample_rate_hz"]),
        scenario=theme,
    )
    pipe = DASCompressionPipeline(cfg)
    pipe.train(data, epochs=epochs, decode_epochs=20)
    pkt = pipe.compress(data)
    recon = pipe.decompress(pkt)
    metrics = pipe.evaluate(data, recon, pkt)
    return metrics, data, recon, pipe


def main():
    p = argparse.ArgumentParser(description="DAS 压缩验证测试")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--location-theme", default="海洋")
    p.add_argument("--title-text", default="DAS 压缩验证 — {{location_theme}}")
    p.add_argument("--all-themes", action="store_true", help="测试全部场景")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--output-dir", default=str(OUTPUTS / "validation"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    themes = ["海洋", "陆地基础设施", "地质活动监测"] if args.all_themes else [args.location_theme]
    all_metrics = {}

    for theme in themes:
        title = args.title_text.replace("{{location_theme}}", theme)
        print(f"\n=== 场景: {theme} | 标题: {title} ===")
        metrics, data, recon, pipe = run_validation(base_cfg, theme, title, epochs=args.epochs)
        all_metrics[theme] = metrics
        print(f"  PSNR: {metrics['psnr_db']:.2f} dB")
        print(f"  压缩比: {metrics['compression_ratio']:.1f}x")
        print(f"  RMSE: {metrics['rmse']:.6f}")

        plot_comparison(
            data, recon, title,
            save_path=out_dir / f"compare_{theme}.png",
        )
        pipe.save_checkpoint(out_dir / f"checkpoint_{theme}.pt")

    report_path = out_dir / "validation_report.json"
    save_report(all_metrics, report_path, args.title_text, get_location_theme(base_cfg))
    print(f"\n测试报告: {report_path}")


if __name__ == "__main__":
    main()
