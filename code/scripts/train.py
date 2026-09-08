#!/usr/bin/env python
"""训练 INR + Neural ODE 模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import load_config
from das_inr_ode.paths import OUTPUTS
from das_inr_ode.pipeline import DASCompressionPipeline
from das_inr_ode.synthetic_data import generate_synthetic_das


def main():
    p = argparse.ArgumentParser(description="训练 DAS INR+ODE 压缩模型")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--location-theme", default=None, help="替换 {{location_theme}}")
    p.add_argument("--title-text", default=None, help="替换 {{title_text}}")
    p.add_argument("--data", default=None, help=".npy 数据路径 (C,T)")
    p.add_argument("--output", default=str(OUTPUTS / "checkpoint.pt"))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config, location_theme=args.location_theme, title_text=args.title_text)

    if args.data:
        data = np.load(args.data)
    else:
        theme = cfg.get("location_theme", "海洋")
        data = generate_synthetic_das(
            n_channels=int(cfg["n_channels"]),
            n_samples=int(cfg["n_samples"]),
            sample_rate_hz=float(cfg["sample_rate_hz"]),
            scenario=theme,
        )

    pipe = DASCompressionPipeline(cfg)
    stats = pipe.train(data, epochs=args.epochs)
    pipe.save_checkpoint(args.output)
    print(f"训练完成，检查点: {args.output}")
    print(f"INR 最终损失: {stats['inr_losses'][-1]:.6f}")
    print(f"ODE 最终损失: {stats['decode_losses'][-1]:.6f}")


if __name__ == "__main__":
    main()
