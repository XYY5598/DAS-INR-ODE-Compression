#!/usr/bin/env python
"""快速验证（小尺寸、少 epoch，用于 CI/冒烟测试）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from das_inr_ode.config import load_config
from das_inr_ode.paths import OUTPUTS
from das_inr_ode.pipeline import DASCompressionPipeline
from das_inr_ode.synthetic_data import generate_synthetic_das


def main():
    theme = "海洋"
    title = "快速验证 — {{title_text}}"
    cfg = load_config(
        location_theme=theme,
        title_text=title,
        overrides={
            "n_channels": 64,
            "n_samples": 100,
            "latent_dim": 64,
            "inr": {"epochs": 4, "n_layers": 3, "n_units": 64, "batch_size": 2048},
            "preprocessing": {"dwt_level": 2, "lpc_order": 4},
            "ode": {"hidden_dim": 32, "solver": "euler"},
        },
    )
    data = generate_synthetic_das(64, 100, 50.0, theme, seed=0)
    pipe = DASCompressionPipeline(cfg)
    pipe.train(data, epochs=4, decode_epochs=8)
    pkt = pipe.compress(data)
    recon = pipe.decompress(pkt)
    metrics = pipe.evaluate(data, recon, pkt)

    out = OUTPUTS / "validation"
    out.mkdir(parents=True, exist_ok=True)
    report = {"location_theme": theme, "title_text": title, "metrics": metrics}
    (out / "quick_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    assert metrics["compression_ratio"] > 1.0, "压缩比应 > 1"
    print("OK: quick validation passed")


if __name__ == "__main__":
    main()
