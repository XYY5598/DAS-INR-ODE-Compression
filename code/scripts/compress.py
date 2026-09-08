#!/usr/bin/env python
"""压缩: 原始 DAS 数据 -> 压缩数据包。"""

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
    p = argparse.ArgumentParser(description="DAS 数据压缩")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--location-theme", default=None)
    p.add_argument("--title-text", default=None)
    p.add_argument("--checkpoint", default=str(OUTPUTS / "checkpoint.pt"))
    p.add_argument("--data", default=None)
    p.add_argument("--output", default=str(OUTPUTS / "packet.bin"))
    args = p.parse_args()

    cfg = load_config(args.config, location_theme=args.location_theme, title_text=args.title_text)
    pipe = DASCompressionPipeline(cfg)
    if Path(args.checkpoint).exists():
        pipe.load_checkpoint(args.checkpoint)

    if args.data:
        data = np.load(args.data)
    else:
        data = generate_synthetic_das(
            int(cfg["n_channels"]), int(cfg["n_samples"]),
            float(cfg["sample_rate_hz"]), cfg.get("location_theme", "海洋"),
        )

    pkt = pipe.compress(data)
    Path(args.output).write_bytes(pkt.serialize())
    ratio = data.nbytes / pkt.compressed_bytes
    print(f"压缩包: {args.output} ({pkt.compressed_bytes} bytes)")
    print(f"原始: {data.nbytes} bytes, 压缩比: {ratio:.1f}x")


if __name__ == "__main__":
    main()
