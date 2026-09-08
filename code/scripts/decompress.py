#!/usr/bin/env python
"""解压: 压缩数据包 -> 重建应变场。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import get_title_text, load_config
from das_inr_ode.packet import CompressionPacket
from das_inr_ode.paths import OUTPUTS
from das_inr_ode.pipeline import DASCompressionPipeline
from das_inr_ode.visualization import plot_strain_field


def main():
    p = argparse.ArgumentParser(description="DAS 数据解压重建")
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--location-theme", default=None)
    p.add_argument("--title-text", default=None)
    p.add_argument("--checkpoint", default=str(OUTPUTS / "checkpoint.pt"))
    p.add_argument("--packet", required=True)
    p.add_argument("--output", default=str(OUTPUTS / "reconstructed.npy"))
    p.add_argument("--plot", default=str(OUTPUTS / "reconstructed.png"))
    args = p.parse_args()

    cfg = load_config(args.config, location_theme=args.location_theme, title_text=args.title_text)
    pipe = DASCompressionPipeline(cfg)
    if Path(args.checkpoint).exists():
        pipe.load_checkpoint(args.checkpoint)

    raw = Path(args.packet).read_bytes()
    pkt = CompressionPacket.deserialize(raw)
    recon = pipe.decompress(pkt)
    np.save(args.output, recon)
    title = get_title_text(cfg)
    plot_strain_field(recon, title, args.plot)
    print(f"重建数据: {args.output}, 可视化: {args.plot}")


if __name__ == "__main__":
    main()
