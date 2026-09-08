#!/usr/bin/env python
"""使用 data/raw 中的 Cook Inlet HDF5 数据验证并绘图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import load_config
from das_inr_ode.paths import DATA_RAW, OUTPUTS, default_h5
from das_inr_ode.pipeline import DASCompressionPipeline, save_report
from das_inr_ode.visualization import plot_comparison, plot_strain_field, plot_waveform_channel


def find_h5_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.h5"))
    return files


def load_das_h5(
    path: Path,
    channel_start: int = 0,
    n_channels: int = 256,
    n_samples: int | None = None,
) -> np.ndarray:
    """加载 HDF5 DAS 数据（Cook Inlet OptaSense 格式）。"""
    with h5py.File(path, "r") as f:
        raw = f["/Acquisition/Raw[0]/RawData"]
        ch_end = min(channel_start + n_channels, raw.shape[1])
        if n_samples is None:
            data = raw[:, channel_start:ch_end].T.astype(np.float32)
        else:
            data = raw[:n_samples, channel_start:ch_end].T.astype(np.float32)

    # 与参考代码一致的归一化预处理（训练友好）
    data = data - np.mean(data, axis=-1, keepdims=True)
    peak = np.max(np.abs(data), axis=-1, keepdims=True)
    peak[peak == 0] = 1.0
    data = (data / peak).astype(np.float32)
    return data


def main():
    p = argparse.ArgumentParser(description="Cook Inlet DAS 数据验证")
    p.add_argument("--h5", default=None, help="HDF5 路径，默认 data/raw/*.h5")
    p.add_argument("--channel-start", type=int, default=0)
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--location-theme", default="海洋")
    p.add_argument("--title-text", default="Cook Inlet KKFLS — {{location_theme}}")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--decode-epochs", type=int, default=20)
    p.add_argument("--output-dir", default=str(OUTPUTS / "real_data_validation"))
    args = p.parse_args()

    if args.h5:
        h5_path = Path(args.h5)
    else:
        try:
            h5_path = default_h5("KKFLS")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n或指定: --h5 data/raw/11715354_KKFLS.h5"
            ) from e

    print(f"数据文件: {h5_path}")
    data = load_das_h5(h5_path, args.channel_start, args.n_channels, args.n_samples)
    nc, ns = data.shape
    print(f"加载数据: {nc} 通道 × {ns} 采样点")

    title = args.title_text.replace("{{location_theme}}", args.location_theme)
    cfg = load_config(
        location_theme=args.location_theme,
        title_text=title,
        overrides={
            "n_channels": nc,
            "n_samples": ns,
            "latent_dim": min(128, max(64, nc // 2)),
            "inr": {
                "epochs": args.epochs,
                "n_layers": 4,
                "n_units": 128,
                "batch_size": 4096,
                "activation": "siren",
            },
            "ode": {"hidden_dim": 48, "solver": "euler"},
            "preprocessing": {"dwt_level": 3, "lpc_order": 8},
        },
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DASCompressionPipeline(cfg)
    print("训练中...")
    train_stats = pipe.train(data, epochs=args.epochs, decode_epochs=args.decode_epochs)
    print(f"INR loss: {train_stats['inr_losses'][-1]:.6f}, ODE loss: {train_stats['decode_losses'][-1]:.6f}")

    pkt = pipe.compress(data)
    recon = pipe.decompress(pkt)
    metrics = pipe.evaluate(data, recon, pkt)
    metrics["source_file"] = str(h5_path)
    metrics["shape"] = [nc, ns]

    # 保存
    np.save(out_dir / "original.npy", data)
    np.save(out_dir / "reconstructed.npy", recon)
    pipe.save_checkpoint(out_dir / "checkpoint.pt")
    (out_dir / "packet.bin").write_bytes(pkt.serialize())
    save_report(metrics, out_dir / "metrics.json", title, args.location_theme)

    # 绘图
    plot_strain_field(
        data,
        f"原始 — {title}",
        save_path=out_dir / "01_original.png",
    )
    plot_strain_field(
        recon,
        f"重建 — {title}",
        save_path=out_dir / "02_reconstructed.png",
    )
    plot_comparison(
        data,
        recon,
        title,
        save_path=out_dir / "03_comparison.png",
    )

    # 单通道波形对比
    ch = nc // 2
    plot_waveform_channel(
        data,
        recon,
        ch,
        title,
        save_path=out_dir / "04_waveform_channel.png",
    )

    print("\n=== 验证结果 ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "source_file"}, indent=2, ensure_ascii=False))
    print(f"\n图像已保存至: {out_dir}")


if __name__ == "__main__":
    main()
