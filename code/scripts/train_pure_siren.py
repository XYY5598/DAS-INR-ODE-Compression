#!/usr/bin/env python
"""纯 SIREN 训练 — DAS-reconstruction 范式（无潜向量，权重即表示）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.metrics import evaluate_reconstruction
from das_inr_ode.paths import OUTPUTS
from das_inr_ode.pipeline_v2 import normalize_das, save_json_report
from das_inr_ode.pure_siren import reconstruct, save_weight_chunks, train_pure_siren
from das_inr_ode.visualization import plot_comparison, plot_strain_field, plot_waveform_channel


def load_h5(path: Path, n_ch: int = 128, n_samples: int = 500) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["/Acquisition/Raw[0]/RawData"][:n_samples, :n_ch].T.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="纯 SIREN 拟合（DAS-reconstruction 范式）")
    p.add_argument("--h5", required=True, help="HDF5 数据路径")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--n-units", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=20)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--gpu-512", action="store_true", help="CUDA 可用时使用 512 units")
    p.add_argument("--chunk-kb", type=int, default=256)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    h5_path = Path(args.h5)
    n_units = 512 if args.gpu_512 and torch.cuda.is_available() else args.n_units
    out_dir = Path(
        args.output_dir
        or OUTPUTS / "pure_siren" / f"{h5_path.stem}_{args.epochs}ep_u{n_units}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_h5(h5_path, n_ch=args.n_channels, n_samples=args.n_samples)
    norm, _, _ = normalize_das(data)
    nc, ns = norm.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"数据: {h5_path.name}, shape={data.shape}, device={device}", flush=True)
    print(
        f"纯 SIREN: epochs={args.epochs}, layers={args.n_layers}, units={n_units}, omega={args.omega}",
        flush=True,
    )

    t0 = time.time()
    model, stats = train_pure_siren(
        norm,
        n_layers=args.n_layers,
        n_units=n_units,
        omega=args.omega,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed = time.time() - t0

    meta = {
        "n_channels": nc,
        "n_samples": ns,
        "n_layers": args.n_layers,
        "n_units": n_units,
        "omega": args.omega,
        "mode": "pure_siren",
        "source": str(h5_path),
    }
    tx_bytes = save_weight_chunks(model, out_dir, args.chunk_kb, metadata=meta)
    recon = reconstruct(model, nc, ns, device, args.batch_size)

    metrics = evaluate_reconstruction(norm, recon, norm.nbytes, tx_bytes)
    metrics["meets_psnr_40db"] = metrics["psnr_db"] >= 40.0
    metrics["train_time_sec"] = elapsed
    metrics["transmit_mode"] = "pure_siren_weights"
    metrics["transmit_note"] = "纯 SIREN 权重分块传输（无潜向量）"
    metrics["train_stats"] = stats

    np.save(out_dir / "original_norm.npy", norm)
    np.save(out_dir / "reconstructed.npy", recon)
    torch.save({"siren": model.state_dict(), "meta": meta}, out_dir / "checkpoint.pt")
    save_json_report(metrics, out_dir / "metrics.json")

    title = f"Pure SIREN — {h5_path.stem} {args.epochs}ep u{n_units}"
    plot_comparison(norm, recon, title, out_dir / "comparison.png")
    plot_strain_field(recon, f"重建 — {title}", out_dir / "reconstructed.png")
    plot_waveform_channel(norm, recon, nc // 2, title, out_dir / "waveform.png")

    print("\n" + "=" * 50)
    print(f"PSNR:     {metrics['psnr_db']:.2f} dB  (目标 ≥40 dB)")
    print(f"传输大小: {tx_bytes / 1024 / 1024:.2f} MB")
    print(f"压缩比:   {metrics['compression_ratio']:.2f}x")
    print(f"SIREN epoch: {stats['epochs_run']}")
    print(f"耗时:     {elapsed / 60:.1f} min")
    print(f"40 dB 达标: {'是' if metrics['meets_psnr_40db'] else '否'}")
    print(f"输出:     {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
