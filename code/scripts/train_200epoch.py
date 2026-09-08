#!/usr/bin/env python
"""40 dB 冲刺训练脚本：支持更长训练与权重分块传输对比。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import load_config
from das_inr_ode.metrics import evaluate_reconstruction
from das_inr_ode.paths import OUTPUTS, default_h5
from das_inr_ode.pipeline_v2 import ImprovedDASPipeline, normalize_das, save_json_report
from das_inr_ode.visualization import plot_comparison, plot_strain_field, plot_waveform_channel


def load_h5(path: Path, n_ch: int = 128, n_samples: int = 500, ch_start: int = 0) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["/Acquisition/Raw[0]/RawData"][:n_samples, ch_start : ch_start + n_ch].T.astype(np.float32)


def find_energy_channel_start(path: Path, n_ch: int, n_samples: int, step: int = 16) -> int:
    with h5py.File(path, "r") as f:
        block = f["/Acquisition/Raw[0]/RawData"][:n_samples, :].astype(np.float32)
    energy = np.sum(block * block, axis=0)
    best_i, best_e = 0, -1.0
    for i in range(0, block.shape[1] - n_ch + 1, step):
        e = float(energy[i : i + n_ch].sum())
        if e > best_e:
            best_e, best_i = e, i
    return best_i


def save_weight_chunks(
    pipe: ImprovedDASPipeline,
    z: torch.Tensor,
    out_dir: Path,
    chunk_kb: int,
) -> int:
    """将 INR 权重 + latent + RFF 基分块写盘，返回总字节数。"""
    payload = {
        "inr": {k: v.detach().cpu() for k, v in pipe.inr.state_dict().items()},
        "latent": z.detach().cpu(),
    }
    if hasattr(pipe.inr.coord_enc, "B"):
        payload["fourier_B"] = pipe.inr.coord_enc.B.detach().cpu()

    torch.save(payload, out_dir / "siren_payload.pt")
    payload_path = out_dir / "siren_payload.pt"
    blob = payload_path.read_bytes()

    chunk_dir = out_dir / "weight_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = max(int(chunk_kb * 1024), 1024)
    num_chunks = (len(blob) + chunk_bytes - 1) // chunk_bytes
    manifest = []

    for idx in range(num_chunks):
        start = idx * chunk_bytes
        end = min((idx + 1) * chunk_bytes, len(blob))
        chunk = blob[start:end]
        path = chunk_dir / f"chunk_{idx:04d}.bin"
        path.write_bytes(chunk)
        manifest.append({"idx": idx, "bytes": len(chunk), "file": path.name})

    (chunk_dir / "manifest.json").write_text(
        json.dumps(
            {
                "total_bytes": len(blob),
                "chunk_bytes": chunk_bytes,
                "num_chunks": num_chunks,
                "chunks": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(blob)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "improved.yaml"))
    p.add_argument("--h5", default=None)
    p.add_argument("--inr-epochs", type=int, default=300)
    p.add_argument("--latent-steps", type=int, default=2000)
    p.add_argument("--n-units", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=20)
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--gpu-512", action="store_true", help="若检测到 CUDA，则将 n_units 提升到 512")
    p.add_argument(
        "--transmit-mode",
        choices=["latent", "siren_weights"],
        default="latent",
        help="latent: 潜向量包；siren_weights: 分块传输 INR 权重(语义不同)",
    )
    p.add_argument("--chunk-kb", type=int, default=256, help="权重分块大小（KB）")
    p.add_argument("--band-loss-mix", type=float, default=0.15, help="潜向量精调时分频段损失权重")
    p.add_argument("--latent-lr", type=float, default=2.0e-3)
    p.add_argument("--channel-start", type=int, default=0, help="通道起始索引")
    p.add_argument("--auto-channel", action="store_true", help="自动选取能量最大通道窗口")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    h5_path = Path(args.h5) if args.h5 else default_h5("KKFLS")
    n_units = args.n_units
    if args.gpu_512 and torch.cuda.is_available():
        n_units = 512
    tag = f"B_time_INR_{args.inr_epochs}ep_u{n_units}_{args.transmit_mode}"
    out_dir = Path(args.output_dir or str(OUTPUTS / "improved_benchmark" / tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_start = args.channel_start
    if args.auto_channel:
        ch_start = find_energy_channel_start(h5_path, args.n_channels, args.n_samples)
    data = load_h5(h5_path, n_ch=args.n_channels, n_samples=args.n_samples, ch_start=ch_start)
    nc, ns = data.shape
    print(f"数据: {h5_path.name}, shape={data.shape}, ch_start={ch_start}", flush=True)
    print(
        f"配置: epochs={args.inr_epochs}, latent_steps={args.latent_steps}, "
        f"band_mix={args.band_loss_mix}, latent_lr={args.latent_lr}, "
        f"n_layers={args.n_layers}, n_units={n_units}, tx={args.transmit_mode}",
        flush=True,
    )

    cfg = load_config(args.config)
    cfg.update({
        "decode_domain": "time",
        "decode_mode": "inr",
        "latent_dim": 256,
        "n_channels": nc,
        "n_samples": ns,
        "title_text": f"Cook Inlet KKFLS — {args.inr_epochs}ep INR u{args.n_units}",
    })
    cfg["inr"].update({
        "n_layers": args.n_layers,
        "n_units": n_units,
        "lr": 1.0e-5,
        "epochs": args.inr_epochs,
        "patience": args.inr_epochs,
        "batch_size": 16384,
        "latent_steps": args.latent_steps,
        "latent_lr": args.latent_lr,
        "reg_lambda": 1.0e-6,
        "use_band_loss": True,
        "band_loss_train": False,
        "best_checkpoint": True,
        "seismic_weight": 8.0,
        "band_loss_mix": args.band_loss_mix,
    })

    pipe = ImprovedDASPipeline(cfg)
    t0 = time.time()
    stats = pipe.train(data, inr_epochs=args.inr_epochs, ode_epochs=0, decode_mode="inr")
    elapsed = time.time() - t0

    pkt = pipe.compress(data, z_init=pipe._z)
    recon = pipe.decompress(pkt)
    tx_bytes = pkt.compressed_bytes
    tx_note = "潜向量传输"
    if args.transmit_mode == "siren_weights":
        z = torch.tensor(pkt.latent, dtype=torch.float32, device=pipe.device)
        with torch.no_grad():
            recon = pipe.reconstruct_internal(z, nc, ns, mode="inr").cpu().numpy().astype(np.float32)
        tx_bytes = save_weight_chunks(pipe, z, out_dir, args.chunk_kb)
        tx_note = "分块传输 INR 权重 + latent（压缩语义不同于潜向量编码）"

    norm, _, _ = normalize_das(data)
    metrics = evaluate_reconstruction(norm, recon, norm.nbytes, tx_bytes)
    metrics["meets_psnr_40db"] = metrics["psnr_db"] >= 40.0
    metrics["train_time_sec"] = elapsed
    metrics["transmit_mode"] = args.transmit_mode
    metrics["transmit_note"] = tx_note
    metrics["train_stats"] = {
        "inr_epochs_run": len(stats["inr_losses"]),
        "train_psnr_db": stats["train_psnr_db"],
        "best_psnr_db": stats.get("best_psnr_db"),
        "best_epoch": stats.get("best_epoch"),
        "inr_final_loss": stats["inr_losses"][-1] if stats["inr_losses"] else None,
        "latent_steps": args.latent_steps,
        "latent_lr": args.latent_lr,
        "band_loss_mix": args.band_loss_mix,
        "channel_start": ch_start,
        "n_units": n_units,
        "use_band_loss": True,
        "best_checkpoint": True,
        "z_warm_start": True,
    }

    np.save(out_dir / "original_norm.npy", norm)
    np.save(out_dir / "reconstructed.npy", recon)
    pipe.save_checkpoint(out_dir / "checkpoint.pt")
    if args.transmit_mode == "latent":
        (out_dir / "packet.bin").write_bytes(pkt.serialize())
    save_json_report(metrics, out_dir / "metrics.json")

    title = cfg["title_text"]
    plot_comparison(norm, recon, title, out_dir / "comparison.png")
    plot_strain_field(recon, f"重建 — {title}", out_dir / "reconstructed.png")
    plot_waveform_channel(norm, recon, nc // 2, title, out_dir / "waveform.png")

    print("\n" + "=" * 50)
    print(f"PSNR:     {metrics['psnr_db']:.2f} dB  (目标 ≥40 dB)")
    print(f"压缩比:   {metrics['compression_ratio']:.1f}x")
    print(f"传输语义: {tx_note}")
    print(f"INR epoch: {len(stats['inr_losses'])}")
    print(f"耗时:     {elapsed/60:.1f} min")
    print(f"40 dB 达标: {'是' if metrics.get('meets_psnr_40db') else '否'}")
    print(f"输出:     {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
