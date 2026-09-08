#!/usr/bin/env python
"""改进版 Cook Inlet 基准实验：对比编解码域与解码模式。"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from das_inr_ode.config import load_config
from das_inr_ode.paths import DATA_RAW, OUTPUTS, default_h5
from das_inr_ode.pipeline_v2 import ImprovedDASPipeline, save_json_report
from das_inr_ode.visualization import plot_comparison, plot_strain_field


def load_h5(path: Path, ch_start: int, n_ch: int, n_samples: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        raw = f["/Acquisition/Raw[0]/RawData"][:n_samples, ch_start : ch_start + n_ch].T
    return raw.astype(np.float32)


def find_h5(data_root: Path) -> Path:
    files = sorted(data_root.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"未找到 HDF5: {data_root}")
    return files[0]


def run_experiment(name: str, cfg: dict, data: np.ndarray, out_dir: Path) -> dict:
    print(f"\n{'='*60}\n实验: {name}\n{'='*60}")
    pipe = ImprovedDASPipeline(cfg)
    t0 = time.time()
    train_stats = pipe.train(
        data,
        domain=cfg.get("decode_domain", "time"),
        decode_mode=cfg.get("decode_mode", "hybrid"),
    )
    elapsed = time.time() - t0
    pkt = pipe.compress(data)
    recon = pipe.decompress(pkt)
    from das_inr_ode.pipeline_v2 import normalize_das
    norm, _, _ = normalize_das(data)
    metrics = pipe.evaluate(data, recon, pkt)
    metrics["experiment"] = name
    metrics["train_time_sec"] = elapsed
    metrics["train_stats"] = {
        "train_psnr_db": train_stats.get("train_psnr_db"),
        "inr_epochs_run": len(train_stats.get("inr_losses", [])),
        "ode_epochs_run": len(train_stats.get("decode_losses", [])),
        "domain": train_stats.get("domain"),
        "decode_mode": train_stats.get("decode_mode"),
    }

    exp_dir = out_dir / name.replace(" ", "_")
    exp_dir.mkdir(parents=True, exist_ok=True)
    np.save(exp_dir / "reconstructed.npy", recon)
    np.save(exp_dir / "original_norm.npy", norm)
    pipe.save_checkpoint(exp_dir / "checkpoint.pt")
    title = cfg.get("title_text", name)
    plot_comparison(norm, recon, title, exp_dir / "comparison.png")
    plot_strain_field(recon, f"重建 — {title}", exp_dir / "reconstructed.png")
    save_json_report(metrics, exp_dir / "metrics.json")

    print(f"  PSNR: {metrics['psnr_db']:.2f} dB")
    print(f"  压缩比(后续任务): {metrics['ratio_subsequent']:.1f}x")
    print(f"  专利目标: {'达标' if metrics.get('meets_patent_target') else '未达标'}")
    print(f"  耗时: {elapsed:.1f}s")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config" / "improved.yaml"))
    p.add_argument("--h5", default=None)
    p.add_argument("--n-channels", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--inr-epochs", type=int, default=None)
    p.add_argument("--ode-epochs", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="快速模式：减少 epoch")
    p.add_argument("--only", default=None, help="仅运行指定实验，如 A_时域_hybrid")
    p.add_argument("--output-dir", default=str(OUTPUTS / "improved_benchmark"))
    args = p.parse_args()

    h5_path = Path(args.h5) if args.h5 else default_h5("KKFLS")
    data = load_h5(h5_path, 0, args.n_channels, args.n_samples)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_config(args.config)
    if args.fast:
        base["inr"]["epochs"] = 50
        base["inr"]["n_layers"] = 10
        base["inr"]["patience"] = 20
        base["ode"]["epochs"] = 35
    if args.inr_epochs:
        base["inr"]["epochs"] = args.inr_epochs
    if args.ode_epochs:
        base["ode"]["epochs"] = args.ode_epochs

    experiments = []

    # A: 改进版 — 统一时域 + hybrid
    cfg_time = deepcopy(base)
    cfg_time["decode_domain"] = "time"
    cfg_time["decode_mode"] = "hybrid"
    experiments.append(("A_time_hybrid", cfg_time))

    # B: 统一时域 — 纯 INR
    cfg_inr = deepcopy(base)
    cfg_inr["decode_domain"] = "time"
    cfg_inr["decode_mode"] = "inr"
    experiments.append(("B_time_INR", cfg_inr))

    # C: 系数域（旧方案对比）
    cfg_coeff = deepcopy(base)
    cfg_coeff["decode_domain"] = "coeff"
    cfg_coeff["decode_mode"] = "ode"
    cfg_coeff["inr"]["epochs"] = min(base["inr"]["epochs"], 60)
    cfg_coeff["ode"]["epochs"] = min(base["ode"]["epochs"], 40)
    experiments.append(("C_coeff_ODE", cfg_coeff))

    if args.only:
        experiments = [(n, c) for n, c in experiments if n == args.only]

    all_metrics = {}
    for name, cfg in experiments:
        all_metrics[name] = run_experiment(name, cfg, data, out_dir)

    summary = {
        "data_file": str(h5_path),
        "shape": list(data.shape),
        "experiments": all_metrics,
    }
    if all_metrics:
        summary["best"] = max(all_metrics.items(), key=lambda x: x[1]["psnr_db"])
    save_json_report(summary, out_dir / "benchmark_summary.json")
    if all_metrics:
        best_name, best_m = max(all_metrics.items(), key=lambda x: x[1]["psnr_db"])
        print(f"\n最佳方案: {best_name}, PSNR={best_m['psnr_db']:.2f} dB")
    else:
        print("\n未运行任何实验（--only 过滤无匹配）")


if __name__ == "__main__":
    main()
