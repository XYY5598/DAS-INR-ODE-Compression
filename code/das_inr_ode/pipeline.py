"""端到端压缩/解压流水线。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from das_inr_ode.config import get_location_theme, get_title_text, load_config
from das_inr_ode.inr_encoder import (
    INRTrainer,
    LatentINREncoder,
    build_coordinate_grid,
    prepare_training_batch,
)
from das_inr_ode.metrics import evaluate_reconstruction
from das_inr_ode.monitoring import LongTermMonitor
from das_inr_ode.neural_ode_decoder import NeuralODEDecoder, pinn_loss
from das_inr_ode.packet import CompressionPacket, ControlLayer, estimate_compression_ratio, latent_from_tensor
from das_inr_ode.preprocessing import inverse_preprocess, preprocess_spatiotemporal
from das_inr_ode.visualization import plot_comparison


class DASCompressionPipeline:
    """物理预处理 — INR 编码 — Neural ODE 解码 三阶段流水线。"""

    def __init__(self, cfg: dict[str, Any] | None = None, **kwargs):
        self.cfg = cfg or load_config(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_models()

    def _build_models(self):
        inr_cfg = self.cfg.get("inr", {})
        ode_cfg = self.cfg.get("ode", {})
        self.inr = LatentINREncoder(
            latent_dim=int(self.cfg["latent_dim"]),
            n_layers=int(inr_cfg.get("n_layers", 5)),
            n_units=int(inr_cfg.get("n_units", 256)),
            activation=inr_cfg.get("activation", "siren"),
            coord_encoding=inr_cfg.get("coord_encoding", "rff"),
            rff_features=int(inr_cfg.get("rff_features", 128)),
            rff_scale=float(inr_cfg.get("rff_scale", 10.0)),
            siren_omega=float(inr_cfg.get("siren_omega", 30.0)),
        ).to(self.device)
        self.decoder = NeuralODEDecoder(
            latent_dim=int(self.cfg["latent_dim"]),
            hidden_dim=int(ode_cfg.get("hidden_dim", 64)),
            n_channels=int(self.cfg["n_channels"]),
            map_layers=int(ode_cfg.get("map_layers", 2)),
            dynamics_layers=int(ode_cfg.get("dynamics_layers", 3)),
            lipschitz_max=float(ode_cfg.get("lipschitz_max", 10.0)),
            solver=ode_cfg.get("solver", "dopri5"),
            rtol=float(ode_cfg.get("rtol", 1e-5)),
            atol=float(ode_cfg.get("atol", 1e-7)),
        ).to(self.device)
        self.monitor = LongTermMonitor(self.cfg)

    def train(
        self,
        data: np.ndarray,
        epochs: int | None = None,
        decode_epochs: int = 30,
    ) -> dict[str, Any]:
        """联合训练 INR 编码器与 ODE 解码器。"""
        inr_cfg = self.cfg.get("inr", {})
        ode_cfg = self.cfg.get("ode", {})
        epochs = epochs or int(inr_cfg.get("epochs", 50))

        preprocessed, meta = preprocess_spatiotemporal(data, self.cfg)
        nc, ns = data.shape
        n_coeff = preprocessed.shape[1]
        self.cfg["n_channels"] = nc
        self.cfg["n_samples"] = ns
        self.cfg["n_coeffs"] = n_coeff
        self.decoder.n_channels = nc
        self.decoder.signal_head = self.decoder.signal_head.__class__(
            int(ode_cfg.get("hidden_dim", 64)), nc
        ).to(self.device)

        coords = build_coordinate_grid(nc, n_coeff, self.device)
        values = torch.tensor(preprocessed, dtype=torch.float32, device=self.device).reshape(-1, 1)
        z_init = torch.randn(1, int(self.cfg["latent_dim"]), device=self.device) * 0.01
        z_batch = z_init.expand(coords.shape[0], -1)

        ds = TensorDataset(coords, values, z_batch)
        loader = DataLoader(ds, batch_size=int(inr_cfg.get("batch_size", 4096)), shuffle=True)

        trainer = INRTrainer(
            self.inr,
            lr=float(inr_cfg.get("lr", 1e-4)),
            reg_lambda=float(inr_cfg.get("reg_lambda", 1e-5)),
        )
        inr_losses = []
        for ep in range(epochs):
            loss = trainer.train_epoch(loader)
            inr_losses.append(loss)
            if loss < 1e-4:
                break

        # 优化潜向量
        z = self.inr.encode_latent(coords, values, steps=300, lr=1e-2)

        # 训练 ODE 解码器
        dec_opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        target = torch.tensor(data, dtype=torch.float32, device=self.device)
        decode_losses = []
        pinn_w = float(ode_cfg.get("pinn_weight", 0.5)) if ode_cfg.get("pinn_mode", True) else 0.0
        pde_type = ode_cfg.get("pde_type", "wave")

        for ep in range(decode_epochs):
            recon = self.decoder.reconstruct(z, ns)
            loss, parts = pinn_loss(
                recon,
                target,
                pde_type=pde_type,
                weight=pinn_w,
                wave_speed=float(ode_cfg.get("wave_speed", 1500.0)),
            )
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()
            decode_losses.append(float(loss.item()))

        self._meta = meta
        self._z = z
        return {
            "inr_losses": inr_losses,
            "decode_losses": decode_losses,
            "latent": latent_from_tensor(z),
        }

    def compress(
        self,
        data: np.ndarray,
        sequence_number: int = 0,
        sample_start_ms: int | None = None,
    ) -> CompressionPacket:
        """编码端: 预处理 + INR 潜向量 + 数据包封装。"""
        preprocessed, meta = preprocess_spatiotemporal(data, self.cfg)
        self._meta = meta
        nc, ns = data.shape
        n_coeff = preprocessed.shape[1]

        coords = build_coordinate_grid(nc, n_coeff, self.device)
        values = torch.tensor(preprocessed, dtype=torch.float32, device=self.device).reshape(-1, 1)

        z = self.inr.encode_latent(coords, values, steps=150, lr=5e-3)
        self._z = z

        fourier_B = None
        if hasattr(self.inr.coord_enc, "B"):
            fourier_B = self.inr.coord_enc.B.detach().cpu().numpy().copy()

        if sample_start_ms is None:
            sample_start_ms = int(time.time() * 1000)

        pkt = CompressionPacket(
            latent=latent_from_tensor(z),
            fourier_B=fourier_B,
            sample_start_ms=sample_start_ms,
            sequence_number=sequence_number,
            metadata={
                "location_theme": get_location_theme(self.cfg),
                "title_text": get_title_text(self.cfg),
                "n_channels": nc,
                "n_samples": ns,
            },
        )
        return pkt

    def decompress(
        self,
        packet: CompressionPacket,
        reference_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """解码端: 解析数据包 + ODE 重建。"""
        meta_info = packet.metadata
        nc = int(meta_info.get("n_channels", self.cfg.get("n_channels", 256)))
        ns = int(meta_info.get("n_samples", self.cfg.get("n_samples", 500)))

        if reference_shape:
            nc, ns = reference_shape

        z = torch.tensor(packet.latent, dtype=torch.float32, device=self.device)
        ode_cfg = self.cfg.get("ode", {})
        kwargs = {}
        if packet.control.reset_signal or self.monitor.state.mode == "fast_response":
            kwargs["method"] = "euler"
            kwargs["euler_step"] = float(self.cfg.get("monitoring", {}).get("fast_response_euler_step", 0.001))

        recon = self.decoder.reconstruct(z, ns, **kwargs)

        # 监测控制
        recon_np = recon.detach().cpu().numpy()
        self.monitor.update_baseline(recon_np)
        anomaly, reason = self.monitor.check_anomaly(recon_np)
        if anomaly:
            resp = self.monitor.apply_anomaly_response(reason)
            packet.control.reset_signal = resp.get("reset_signal", False)
            packet.control.pinn_mode = ode_cfg.get("pinn_mode", True)

        traj = self.decoder.integrate(z, torch.linspace(0, 1, ns, device=self.device), **kwargs)
        self.monitor.maybe_snapshot(traj[-1], float(ns), packet.sequence_number)

        return recon_np.astype(np.float32)

    def evaluate(self, original: np.ndarray, reconstructed: np.ndarray, packet: CompressionPacket) -> dict:
        orig_bytes = original.nbytes
        comp_bytes = packet.compressed_bytes
        metrics = evaluate_reconstruction(original, reconstructed, orig_bytes, comp_bytes)
        metrics["location_theme"] = get_location_theme(self.cfg)
        metrics["title_text"] = get_title_text(self.cfg)
        metrics["packet_compression_ratio"] = estimate_compression_ratio(orig_bytes, packet)
        return metrics

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "inr": self.inr.state_dict(),
                "decoder": self.decoder.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.inr.load_state_dict(ckpt["inr"])
        self.decoder.load_state_dict(ckpt["decoder"])
        if "cfg" in ckpt:
            self.cfg.update(ckpt["cfg"])


def save_report(metrics: dict, path: str | Path, title_text: str, location_theme: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "title_text": title_text,
        "location_theme": location_theme,
        "metrics": metrics,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
