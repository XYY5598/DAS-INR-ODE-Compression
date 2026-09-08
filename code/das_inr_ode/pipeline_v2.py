"""改进版端到端流水线 v2：统一时域编解码 + 时空 ODE 解码头 + 分频段损失。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from das_inr_ode.config import get_location_theme, get_title_text, load_config
from das_inr_ode.deployment_metrics import deployment_compression_ratio
from das_inr_ode.inr_encoder import INRTrainer, LatentINREncoder, build_coordinate_grid
from das_inr_ode.losses import band_weighted_mse, combined_loss
from das_inr_ode.metrics import evaluate_reconstruction, psnr
from das_inr_ode.monitoring import LongTermMonitor
from das_inr_ode.neural_ode_decoder import NeuralODEDecoder
from das_inr_ode.packet import CompressionPacket, latent_from_tensor
from das_inr_ode.preprocessing import inverse_preprocess, preprocess_spatiotemporal


def normalize_das(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与 DAS-reconstruction-main 一致的逐通道归一化。"""
    means = data.mean(axis=-1, keepdims=True)
    centered = data - means
    peak = np.max(np.abs(centered), axis=-1, keepdims=True)
    peak[peak == 0] = 1.0
    return (centered / peak).astype(np.float32), means.squeeze(-1), peak.squeeze(-1)


class LearnableLatentTrainer:
    """INR 训练：联合优化网络权重与潜向量 z。"""

    def __init__(self, model: LatentINREncoder, latent_dim: int, lr: float = 1e-4, reg_lambda: float = 1e-5):
        self.model = model
        model_device = next(model.parameters()).device
        self.z = torch.nn.Parameter(torch.randn(1, latent_dim, device=model_device) * 0.01)
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + [self.z], lr=lr
        )
        self.reg_lambda = reg_lambda

    def set_z(self, z: torch.Tensor) -> None:
        self.z.data = z.detach().reshape(1, -1).to(self.z.device)

    def train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for coords, targets in loader:
            pred = self.model(coords, self.z.squeeze(0))
            mse = torch.nn.functional.mse_loss(pred, targets)
            reg = self.reg_lambda * self.z.pow(2).mean()
            loss = mse + reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def band_loss_step(
        self,
        norm: np.ndarray,
        nc: int,
        ns: int,
        sample_rate_hz: float,
        seismic_weight: float,
        mix_weight: float = 0.25,
    ) -> float:
        """全网格 MSE + 分频段损失混合微调（避免频域项主导）。"""
        self.model.train()
        coords = build_coordinate_grid(nc, ns, self.z.device)
        pred = self.model(coords, self.z.squeeze(0)).reshape(nc, ns)
        target = torch.tensor(norm, dtype=torch.float32, device=self.z.device)
        spatial = torch.nn.functional.mse_loss(pred, target)
        bw, _ = band_weighted_mse(
            pred, target, sample_rate_hz=sample_rate_hz, seismic_weight=seismic_weight
        )
        reg = self.reg_lambda * self.z.pow(2).mean()
        loss = (1.0 - mix_weight) * spatial + mix_weight * bw + reg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def get_z(self) -> torch.Tensor:
        return self.z.detach().squeeze(0)


class ImprovedDASPipeline:
    """改进版三阶段流水线。"""

    def __init__(self, cfg: dict[str, Any] | None = None, **kwargs):
        self.cfg = cfg or load_config(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._norm_means: np.ndarray | None = None
        self._norm_peak: np.ndarray | None = None
        self._build_models()

    def _build_models(self):
        inr_cfg = self.cfg.get("inr", {})
        ode_cfg = self.cfg.get("ode", {})
        self.inr = LatentINREncoder(
            latent_dim=int(self.cfg["latent_dim"]),
            n_layers=int(inr_cfg.get("n_layers", 20)),
            n_units=int(inr_cfg.get("n_units", 128)),
            activation=inr_cfg.get("activation", "siren"),
            coord_encoding=inr_cfg.get("coord_encoding", "rff"),
            rff_features=int(inr_cfg.get("rff_features", 128)),
            rff_scale=float(inr_cfg.get("rff_scale", 10.0)),
            siren_omega=float(inr_cfg.get("siren_omega", 30.0)),
        ).to(self.device)
        self.decoder = NeuralODEDecoder(
            latent_dim=int(self.cfg["latent_dim"]),
            hidden_dim=int(ode_cfg.get("hidden_dim", 128)),
            n_channels=int(self.cfg.get("n_channels", 256)),
            map_layers=int(ode_cfg.get("map_layers", 2)),
            dynamics_layers=int(ode_cfg.get("dynamics_layers", 3)),
            lipschitz_max=float(ode_cfg.get("lipschitz_max", 10.0)),
            solver=ode_cfg.get("solver", "euler"),
            use_spatiotemporal_head=ode_cfg.get("use_spatiotemporal_head", True),
            head_units=int(ode_cfg.get("head_units", 128)),
        ).to(self.device)
        self.monitor = LongTermMonitor(self.cfg)

    def _prepare_data(
        self, data: np.ndarray, domain: Literal["time", "coeff"]
    ) -> tuple[np.ndarray, np.ndarray, Any | None]:
        norm, means, peak = normalize_das(data.astype(np.float32))
        self._norm_means = means
        self._norm_peak = peak
        nc, ns = norm.shape
        self.cfg["n_channels"] = nc
        self.cfg["n_samples"] = ns
        self.decoder.n_channels = nc

        if domain == "coeff":
            preprocessed, meta = preprocess_spatiotemporal(data, self.cfg)
            self.cfg["n_coeffs"] = preprocessed.shape[1]
            coords = build_coordinate_grid(nc, preprocessed.shape[1], self.device)
            values = torch.tensor(preprocessed, dtype=torch.float32, device=self.device).reshape(-1, 1)
            return norm, values, meta
        coords = build_coordinate_grid(nc, ns, self.device)
        values = torch.tensor(norm, dtype=torch.float32, device=self.device).reshape(-1, 1)
        return norm, values, None

    def _get_coords(self, nc: int, ns: int, domain: str) -> torch.Tensor:
        if domain == "coeff":
            return build_coordinate_grid(nc, self.cfg["n_coeffs"], self.device)
        return build_coordinate_grid(nc, ns, self.device)

    def _inr_reconstruct(self, z: torch.Tensor, nc: int, ns: int) -> torch.Tensor:
        coords = build_coordinate_grid(nc, ns, self.device)
        with torch.no_grad():
            pred = self.inr(coords, z)
        return pred.reshape(nc, ns)

    def train(
        self,
        data: np.ndarray,
        inr_epochs: int | None = None,
        ode_epochs: int | None = None,
        domain: Literal["time", "coeff"] | None = None,
        decode_mode: Literal["ode", "inr", "hybrid"] | None = None,
    ) -> dict[str, Any]:
        inr_cfg = self.cfg.get("inr", {})
        ode_cfg = self.cfg.get("ode", {})
        domain = domain or self.cfg.get("decode_domain", "time")
        decode_mode = decode_mode or self.cfg.get("decode_mode", "hybrid")
        inr_epochs = inr_epochs or int(inr_cfg.get("epochs", 200))
        ode_epochs = ode_epochs or int(ode_cfg.get("epochs", 100))
        fs = float(self.cfg.get("sample_rate_hz", 50.0))

        norm, values, meta = self._prepare_data(data, domain)
        nc, ns = norm.shape
        self._meta = meta

        # subsample for INR mini-batch
        coords_full = self._get_coords(nc, ns, domain)

        n_pts = coords_full.shape[0]
        batch_size = int(inr_cfg.get("batch_size", 8192))
        if n_pts > batch_size:
            idx = torch.randperm(n_pts, device=self.device)[: batch_size * 20]
            coords_train = coords_full[idx]
            values_train = values[idx]
        else:
            coords_train, values_train = coords_full, values

        ds = TensorDataset(coords_train, values_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        trainer = LearnableLatentTrainer(
            self.inr,
            int(self.cfg["latent_dim"]),
            lr=float(inr_cfg.get("lr", 1e-4)),
            reg_lambda=float(inr_cfg.get("reg_lambda", 1e-5)),
        )
        use_band_loss = bool(inr_cfg.get("use_band_loss", False))
        band_loss_train = bool(inr_cfg.get("band_loss_train", False))
        use_best_ckpt = bool(inr_cfg.get("best_checkpoint", False))
        seismic_w = float(inr_cfg.get("seismic_weight", ode_cfg.get("seismic_weight", 8.0)))
        band_mix = float(inr_cfg.get("band_loss_mix", 0.25))
        band_every = int(inr_cfg.get("band_loss_every", 5))
        psnr_eval_every = int(inr_cfg.get("psnr_eval_every", max(1, inr_epochs // 20)))

        inr_losses = []
        psnr_history: list[float] = []
        best_loss = float("inf")
        best_psnr = float("-inf")
        best_inr_state: dict | None = None
        best_z: torch.Tensor | None = None
        best_epoch = 0
        patience = int(inr_cfg.get("patience", 30))
        stale = 0

        for ep in range(inr_epochs):
            loss = trainer.train_epoch(loader)
            if band_loss_train and use_band_loss and (ep + 1) % band_every == 0:
                trainer.band_loss_step(norm, nc, ns, fs, seismic_w, mix_weight=band_mix)
            inr_losses.append(loss)
            if loss < best_loss - 1e-6:
                best_loss = loss
                stale = 0
            else:
                stale += 1

            if use_best_ckpt and ((ep + 1) % psnr_eval_every == 0 or ep == inr_epochs - 1):
                with torch.no_grad():
                    recon_ep = self._inr_reconstruct(trainer.get_z(), nc, ns)
                    ep_psnr = psnr(norm, recon_ep.cpu().numpy())
                psnr_history.append(float(ep_psnr))
                if ep_psnr > best_psnr:
                    best_psnr = float(ep_psnr)
                    best_inr_state = {k: v.detach().cpu().clone() for k, v in self.inr.state_dict().items()}
                    best_z = trainer.get_z().detach().cpu().clone()
                    best_epoch = ep + 1
                print(
                    f"  INR epoch {ep+1}/{inr_epochs}, loss={loss:.6f}, psnr={ep_psnr:.2f} dB",
                    flush=True,
                )
            elif (ep + 1) % max(1, inr_epochs // 10) == 0 or ep == 0:
                print(f"  INR epoch {ep+1}/{inr_epochs}, loss={loss:.6f}", flush=True)

            if loss < 1e-5 or stale >= patience:
                break

        if use_best_ckpt and best_inr_state is not None and best_z is not None:
            self.inr.load_state_dict(best_inr_state)
            trainer.set_z(best_z.to(self.device))
            print(f"  恢复 best checkpoint: epoch={best_epoch}, psnr={best_psnr:.2f} dB", flush=True)

        z_warm = trainer.get_z()
        z = self.inr.encode_latent(
            coords_full,
            values,
            steps=int(inr_cfg.get("latent_steps", 500)),
            lr=float(inr_cfg.get("latent_lr", 5e-3)),
            z_init=z_warm,
            n_channels=nc,
            n_samples=ns,
            sample_rate_hz=fs,
            use_band_loss=use_band_loss,
            seismic_weight=seismic_w,
            band_mix=float(inr_cfg.get("band_loss_mix", 0.15)),
        )
        self._z = z

        if decode_mode == "inr":
            self.cfg["decode_mode"] = decode_mode
            self.cfg["decode_domain"] = domain
            with torch.no_grad():
                final = self.reconstruct_internal(z, nc, ns, "inr")
                train_psnr = psnr(norm, final.cpu().numpy())
            return {
                "inr_losses": inr_losses,
                "decode_losses": [],
                "latent": latent_from_tensor(z),
                "train_psnr_db": float(train_psnr),
                "best_psnr_db": best_psnr if use_best_ckpt else float(train_psnr),
                "best_epoch": best_epoch if use_best_ckpt else len(inr_losses),
                "psnr_history": psnr_history,
                "domain": domain,
                "decode_mode": decode_mode,
            }

        # 训练 ODE 解码器
        target = torch.tensor(norm, dtype=torch.float32, device=self.device)
        dec_params = list(self.decoder.parameters())
        dec_opt = torch.optim.Adam(dec_params, lr=float(ode_cfg.get("lr", 1e-3)))
        decode_losses = []
        pinn_w = float(ode_cfg.get("pinn_weight", 0.1)) if ode_cfg.get("pinn_mode", True) else 0.0
        seismic_w = float(ode_cfg.get("seismic_weight", 5.0))

        for ep in range(ode_epochs):
            recon_ode = self.decoder.reconstruct(z, ns, n_channels=nc)
            if decode_mode == "inr":
                recon = self._inr_reconstruct(z, nc, ns)
            elif decode_mode == "hybrid":
                recon_inr = self._inr_reconstruct(z, nc, ns)
                recon = 0.5 * recon_inr + 0.5 * recon_ode
            else:
                recon = recon_ode

            loss, parts = combined_loss(
                recon, target,
                sample_rate_hz=fs,
                pinn_weight=pinn_w,
                pde_type=ode_cfg.get("pde_type", "wave"),
                wave_speed=float(ode_cfg.get("wave_speed", 1500.0)),
                seismic_weight=seismic_w,
            )
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()
            decode_losses.append(parts)

            if parts.get("spatial", 1.0) < 1e-5:
                break

        self.cfg["decode_mode"] = decode_mode
        self.cfg["decode_domain"] = domain

        with torch.no_grad():
            final = self.reconstruct_internal(z, nc, ns, decode_mode)
            train_psnr = psnr(norm, final.cpu().numpy())

        return {
            "inr_losses": inr_losses,
            "decode_losses": decode_losses,
            "latent": latent_from_tensor(z),
            "train_psnr_db": float(train_psnr),
            "domain": domain,
            "decode_mode": decode_mode,
        }

    def reconstruct_internal(
        self, z: torch.Tensor, nc: int, ns: int, mode: str | None = None
    ) -> torch.Tensor:
        mode = mode or self.cfg.get("decode_mode", "hybrid")
        if mode == "inr":
            return self._inr_reconstruct(z, nc, ns)
        if mode == "ode":
            return self.decoder.reconstruct(z, ns, n_channels=nc)
        recon_inr = self._inr_reconstruct(z, nc, ns)
        recon_ode = self.decoder.reconstruct(z, ns, n_channels=nc)
        alpha = float(self.cfg.get("hybrid_alpha", 0.5))
        return alpha * recon_inr + (1 - alpha) * recon_ode

    def compress(self, data: np.ndarray, z_init: torch.Tensor | None = None, **kwargs) -> CompressionPacket:
        domain = self.cfg.get("decode_domain", "time")
        norm, values, meta = self._prepare_data(data, domain)
        nc, ns = norm.shape
        coords = self._get_coords(nc, ns, domain)
        inr_cfg = self.cfg.get("inr", {})
        ode_cfg = self.cfg.get("ode", {})
        fs = float(self.cfg.get("sample_rate_hz", 50.0))
        z_init = z_init if z_init is not None else getattr(self, "_z", None)
        z = self.inr.encode_latent(
            coords,
            values,
            steps=int(inr_cfg.get("latent_steps", 500)),
            lr=float(inr_cfg.get("latent_lr", 5e-3)),
            z_init=z_init,
            n_channels=nc,
            n_samples=ns,
            sample_rate_hz=fs,
            use_band_loss=bool(inr_cfg.get("use_band_loss", False)),
            seismic_weight=float(inr_cfg.get("seismic_weight", ode_cfg.get("seismic_weight", 8.0))),
            band_mix=float(inr_cfg.get("band_loss_mix", 0.15)),
        )
        self._z = z
        self._meta = meta

        fourier_B = None
        if hasattr(self.inr.coord_enc, "B"):
            fourier_B = self.inr.coord_enc.B.detach().cpu().numpy().copy()

        pkt = CompressionPacket(
            latent=latent_from_tensor(z),
            fourier_B=fourier_B,
            sample_start_ms=int(time.time() * 1000),
            metadata={
                "location_theme": get_location_theme(self.cfg),
                "title_text": get_title_text(self.cfg),
                "n_channels": nc,
                "n_samples": ns,
                "decode_domain": domain,
                "decode_mode": self.cfg.get("decode_mode", "hybrid"),
            },
        )
        return pkt

    def decompress(self, packet: CompressionPacket) -> np.ndarray:
        meta_info = packet.metadata
        nc = int(meta_info.get("n_channels", self.cfg["n_channels"]))
        ns = int(meta_info.get("n_samples", self.cfg["n_samples"]))
        z = torch.tensor(packet.latent, dtype=torch.float32, device=self.device)
        mode = meta_info.get("decode_mode", self.cfg.get("decode_mode", "hybrid"))
        recon = self.reconstruct_internal(z, nc, ns, mode)
        return recon.detach().cpu().numpy().astype(np.float32)

    def evaluate(self, original: np.ndarray, reconstructed: np.ndarray, packet: CompressionPacket) -> dict:
        norm, _, _ = normalize_das(original)
        metrics = evaluate_reconstruction(norm, reconstructed, norm.nbytes, packet.compressed_bytes)
        deploy = deployment_compression_ratio(
            norm.nbytes, packet, self.inr, self.decoder
        )
        metrics.update(deploy)
        metrics["meets_psnr_40db"] = metrics["psnr_db"] >= 40.0
        metrics["meets_patent_target"] = metrics["psnr_db"] >= 40.0 and deploy["meets_50x_subsequent"]
        return metrics

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inr": self.inr.state_dict(),
            "decoder": self.decoder.state_dict(),
            "cfg": self.cfg,
        }
        if getattr(self, "_z", None) is not None:
            payload["latent"] = self._z.detach().cpu()
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.inr.load_state_dict(ckpt["inr"])
        self.decoder.load_state_dict(ckpt["decoder"])
        if "cfg" in ckpt:
            self.cfg.update(ckpt["cfg"])
        if "latent" in ckpt:
            self._z = ckpt["latent"].to(self.device)


def save_json_report(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o) if isinstance(o, np.integer) else bool(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
