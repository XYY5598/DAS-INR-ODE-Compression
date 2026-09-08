"""DAS-reconstruction 式纯 SIREN 拟合：无潜向量，网络权重即数据表示。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from das_inr_ode.metrics import psnr


class PureSIREN(nn.Module):
    """与 DAS-reconstruction-main/scripts/models.py 一致的纯 SIREN。"""

    def __init__(self, n_layers: int = 20, n_units: int = 128, omega: float = 30.0):
        super().__init__()
        self.omega = omega
        self.n_layers = n_layers
        self.n_units = n_units

        self.inputs = nn.Linear(2, n_units, bias=False)
        self.inputs_bias = nn.Parameter(torch.rand(n_units))

        for i in range(n_layers):
            setattr(self, f"ln{i + 1}", nn.Linear(n_units, n_units, bias=False))
            setattr(self, f"ln{i + 1}_bias", nn.Parameter(torch.rand(n_units)))

        self.outputs_bias = nn.Parameter(torch.rand(1))
        self.outputs = nn.Linear(n_units, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self.omega * self.inputs(coords) + self.inputs_bias)
        for i in range(self.n_layers):
            layer = getattr(self, f"ln{i + 1}")
            bias = getattr(self, f"ln{i + 1}_bias")
            x = torch.sin(self.omega * layer(x) + bias)
        return 2 * self.sigmoid(self.outputs(x) + self.outputs_bias) - 1


def init_siren_weights(model: PureSIREN, n_input: int = 2) -> None:
    """SIREN 官方初始化，对齐 DAS-reconstruction-main/scripts/siren/run.py。"""
    n_units = model.n_units
    omega = model.omega
    for name, mod in model.named_parameters():
        if "inputs" in name:
            if "bias" in name:
                mod.data.uniform_(-1 / np.sqrt(n_input), 1 / np.sqrt(n_input))
            elif "weight" in name:
                mod.data.uniform_(-0.5, 0.5)
        else:
            if "bias" in name:
                mod.data.uniform_(-1 / np.sqrt(n_units), 1 / np.sqrt(n_units))
            elif "weight" in name:
                bound = np.sqrt(6 / n_units) / omega
                mod.data.uniform_(-bound, bound)


def build_coords(nc: int, ns: int, device: torch.device) -> torch.Tensor:
    """构建 [t, x] 坐标，顺序与 DAS-reconstruction 一致。"""
    in_x, in_t = torch.meshgrid(torch.arange(nc), torch.arange(ns), indexing="ij")
    t = (in_t / max(ns - 1, 1)).reshape(-1, 1)
    x = (in_x / max(nc - 1, 1)).reshape(-1, 1)
    return torch.cat([t, x], dim=-1).to(device)


def make_loader(
    norm: np.ndarray,
    device: torch.device,
    batch_size: int = 16384,
) -> DataLoader:
    nc, ns = norm.shape
    coords = build_coords(nc, ns, device)
    values = torch.tensor(norm, dtype=torch.float32, device=device).reshape(-1, 1)
    return DataLoader(TensorDataset(coords, values), batch_size=batch_size, shuffle=True)


@torch.no_grad()
def reconstruct(model: PureSIREN, nc: int, ns: int, device: torch.device, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    coords = build_coords(nc, ns, device)
    chunks: list[torch.Tensor] = []
    for start in range(0, coords.shape[0], batch_size):
        chunks.append(model(coords[start : start + batch_size]))
    return torch.cat(chunks).reshape(nc, ns).cpu().numpy().astype(np.float32)


def train_pure_siren(
    norm: np.ndarray,
    n_layers: int = 20,
    n_units: int = 128,
    omega: float = 30.0,
    epochs: int = 400,
    lr: float = 1e-5,
    batch_size: int = 16384,
    loss_stop: float = 1e-5,
    device: torch.device | None = None,
) -> tuple[PureSIREN, dict[str, Any]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc, ns = norm.shape
    model = PureSIREN(n_layers=n_layers, n_units=n_units, omega=omega).to(device)
    init_siren_weights(model)
    loader = make_loader(norm, device, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        for coords, targets in loader:
            pred = model(coords)
            loss = nn.functional.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        losses.append(avg)
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  SIREN epoch {ep + 1}/{epochs}, loss={avg:.6f}", flush=True)
        if avg < loss_stop:
            break

    recon = reconstruct(model, nc, ns, device, batch_size)
    return model, {
        "losses": losses,
        "epochs_run": len(losses),
        "train_psnr_db": float(psnr(norm, recon)),
        "final_loss": losses[-1] if losses else None,
        "n_layers": n_layers,
        "n_units": n_units,
        "omega": omega,
    }


def save_weight_chunks(
    model: PureSIREN,
    out_dir: Path,
    chunk_kb: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    """分块保存纯 SIREN 权重（无潜向量）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "siren": model.state_dict(),
        "meta": metadata or {},
    }
    payload_path = out_dir / "siren_weights.pt"
    torch.save(payload, payload_path)
    blob = payload_path.read_bytes()

    chunk_dir = out_dir / "weight_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = max(int(chunk_kb * 1024), 1024)
    num_chunks = (len(blob) + chunk_bytes - 1) // chunk_bytes
    manifest = []
    for idx in range(num_chunks):
        start = idx * chunk_bytes
        end = min((idx + 1) * chunk_bytes, len(blob))
        chunk_path = chunk_dir / f"chunk_{idx:04d}.bin"
        chunk_path.write_bytes(blob[start:end])
        manifest.append({"idx": idx, "bytes": end - start, "file": chunk_path.name})

    (chunk_dir / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "pure_siren",
                "total_bytes": len(blob),
                "chunk_bytes": chunk_bytes,
                "num_chunks": num_chunks,
                "chunks": manifest,
                "meta": metadata or {},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(blob)


def load_from_chunks(chunk_dir: Path, device: torch.device | None = None) -> PureSIREN:
    """从分块 manifest 还原纯 SIREN 模型。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
    blob = b"".join((chunk_dir / c["file"]).read_bytes() for c in manifest["chunks"])
    payload_path = chunk_dir.parent / "_reassembled.pt"
    payload_path.write_bytes(blob)
    payload = torch.load(payload_path, map_location=device, weights_only=False)
    meta = payload.get("meta", manifest.get("meta", {}))
    model = PureSIREN(
        n_layers=int(meta.get("n_layers", 20)),
        n_units=int(meta.get("n_units", 128)),
        omega=float(meta.get("omega", 30.0)),
    ).to(device)
    model.load_state_dict(payload["siren"])
    payload_path.unlink(missing_ok=True)
    return model
