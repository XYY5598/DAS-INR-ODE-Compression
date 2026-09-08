"""长时间动态监测控制模块。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class StateSnapshot:
    hidden_state: np.ndarray
    integrated_time: float
    dynamics_mode: str
    timestamp_ms: int
    sequence_number: int


@dataclass
class MonitoringState:
    baseline_energy: float = 1.0
    energy_history: list[float] = field(default_factory=list)
    mode: str = "normal"  # normal | fast_response | recovery
    last_snapshot_ms: int = 0
    snapshots: list[StateSnapshot] = field(default_factory=list)


class LongTermMonitor:
    """时间同步、状态持久化、异常响应、内存优化控制。"""

    def __init__(self, cfg: dict[str, Any], snapshot_dir: str | Path = "./snapshots"):
        mon = cfg.get("monitoring", {})
        self.snapshot_interval_ms = int(mon.get("snapshot_interval_sec", 600) * 1000)
        self.anomaly_factor = float(mon.get("anomaly_energy_factor", 5.0))
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.state = MonitoringState()

    def align_timeline(
        self,
        sample_start_ms: int,
        clock_offset_ms: int,
        n_samples: int,
        sample_rate_hz: float,
    ) -> np.ndarray:
        """跨设备时间轴对齐。"""
        t0 = sample_start_ms + clock_offset_ms
        dt_ms = 1000.0 / sample_rate_hz
        return np.array([t0 + i * dt_ms for i in range(n_samples)])

    def update_baseline(self, signal: np.ndarray, window_days: int = 30):
        energy = float(np.mean(signal ** 2))
        self.state.energy_history.append(energy)
        if len(self.state.energy_history) > 1000:
            self.state.energy_history = self.state.energy_history[-1000:]
        if self.state.baseline_energy <= 0:
            self.state.baseline_energy = energy
        else:
            alpha = min(0.1, 1.0 / max(window_days, 1))
            self.state.baseline_energy = (1 - alpha) * self.state.baseline_energy + alpha * energy

    def check_anomaly(self, signal: np.ndarray) -> tuple[bool, str]:
        """检测短时能量突增，触发异常响应。"""
        energy = float(np.mean(signal ** 2))
        threshold = self.anomaly_factor * self.state.baseline_energy
        if energy >= threshold and self.anomaly_factor >= 5:
            return True, "energy_surge"
        return False, "normal"

    def maybe_snapshot(
        self,
        hidden: torch.Tensor,
        integrated_time: float,
        seq: int,
    ) -> StateSnapshot | None:
        now_ms = int(time.time() * 1000)
        if now_ms - self.state.last_snapshot_ms < self.snapshot_interval_ms:
            return None
        snap = StateSnapshot(
            hidden_state=hidden.detach().cpu().numpy(),
            integrated_time=integrated_time,
            dynamics_mode=self.state.mode,
            timestamp_ms=now_ms,
            sequence_number=seq,
        )
        self.state.snapshots.append(snap)
        self.state.last_snapshot_ms = now_ms
        path = self.snapshot_dir / f"snap_{seq}_{now_ms}.npz"
        np.savez(
            path,
            hidden=snap.hidden_state,
            t=snap.integrated_time,
            mode=snap.dynamics_mode,
            ts=snap.timestamp_ms,
        )
        return snap

    def load_latest_snapshot(self) -> StateSnapshot | None:
        files = sorted(self.snapshot_dir.glob("snap_*.npz"))
        if not files:
            return None
        data = np.load(files[-1])
        return StateSnapshot(
            hidden_state=data["hidden"],
            integrated_time=float(data["t"]),
            dynamics_mode=str(data["mode"]),
            timestamp_ms=int(data["ts"]),
            sequence_number=0,
        )

    def apply_anomaly_response(self, reason: str) -> dict[str, Any]:
        """切换快速响应模式。"""
        self.state.mode = "fast_response"
        return {
            "reset_signal": reason == "energy_surge",
            "solver": "euler",
            "euler_step": 0.001,
            "high_freq_sampling": True,
        }

    def export_log(self, path: Path):
        log = {
            "baseline_energy": self.state.baseline_energy,
            "mode": self.state.mode,
            "n_snapshots": len(self.state.snapshots),
        }
        path.write_text(json.dumps(log, indent=2), encoding="utf-8")
