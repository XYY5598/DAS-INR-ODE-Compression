"""压缩数据包: 一级数据层 + 二级控制层接口。"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import zlib


MAGIC = b"DASZ"
VERSION = 1
HEADER_FMT = "!4sHIIQqI16s"  # magic, version, latent_dim, crc, t_start_ms, clock_offset_ms, seq, uuid


@dataclass
class ControlLayer:
    """二级控制层接口。"""
    time_sync_enabled: bool = True
    snapshot_request: bool = False
    reset_signal: bool = False
    pinn_mode: bool = True
    fast_response: bool = False


@dataclass
class CompressionPacket:
    """一级数据层 + 控制层。"""
    latent: np.ndarray
    fourier_B: np.ndarray | None = None
    network_version: int = 1
    sequence_id: uuid.UUID = field(default_factory=uuid.uuid4)
    sample_start_ms: int = 0
    clock_offset_ms: int = 0
    sequence_number: int = 0
    control: ControlLayer = field(default_factory=ControlLayer)
    metadata: dict[str, Any] = field(default_factory=dict)

    def crc32(self) -> int:
        payload = self.latent.astype(np.float32).tobytes()
        if self.fourier_B is not None:
            payload += self.fourier_B.astype(np.float32).tobytes()
        return zlib.crc32(payload) & 0xFFFFFFFF

    def serialize(self) -> bytes:
        latent = self.latent.astype(np.float32)
        d = latent.size
        body = latent.tobytes()
        if self.fourier_B is not None:
            body += self.fourier_B.astype(np.float32).tobytes()
        crc = zlib.crc32(body) & 0xFFFFFFFF
        uid = self.sequence_id.bytes
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            VERSION,
            d,
            crc,
            self.sample_start_ms,
            self.clock_offset_ms,
            self.sequence_number,
            uid,
        )
        ctrl = struct.pack(
            "!BBBB",
            int(self.control.time_sync_enabled),
            int(self.control.snapshot_request),
            int(self.control.reset_signal),
            int(self.control.pinn_mode),
        )
        meta_json = str(self.metadata).encode("utf-8")
        meta_len = struct.pack("!I", len(meta_json))
        return header + ctrl + meta_len + meta_json + body

    @classmethod
    def deserialize(cls, data: bytes) -> "CompressionPacket":
        hsize = struct.calcsize(HEADER_FMT)
        magic, ver, d, crc, t0, off, seq, uid_bytes = struct.unpack(HEADER_FMT, data[:hsize])
        if magic != MAGIC:
            raise ValueError("Invalid packet magic")
        pos = hsize
        ctrl_bytes = data[pos : pos + 4]
        pos += 4
        (meta_len,) = struct.unpack("!I", data[pos : pos + 4])
        pos += 4
        meta_str = data[pos : pos + meta_len].decode("utf-8")
        pos += meta_len
        body = data[pos:]
        latent_size = d * 4
        latent = np.frombuffer(body[:latent_size], dtype=np.float32).copy()
        fourier_B = None
        if len(body) > latent_size:
            fourier_B = np.frombuffer(body[latent_size:], dtype=np.float32).copy()
        computed = zlib.crc32(body) & 0xFFFFFFFF
        if computed != crc:
            raise ValueError(f"CRC mismatch: {computed} vs {crc}")
        c0, c1, c2, c3 = ctrl_bytes
        control = ControlLayer(
            time_sync_enabled=bool(c0),
            snapshot_request=bool(c1),
            reset_signal=bool(c2),
            pinn_mode=bool(c3),
        )
        return cls(
            latent=latent,
            fourier_B=fourier_B,
            network_version=ver,
            sequence_id=uuid.UUID(bytes=uid_bytes),
            sample_start_ms=t0,
            clock_offset_ms=off,
            sequence_number=seq,
            control=control,
            metadata={"raw": meta_str},
        )

    @property
    def compressed_bytes(self) -> int:
        return len(self.serialize())


def estimate_compression_ratio(original_bytes: int, packet: CompressionPacket) -> float:
    return original_bytes / max(packet.compressed_bytes, 1)


def latent_from_tensor(z: torch.Tensor) -> np.ndarray:
    return z.detach().cpu().numpy().astype(np.float32)
