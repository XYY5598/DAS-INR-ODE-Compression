"""部署级压缩比评估。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from das_inr_ode.packet import CompressionPacket


def count_model_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def deployment_compression_ratio(
    original_bytes: int,
    packet: CompressionPacket,
    inr_model: torch.nn.Module,
    decoder_model: torch.nn.Module,
    shared_across_tasks: bool = True,
) -> dict[str, Any]:
    """
    部署级压缩比：
    - 首任务：潜向量包 + 共享模型权重
    - 后续任务：仅潜向量包（模型已部署）
    """
    packet_bytes = packet.compressed_bytes
    inr_bytes = count_model_bytes(inr_model)
    dec_bytes = count_model_bytes(decoder_model)
    model_bytes = inr_bytes + dec_bytes

    first_task_bytes = packet_bytes + model_bytes
    subsequent_bytes = packet_bytes

    return {
        "original_bytes": original_bytes,
        "packet_bytes": packet_bytes,
        "inr_model_bytes": inr_bytes,
        "decoder_model_bytes": dec_bytes,
        "model_bytes_total": model_bytes,
        "first_task_total_bytes": first_task_bytes,
        "subsequent_task_bytes": subsequent_bytes,
        "ratio_first_task": original_bytes / max(first_task_bytes, 1),
        "ratio_subsequent": original_bytes / max(subsequent_bytes, 1),
        "ratio_packet_only": original_bytes / max(packet_bytes, 1),
        "meets_50x_subsequent": (original_bytes / max(subsequent_bytes, 1)) >= 50.0,
        "shared_weights_amortized": shared_across_tasks,
    }
