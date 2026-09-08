"""评估指标。"""

from __future__ import annotations

import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(mse(original, reconstructed)))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    peak = float(np.max(np.abs(original)))
    if peak == 0:
        peak = 1.0
    return 20 * np.log10(peak / np.sqrt(err))


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / max(compressed_bytes, 1)


def peak_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.max(np.abs(original - reconstructed)))


def evaluate_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    original_bytes: int,
    compressed_bytes: int,
) -> dict[str, float]:
    return {
        "mse": mse(original, reconstructed),
        "rmse": rmse(original, reconstructed),
        "psnr_db": psnr(original, reconstructed),
        "peak_error": peak_error(original, reconstructed),
        "compression_ratio": compression_ratio(original_bytes, compressed_bytes),
        "original_bytes": float(original_bytes),
        "compressed_bytes": float(compressed_bytes),
    }
