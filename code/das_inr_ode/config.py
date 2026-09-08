"""配置加载与占位符替换。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLACEHOLDERS = {
    "location_theme": "海洋",
    "title_text": "DAS 应变场动态监测",
}


def resolve_latent_dim(fiber_length_km: float) -> int:
    """潜向量维度经验公式: d_z = clamp(alpha * L, 64, 256)。"""
    if fiber_length_km < 10:
        alpha = 0.64
    elif fiber_length_km < 50:
        alpha = 1.28
    else:
        alpha = 2.56
    return int(max(64, min(256, round(alpha * fiber_length_km))))


def apply_placeholders(obj: Any, replacements: dict[str, str]) -> Any:
    """递归替换 {{key}} 占位符。"""
    if isinstance(obj, str):
        out = obj
        for key, val in replacements.items():
            out = out.replace(f"{{{{{key}}}}}", val)
        return out
    if isinstance(obj, dict):
        return {k: apply_placeholders(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [apply_placeholders(v, replacements) for v in obj]
    return obj


def load_config(
    path: str | Path | None = None,
    location_theme: str | None = None,
    title_text: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """加载 YAML 配置并应用场景预设与占位符。"""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    replacements = dict(DEFAULT_PLACEHOLDERS)
    if location_theme is not None:
        replacements["location_theme"] = location_theme
    if title_text is not None:
        replacements["title_text"] = title_text

    cfg = apply_placeholders(cfg, replacements)

    theme = cfg.get("location_theme", replacements["location_theme"])
    scenarios = cfg.get("scenarios", {})
    if theme in scenarios:
        preset = scenarios[theme]
        for key, val in preset.items():
            if key == "degradation_thresholds":
                cfg.setdefault("preprocessing", {})["degradation_thresholds"] = val
            elif key in ("activation", "coord_encoding"):
                cfg.setdefault("inr", {})[key] = val
            elif key == "pinn_mode":
                cfg.setdefault("ode", {})["pinn_mode"] = val
            elif key == "pde":
                cfg.setdefault("ode", {})["pde_type"] = val
            else:
                cfg[key] = val

    if "latent_dim" not in cfg or cfg["latent_dim"] is None:
        cfg["latent_dim"] = resolve_latent_dim(float(cfg.get("fiber_length_km", 10)))

    if overrides:
        deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_title_text(cfg: dict) -> str:
    return str(cfg.get("title_text", DEFAULT_PLACEHOLDERS["title_text"]))


def get_location_theme(cfg: dict) -> str:
    return str(cfg.get("location_theme", DEFAULT_PLACEHOLDERS["location_theme"]))
