"""仓库路径约定：code/ 为包根，仓库根为上一级。"""

from __future__ import annotations

from pathlib import Path

# code/ 目录（含 das_inr_ode、config、scripts）
CODE_ROOT = Path(__file__).resolve().parents[1]
# 仓库根（本目录上一级，即 DAS-INR-ODE-Compression/）
REPO_ROOT = CODE_ROOT.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_GEOM = REPO_ROOT / "data" / "geom"
OUTPUTS = REPO_ROOT / "outputs"
DOCS = REPO_ROOT / "docs"


def default_h5(prefer: str = "KKFLS") -> Path:
    """返回 data/raw 下默认 HDF5；优先匹配 prefer 关键字。"""
    files = sorted(DATA_RAW.glob("*.h5"))
    if not files:
        raise FileNotFoundError(
            f"未找到 HDF5。请将 Cook Inlet 数据放到: {DATA_RAW}\n"
            f"或运行: python data/download.py（见 data/ 说明）"
        )
    for f in files:
        if prefer.upper() in f.name.upper():
            return f
    return files[0]
