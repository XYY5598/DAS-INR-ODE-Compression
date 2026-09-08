# DAS-reconstruction 精简参考

来源：Ni et al., 2024 — Wavefield Reconstruction of Distributed Acoustic Sensing（原完整仓已归档至 `_archive/DAS-reconstruction-main`）。

本目录仅保留与 INR 压缩对照相关的脚本：

- `scripts/models.py` — SIREN / RFFN / SHRED
- `scripts/datasets.py`、`utils.py`
- `scripts/siren/`、`scripts/rffn/` — 训练入口示例

训练用 HDF5 请使用仓库根 `data/raw/`，勿再依赖旧路径。
