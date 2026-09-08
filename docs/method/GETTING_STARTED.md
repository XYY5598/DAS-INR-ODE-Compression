# 新手入门：方法与仓库用法

## 一句话架构

```text
原始 DAS (通道×时间)
  → 物理预处理（可选）
  → INR 隐式编码 → 潜向量 z（压缩包）
  → Neural ODE / INR 连续时间解码 → 重建应变场
```

编码端把时空场压成低维 `z`（及可选共享网络权重）；解码端用 `z` 重建任意时刻信号。

## 本仓库怎么读

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | 本文件 | 总览 |
| 2 | `METHOD_OVERVIEW.md` | 模块与伪代码 |
| 3 | `VARIABLES.md` | `{{location_theme}}` / `{{title_text}}` |
| 4 | `../patent_report/PATENT_RESULTS_ANALYSIS.md` | 指标与实施例 |
| 5 | `PHASE1`→`PHASE5` | 问题定位到改进总结（可选） |

## 代码入口（`../../code`）

| 脚本 | 作用 |
|------|------|
| `scripts/quick_validate.py` | 合成数据冒烟 |
| `scripts/validate_real_data.py` | Cook Inlet HDF5 短训+出图 |
| `scripts/train_200epoch.py` | 长训 / 冲刺 PSNR |
| `scripts/train_pure_siren.py` | 纯 SIREN 权重压缩对照 |
| `scripts/compress.py` / `decompress.py` | 压缩包编解码 |
| `config/default.yaml` / `improved.yaml` | 超参 |

默认数据目录：仓库根下 `data/raw/`（由 `das_inr_ode.paths` 解析）。

## 占位符示例

```bash
python scripts/train.py --location-theme 海洋 --title-text "Cook Inlet 验证"
```

```yaml
# config/default.yaml
location_theme: "陆地基础设施"
title_text: "桥梁应变场"
```

## 常见问题

**Q: 找不到 .h5？**  
把文件放到 `data/raw/`，或 `--h5` 指定绝对路径。

**Q: 旧路径 `DAS-reconstruction-main/data` 失效？**  
已改为 `data/raw`；完整旧仓在 `_archive/`。

**Q: 中文图乱码？**  
可视化模块会加载 Windows 宋体 `simsun.ttc`。
