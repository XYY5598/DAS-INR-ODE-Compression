# 基于隐式神经表示与 Neural ODE 的 DAS 数据压缩

> 仓库目录名：`DAS-INR-ODE-Compression`（原 `DAS4Whales-main`，语义对齐本专利实现）

本仓库是专利《一种基于隐式神经表示与神经常微分方程的分布式光纤应变感知数据压缩方法》的**可运行实现**与实验材料，已按新手可读结构整理。

## 目录地图

```text
.
├── README.md                 ← 你在这里
├── code/                     权威实现（Python 包 + 脚本）
├── data/                     Cook Inlet HDF5 与几何
├── docs/
│   ├── patent/               专利原文（docx）
│   ├── method/               方法与阶段分析
│   └── patent_report/        实验结果与专利撰写支撑
├── outputs/                  代表性训练结果与图
├── reference/
│   └── DAS-reconstruction-lite/   SIREN/RFFN 精简参考
└── _archive/                 整理前无关内容（可恢复）
```

## 5 分钟上手

```bash
# 1. 安装依赖
cd code
pip install -r requirements.txt

# 2. 合成数据冒烟（无需 HDF5）
python scripts/quick_validate.py

# 3. 真实数据（默认用 data/raw 下 KKFLS）
python scripts/validate_real_data.py --n-channels 64 --n-samples 200 --epochs 5
# 或显式指定：
python scripts/train_200epoch.py --h5 ../data/raw/11715354_KKFLS.h5 --inr-epochs 50 --n-units 128
```

数据说明见 [`data/README.md`](data/README.md)。

## 推荐阅读顺序

1. [`docs/method/GETTING_STARTED.md`](docs/method/GETTING_STARTED.md) — 架构与占位符
2. [`docs/method/METHOD_OVERVIEW.md`](docs/method/METHOD_OVERVIEW.md) — 方法与伪代码
3. [`docs/patent_report/PATENT_RESULTS_ANALYSIS.md`](docs/patent_report/PATENT_RESULTS_ANALYSIS.md) — 实验结果
4. [`docs/patent/`](docs/patent/) — 专利原文 docx
5. [`docs/method/PHASE5_FINAL_REPORT.md`](docs/method/PHASE5_FINAL_REPORT.md) — 改进总结

## 可替换变量

- `{{location_theme}}`：场景（海洋 / 陆地基础设施 / 地质活动监测）
- `{{title_text}}`：图表与报告标题  

详见 [`docs/method/VARIABLES.md`](docs/method/VARIABLES.md)。

## 参考代码

[`reference/DAS-reconstruction-lite`](reference/DAS-reconstruction-lite) 保留 Ni et al. 2024 中 SIREN/RFFN 风格实现，供对照；完整原仓在 `_archive/DAS-reconstruction-main/`。

## 备份说明

与本专利主线无关的 DASPack、去噪、旧嵌套包等已移至 [`_archive/`](_archive/README.md)，需要时可移回。
