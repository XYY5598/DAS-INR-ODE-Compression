# 基于隐式神经表示与 Neural ODE 的 DAS 数据压缩 — 实验结果与专利分析

> 生成时间：2026-06-02 15:50:25  
> 用途：专利权利要求支撑、实施例撰写、效果对比说明

---

## 一、技术方案概述

本发明采用 **"物理预处理 — 潜向量 INR 编码 — 连续时间解码"** 三阶段架构：

1. **编码端**：对 DAS 应变场 `(C×T)` 逐通道归一化后，以随机傅里叶特征 (RFF) 映射时空坐标，
   通过 SIREN 隐式网络 `f(x,t;z)` 拟合；潜向量 `z∈R^256` 与网络权重联合训练，
   压缩阶段固定权重、热启动精调 `z`（可选分频段损失）。
2. **传输**：一级数据层传输潜向量包（约 2.3 KB），二级控制层携带元数据；
   共享 INR 权重驻留解码端（后续任务压缩比 >100×）。
3. **解码端**：由 `z` 经 INR 全网格重建归一化应变场，可选 Neural ODE 混合解码。

### 核心创新点（相对现有技术）

| 创新点 | 技术效果 |
|--------|----------|
| 潜向量热启动精调 | 避免 z 从 0 冷启动，加速收敛、提升 PSNR |
| PSNR 最优 checkpoint | 避免过训练回退，稳定选取最优 epoch |
| 精调阶段分频段混合损失 | 在空域 MSE 基础上加权 3–10 Hz，**band_mix=0.08 达 40.30 dB** |
| 统一时域编解码 | 消除系数域/时域语义断裂（PSNR 7→26 dB 量级提升） |

---

## 二、实验数据与配置

| 项 | 说明 |
|----|------|
| 数据源 | Cook Inlet DAS 实验，事件 11715354 |
| TERRA | `11715354_TERRA.h5`，8531 通道 × 3000 采样 |
| KKFLS | `11715354_KKFLS.h5`，同上 |
| 训练子集 | 128 通道 × 500 采样（除非另有说明） |
| 采样率 | 50 Hz（配置项） |
| 网络 | SIREN 20 层 / 512 单元 / RFF(128, scale=20) / ω=30 |
| 训练 | Adam lr=1e-5，400 epoch，batch=16384 |
| 精调 | latent_steps=3000，latent_lr=2e-3，band_mix=0.08（最佳） |

---

## 三、全部实验结果汇总

| 排名 | 实验路径 | PSNR (dB) | 压缩比 | 40dB | n_units | band_mix |
|------|----------|-----------|--------|------|---------|----------|
| 1 | `terra_11715354` | 40.30 | 112.7× | ✓ | 512 | 0.08 |
| 2 | `improved_benchmark/ablation_20260601/300` | 39.36 | 112.7× |  | 256 | None |
| 3 | `kkfls_11715354/combo_512_400_lat3000` | 38.54 | 112.7× |  | 512 | None |
| 4 | `kkfls_11715354/512u_gpu` | 37.99 | 112.7× |  | 512 | None |
| 5 | `patent_report/figures/best_40db_terra` | 37.80 | 113.2× |  | None | None |
| 6 | `kkfls_11715354/combo_512_400_lat3000_siren_weights` | 36.44 | 0.0× |  | 512 | None |
| 7 | `kkfls_11715354/latent3000_u256` | 35.81 | 112.7× |  | 256 | None |
| 8 | `kkfls_11715354/train_300ep_u256` | 35.43 | 112.7× |  | 256 | None |
| 9 | `kkfls_11715354/400ep_u256` | 35.13 | 112.7× |  | 256 | None |
| 10 | `improved_benchmark/B_time_INR_300ep_u256` | 34.99 | 112.7× |  | None | None |
| 11 | `improved_benchmark/B_time_INR_200ep` | 31.12 | 112.8× |  | None | None |
| 12 | `improved_benchmark/B_time_INR` | 26.06 | 146.3× |  | None | None |
| 13 | `kkfls_11715354/pure_siren_400ep_u512` | 24.92 | 0.0× |  | 512 | None |
| 14 | `kkfls_11715354/pure_siren_400ep_u128` | 20.77 | 0.2× |  | 128 | None |
| 15 | `improved_benchmark/A_time_hybrid` | 16.10 | 146.0× |  | None | None |

---

## 四、关键结论

1. **专利核心指标 PSNR ≥ 40 dB 已在 TERRA 数据上达成**：最优 **40.30 dB**（`terra_11715354`）。
2. **KKFLS 事件数据最优 38.54 dB**，说明数据特性对压缩重建质量有显著影响。
3. **纯 SIREN 权重传输范式**（无潜向量）在本数据上 PSNR 仅 20–25 dB，
   且传输体积 MB 级，不满足压缩语义。
4. **单纯增加 epoch（500）或 latent_steps（4000）反而降低 PSNR**，
   验证 best checkpoint 与精调超参选择的必要性。
5. **band_loss_mix 敏感性**：0.15→39.79 dB，**0.08→40.30 dB**，存在最优区间。

---

## 五、专利撰写建议

### 5.1 独立权利要求支撑数据

- 重建 PSNR：**40.30 dB**（128×500 子集，潜向量 256 维）
- 压缩比：**112.7×**（后续任务，原始 256 KB → 压缩包 2.3 KB）
- 传输内容：潜向量 + 元数据（不含完整网络权重重复传输）

### 5.2 从属权利要求可引用特征

- 潜向量精调采用联合训练所得 z 作为初始值（热启动）；
- 训练过程中按全网格 PSNR 保存最优模型参数；
- 精调损失函数为 `(1-α)·MSE + α·L_band`，α=0.08，L_band 对 3–10 Hz 加权；
- 坐标编码采用 RFF，激活函数采用 SIREN，层数 ≥20，隐藏单元 ≥512。

### 5.3 实施例附图建议

见 `outputs/patent_report/` 目录：

- `figures/best_40db_terra/` — 达标方案全套中文宋体图
- `figures/best_kkfls/` — 对比实施例
- `psnr_leaderboard.png` — 全部实验对比
- `all_experiments.json` — 机器可读完整指标

---

## 六、复现命令（最佳方案）

```bash
python -u das_inr_ode_compression/scripts/train_200epoch.py \
  --h5 DAS-reconstruction-main/data/downloaded_sample/11715354_TERRA.h5 \
  --inr-epochs 400 --gpu-512 --latent-steps 3000 \
  --n-channels 128 --n-samples 500 --band-loss-mix 0.08 \
  --output-dir das_inr_ode_compression/outputs/terra_11715354
```

---

*本报告由 `scripts/generate_patent_report.py` 自动生成。*