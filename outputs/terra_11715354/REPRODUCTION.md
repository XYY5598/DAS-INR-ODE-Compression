# TERRA 11715354 复现记录 — PSNR 40.30 dB（band_mix=0.08）

> 运行时间：2026-06-02 · PSNR **40.30 dB**（128×500，≥40 dB 达标）

---

## 目录结构（整合后）

```
outputs/terra_11715354/
├── REPRODUCTION.md      # 本复现文档
├── training_log.txt     # 训练日志
├── metrics.json         # 最终指标
├── checkpoint.pt        # INR 权重 + latent
├── packet.bin           # 潜向量压缩包
├── original_norm.npy
├── reconstructed.npy
├── comparison.png       # 中文宋体可视化
├── reconstructed.png
└── waveform.png
```

已删除的中间实验目录：`combo_*`、`combo_v2_band_best_warm`、`finetune_40db/{band005,band010,latlr1e3,auto_ch}` 等（对比数据见下文 §8）。

---

## 1. 一键复现命令

```bash
cd /path/to/DAS-INR-ODE-Compression
# 或 Windows：d:\browserdownload\DAS-INR-ODE-Compression

python -m pip install -r das_inr_ode_compression/requirements.txt

python -u das_inr_ode_compression/scripts/train_200epoch.py \
  --h5 "DAS-reconstruction-main/data/downloaded_sample/11715354_TERRA.h5" \
  --config das_inr_ode_compression/config/improved.yaml \
  --inr-epochs 400 \
  --gpu-512 \
  --latent-steps 3000 \
  --n-layers 20 \
  --n-channels 128 \
  --n-samples 500 \
  --band-loss-mix 0.08 \
  --transmit-mode latent \
  --output-dir das_inr_ode_compression/outputs/terra_11715354
```

---

## 2. 运行环境

| 项 | 值 |
|---|---|
| OS | Linux 5.15 |
| Python | 3.12 (miniconda3) |
| PyTorch | 2.7.0+cu128 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 可用 |
| 中文字体 | AR PL UMing CN（宋体风格，用于可视化） |

---

## 3. 数据

| 项 | 值 |
|---|---|
| 文件 | `DAS-reconstruction-main/data/downloaded_sample/11715354_TERRA.h5` |
| 事件 ID | 11715354 |
| 原始 H5 shape | `(3000, 8531)` → `[samples, channels]` |
| 训练子集 | 通道 0–127，采样 0–499 → `(128, 500)` |
| HDF5 路径 | `/Acquisition/Raw[0]/RawData` |
| 归一化 | 逐通道：去均值 / 峰值 |

---

## 4. 完整训练参数

### 4.1 命令行参数

| 参数 | 值 |
|---|---|
| `--inr-epochs` | 400 |
| `--latent-steps` | 3000 |
| `--n-layers` | 20 |
| `--n-units` | 512（`--gpu-512`） |
| `--n-channels` | 128 |
| `--n-samples` | 500 |
| **`--band-loss-mix`** | **0.08**（关键：从 0.15 微调至 0.08 突破 40 dB） |
| `--latent-lr` | 2.0e-3（默认） |
| `--transmit-mode` | latent |

### 4.2 脚本内 cfg 覆盖

```yaml
decode_domain: time
decode_mode: inr
latent_dim: 256

inr:
  n_layers: 20
  n_units: 512
  activation: siren
  coord_encoding: rff
  rff_features: 128
  rff_scale: 20.0
  siren_omega: 30.0
  lr: 1.0e-5
  epochs: 400
  patience: 400
  batch_size: 16384
  latent_steps: 3000
  latent_lr: 2.0e-3
  reg_lambda: 1.0e-6
  use_band_loss: true
  band_loss_train: false
  band_loss_mix: 0.08      # 精调：92% MSE + 8% 分频段损失
  best_checkpoint: true
  seismic_weight: 8.0
  psnr_eval_every: 20
```

### 4.3 v2 训练流程

1. 联合训练：Adam 优化 INR + z，点级 MSE  
2. Best checkpoint：每 20 epoch 评估 PSNR，保留最优  
3. 潜向量精调：`encode_latent(z_init=联合训练 z)`，3000 步，混合损失（band_mix=0.08）  
4. 压缩：`compress(z_init=训练 z)`  

---

## 5. 训练日志（INR epoch / PSNR）

```
数据: 11715354_TERRA.h5, shape=(128, 500), ch_start=0
配置: epochs=400, latent_steps=3000, band_mix=0.08, n_units=512

  INR epoch   1/400, loss=0.200700
  INR epoch  20/400, loss=0.000727, psnr=31.72 dB
  INR epoch 100/400, loss=0.000537, psnr=33.48 dB
  INR epoch 200/400, loss=0.000277, psnr=36.50 dB
  INR epoch 300/400, loss=0.000220, psnr=37.81 dB
  INR epoch 380/400, loss=0.000113, psnr=40.11 dB
  INR epoch 400/400, loss=0.000259, psnr=36.70 dB
  恢复 best checkpoint: epoch=380, psnr=40.11 dB

==================================================
PSNR:     40.30 dB  (目标 ≥40 dB)
压缩比:   112.7x
40 dB 达标: 是
耗时:     ~9.5 min
==================================================
```

---

## 6. 最终指标（`metrics.json`）

```json
{
  "psnr_db": 40.303,
  "mse": 9.33e-05,
  "rmse": 0.00966,
  "peak_error": 0.0714,
  "compression_ratio": 112.73,
  "meets_psnr_40db": true,
  "train_time_sec": 567.7,
  "train_stats": {
    "best_psnr_db": 40.108,
    "best_epoch": 380,
    "band_loss_mix": 0.08,
    "latent_steps": 3000,
    "n_units": 512
  }
}
```

---

## 7. 输出文件

| 文件 | 说明 |
|---|---|
| `metrics.json` | 完整指标 |
| `checkpoint.pt` | INR 权重 + latent |
| `packet.bin` | 潜向量压缩包 |
| `original_norm.npy` / `reconstructed.npy` | 归一化数据 |
| `comparison.png` | 原/重建/误差（中文宋体） |
| `reconstructed.png` | 重建应变场 |
| `waveform.png` | 中间通道波形 |
| `training_log.txt` | 训练日志 |
| `REPRODUCTION.md` | 本复现文档 |

---

## 8. 历史实验对比（TERRA 128×500）

| 实验 | band_mix | PSNR | 40 dB |
|------|----------|------|-------|
| 改进前 | — | 39.65 dB | 否 |
| v2 基线 | 0.15 | 39.79 dB | 否 |
| band005 | 0.05 | 39.52 dB | 否 |
| band010 | 0.10 | 38.59 dB | 否 |
| latent_lr=1e-3 | 0.15 | 39.35 dB | 否 |
| **band008（本方案）** | **0.08** | **40.30 dB** | **是** |

---

## 9. 重绘可视化（中文宋体）

```bash
python -u das_inr_ode_compression/scripts/redraw_plots.py \
  --dir das_inr_ode_compression/outputs/terra_11715354 \
  --title "TERRA 11715354 — PSNR 40.30 dB"
```

---

## 10. 注意事项

- **`band_loss_train` 必须为 `false`**，否则 PSNR 崩溃。  
- **`band_loss_mix=0.08`** 为当前最优；0.15 为 39.79 dB，0.05 为 39.52 dB。  
- 随机性：RFF、z 初始化导致复现 PSNR 可能有 ±0.1 dB 波动。
